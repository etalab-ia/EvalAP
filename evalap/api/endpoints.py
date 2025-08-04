import os
import re
import shutil
import yaml
from datetime import datetime

import aiofiles
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import evalap.api.crud as crud
import evalap.api.schemas as schemas
from evalap.api.db import get_db
from evalap.api.errors import CustomIntegrityError, SchemaError
from evalap.api.metrics import Metric, metric_registry
from evalap.api.security import admin_only
from evalap.clients import LlmClient, MCPBridgeClient, multi_step_generate
from evalap.logger import logger
from evalap.runners import dispatch_retries, dispatch_tasks

router = APIRouter()

#TODO: change path
CONFIG_FILE_PATH = "/Users/audrey_clevy/code/evalap/evalap/config/products/product_config.yml"

def _needs_output(db_exp):
    return not db_exp.model.has_raw_output and any(
        "output" in metric_registry.get_metric(r.metric_name).require for r in db_exp.results
    )


#
# Datasets
#


@router.post("/dataset", response_model=schemas.Dataset, tags=["datasets"])
def create_dataset(dataset: schemas.DatasetCreate, db: Session = Depends(get_db)):
    try:
        db_dataset = crud.create_dataset(db, dataset)
        return db_dataset
    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get("/datasets", response_model=list[schemas.Dataset], tags=["datasets"])
def read_datasets(db: Session = Depends(get_db)):
    return crud.get_datasets(db)


@router.get("/dataset/{id}", response_model=schemas.Dataset | schemas.DatasetFull, tags=["datasets"])
def read_dataset(id: int, with_df: bool = False, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if with_df:
        return schemas.DatasetFull.model_validate(dataset)
    else:
        return schemas.Dataset.model_validate(dataset)


@router.get("/dataset", response_model=schemas.Dataset | schemas.DatasetFull, tags=["datasets"])
def read_dataset_by_query(
    id: str | None = None,
    name: str | None = None,
    with_df: bool = False,
    db: Session = Depends(get_db),
):
    dataset = None
    if name:
        dataset = crud.get_dataset_by_name(db, name)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset with given name not found")
    if id:
        dataset = crud.get_dataset(db, id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset:
        if with_df:
            return schemas.DatasetFull.model_validate(dataset)
        else:
            return schemas.Dataset.model_validate(dataset)

    raise HTTPException(status_code=400, detail="No query parameters provided")


@router.patch("/dataset/{id}", response_model=schemas.Dataset, tags=["datasets"])
def patch_dataset(id: int, dataset_patch: schemas.DatasetPatch, db: Session = Depends(get_db)):
    db_dataset = crud.update_dataset(db, id, dataset_patch)
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return db_dataset


@router.delete(
    "/dataset/{id}",
    tags=["datasets"],
)
def delete_dataset(id: int, db: Session = Depends(get_db), admin_check=Depends(admin_only)):
    try:
        if not crud.remove_dataset(db, id):
            raise HTTPException(status_code=404, detail="Dataset not found")
        return "ok"
    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


# @TODO write /dataset/{id}/download_parquet
@router.post("/dataset/{id}/upload_parquet", response_model=schemas.Dataset, tags=["datasets"])
async def upload_parquet_dataset(id: int, request: Request, db: Session = Depends(get_db)):
    """
    Endpoint to handle streaming upload of Parquet data for a specific dataset.

    Args:
        dataset_id: ID of the dataset in the database
        request: The streaming request containing Parquet data
        db: Database session
    """
    # Fetch the dataset from the database
    dataset = crud.get_dataset(db, id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create the directory structure if it doesn't exist
    data_dir = "/data/datasets"
    os.makedirs(data_dir, exist_ok=True)

    # Define file paths
    temp_file_path = f"{data_dir}/temp_{dataset.name}.parquet.tmp"
    final_file_path = f"{data_dir}/{dataset.name}.parquet"

    total_bytes = 0  # Track upload size for updating the database
    try:
        # Open a file to write the chunks
        async with aiofiles.open(temp_file_path, "wb") as f:
            # Process the incoming stream in chunks
            async for chunk in request.stream():
                if chunk:
                    # Write the chunk to the file
                    await f.write(chunk)
                    total_bytes += len(chunk)

    except Exception as e:
        # Handle any other errors during the upload process
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # Validate the uploaded file is a valid Parquet file
    try:
        # Try to open the file with pyarrow to verify it's valid
        pf = pq.ParquetFile(temp_file_path)
        num_rows = pf.metadata.num_rows
        column_names = pf.schema_arrow.names

        # Move the temporary file to the final destination
        shutil.move(temp_file_path, final_file_path)

        # Update the dataset record in the database
        dataset.parquet_path = final_file_path
        dataset.parquet_size = num_rows
        dataset.parquet_columns = column_names
        dataset.parquet_byte_size = total_bytes
        db.commit()
        db.refresh(dataset)

        return dataset
    except Exception as e:
        # If validation fails, clean up and return an error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=f"Invalid Parquet file: {str(e)}")


#
# Metrics
#


@router.get("/metrics", response_model=list[Metric], tags=["metrics"])
def read_metrics(db: Session = Depends(get_db)):
    return crud.get_metrics(db)


#
# Experiments
#


@router.post(
    "/experiment",
    response_model=schemas.Experiment,
    description="Launch an experiment. If a model is given, it will be use to generate the model output (answer), otherwise it will use the `output` column of the given dataset.",
    tags=["experiments"],
)
def create_experiment(experiment: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    try:
        db_exp = crud.create_experiment(db, experiment)
        if _needs_output(db_exp):
            dispatch_tasks(db, db_exp, "answers")
        else:
            dispatch_tasks(db, db_exp, "observations")

        return db_exp

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.patch(
    "/experiment/{id}",
    response_model=schemas.Experiment,
    description="Update an experiment. The given metrics will be added (or rerun) to the existing results for this experiments. Use rerun_answers if want to re-generate the answers/output.",
    tags=["experiments"],
)
def patch_experiment(id: int, experiment_patch: schemas.ExperimentPatch, db: Session = Depends(get_db)):
    db_exp = crud.update_experiment(db, id, experiment_patch)
    if db_exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    elif db_exp.experiment_status not in [
        schemas.ExperimentStatus.pending,
        schemas.ExperimentStatus.finished,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is running ({db_exp.experiment_status}), please try again later",
        )

    # Parameters you can not patch
    if experiment_patch.judge_model and experiment_patch.judge_model != db_exp.judge_model:
        raise HTTPException(
            status_code=400,
            detail="You cannot patch the judge_model in an experiment.",
        )

    # Rerun experiment
    # --
    # Initialize metric results
    for metric in experiment_patch.metrics or []:
        result = crud.get_result(db, experiment_id=db_exp.id, metric_name=metric)
        if result:
            crud.update_result(db, result.id, dict(metric_status="pending"))
        else:
            result = schemas.ResultCreate(experiment_id=db_exp.id, metric_name=metric)
            crud.create_result(db, result)
    # Dispatch tasks
    if experiment_patch.rerun_answers and _needs_output(db_exp):
        dispatch_tasks(db, db_exp, "answers")
    else:
        dispatch_tasks(db, db_exp, "observations")

    return db_exp


@router.delete(
    "/experiment/{id}",
    tags=["experiments"],
)
def delete_experiment(
    id: int,
    db: Session = Depends(get_db),
    admin_check=Depends(admin_only),
):
    if not crud.remove_experiment(db, id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return "ok"


@router.get(
    "/experiment/{id}",
    response_model=schemas.ExperimentRO
    | schemas.ExperimentWithResults
    | schemas.ExperimentWithAnswers
    | schemas.ExperimentFullWithEco
    | schemas.ExperimentFull
    | schemas.ExperimentFullWithDataset,
    tags=["experiments"],
)
def read_experiment(
    id: int,
    with_results: bool = False,
    with_answers: bool = False,
    with_dataset: bool = False,
    with_eco: bool = False,
    db: Session = Depends(get_db),
):
    experiment = crud.get_experiment(db, id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if with_dataset:
        return schemas.ExperimentFullWithDataset.model_validate(experiment)
    elif with_eco:
        return schemas.ExperimentFullWithEco.model_validate(experiment)
    elif with_answers and with_results:
        return schemas.ExperimentFull.model_validate(experiment)
    elif with_results:
        return schemas.ExperimentWithResults.model_validate(experiment)
    elif with_answers:
        return schemas.ExperimentWithAnswers.model_validate(experiment)
    else:
        return schemas.ExperimentRO.model_validate(experiment)


@router.get(
    "/experiments",
    response_model=list[schemas.ExperimentWithResults],
    tags=["experiments"],
)
def read_experiments(
    skip: int = 0,
    limit: int = 100,
    backward: bool = True,
    set_id: int | None = None,
    orphan: bool = True,
    db: Session = Depends(get_db),
):
    experiments = crud.get_experiments(
        db, skip=skip, limit=limit, backward=backward, set_id=set_id, orphan=orphan
    )

    if not experiments:
        raise HTTPException(status_code=404, detail="No experiments found")

    return experiments


#
# Experiment Sets
#


@router.post(
    "/experiment_set",
    response_model=schemas.ExperimentSet,
    tags=["experiment_set"],
)
def create_experimentset(experimentset: schemas.ExperimentSetCreate, db: Session = Depends(get_db)):
    try:
        db_expset = crud.create_experimentset(db, experimentset)
        for db_exp in db_expset.experiments:
            if _needs_output(db_exp):
                dispatch_tasks(db, db_exp, "answers")
            else:
                dispatch_tasks(db, db_exp, "observations")

        return db_expset
    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.patch(
    "/experiment_set/{id}",
    response_model=schemas.ExperimentSet,
    description="Update an experimentset: New experiments will be added to the runner queue.",
    tags=["experiment_set"],
)
def patch_experimentset(
    id: int, experimentset_patch: schemas.ExperimentSetPatch, db: Session = Depends(get_db)
):
    db_expset = crud.update_experimentset(db, id, experimentset_patch)
    if db_expset is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        expset = experimentset_patch.to_table_init(db)
        # Check the judge_model unicity
        # --
        judge = next((e.judge_model for e in db_expset.experiments if e.judge_model), None)
        new_judge = next(
            (e["judge_model"] for e in (expset.get("experiments") or []) if e.get("judge_model")), None
        )
        # Judge and new_judge must be defined as we can have some experiments that won't use llm-as-a-judge model.
        if judge and new_judge and judge != new_judge:
            raise HTTPException(
                status_code=400,
                detail="You cannot patch the judge_model in an experiment.",
            )

        for experiment in expset.get("experiments") or []:
            experiment["experiment_set_id"] = id
            # Respect the unique constraint for auto-naming experiment !
            # -> add an increment suffix to the experiment name
            if re.search(r"__\d+$", experiment["name"]):
                parts = experiment["name"].split("__")
                parts[-1] = str(int(parts[-1]) + len(db_expset.experiments))
                if parts[0] == "None":
                    parts[0] = db_expset.name
                experiment["name"] = "__".join(parts)
            db_exp = crud.create_experiment(db, experiment)
            if _needs_output(db_exp):
                dispatch_tasks(db, db_exp, "answers")
            else:
                dispatch_tasks(db, db_exp, "observations")

        return db_expset

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/experiment_sets",
    response_model=list[schemas.ExperimentSetRO],
    tags=["experiment_set"],
)
def read_experimentsets(skip: int = 0, limit: int = 100, backward: bool = True, db: Session = Depends(get_db)):
    experimentsets = crud.get_experimentsets(db, skip=skip, limit=limit, backward=backward)
    if experimentsets is None:
        raise HTTPException(status_code=404, detail="ExperimentSets not found")
    return experimentsets
    # return [schemas.ExperimentSet.model_validate(x) for x in experimentsets]


@router.get(
    "/experiment_set/{id}",
    response_model=schemas.ExperimentSetRO,
    tags=["experiment_set"],
)
def read_experimentset(id: int, db: Session = Depends(get_db)):
    experimentset = crud.get_experimentset(db, id)
    if experimentset is None:
        raise HTTPException(status_code=404, detail="ExperimentSet not found")
    return experimentset


@router.delete(
    "/experiment_set/{id}",
    tags=["experiment_set"],
)
def delete_experimentset(id: int, db: Session = Depends(get_db), admin_check=Depends(admin_only)):
    if not crud.remove_experimentset(db, id):
        raise HTTPException(status_code=404, detail="ExperimentSet not found")
    return "ok"


@router.post(
    "/retry/experiment_set/{id}",
    response_model=schemas.RetryRuns,
    description="Re-run failed runs and continue unfinished answers or metrics.",
    tags=["experiment_set"],
)
def retry_runs(
    id: int,
    force: bool = Query(
        default=False,
        description="Force retry of all unfinished runs, by resetting their status to pending. <!> Warning this can cause incoherent num_try/num_success value if another runner work on the same experiments.",
    ),
    db: Session = Depends(get_db),
):
    experimentset = crud.get_experimentset(db, id)
    if experimentset is None:
        raise HTTPException(status_code=404, detail="ExperimentSet not found")

    rr = schemas.RetryRuns(
        experiment_ids=[], result_ids=[], unfinished_experiment_ids=[], unfinished_result_ids=[]
    )

    # Retry failed runs (answers and metrics)
    for exp in experimentset.experiments:
        if exp.experiment_status != "finished" and not force:
            continue

        if exp.num_try != exp.num_success and _needs_output(exp):
            rr.experiment_ids += [exp.id]
            continue

        for result in exp.results:
            if result.metric_status != "finished" and not force:
                continue

            if result.num_try != result.num_success:
                rr.result_ids += [result.id]
                if force:
                    crud.update_result(db, result.id, dict(metric_status="finished"))

    # Unfinished business
    for exp in experimentset.experiments:
        if exp.experiment_status != "finished" and not force:
            continue

        expected_output_len = exp.dataset.parquet_size if exp.with_vision else exp.dataset.size

        actual_output_len = exp.num_try
        if actual_output_len != expected_output_len:
            rr.unfinished_experiment_ids += [exp.id]
            continue

        for result in exp.results:
            if result.metric_status != "finished" and not force:
                continue

            actual_output_len = result.num_try
            if actual_output_len != expected_output_len:
                rr.unfinished_result_ids += [result.id]
                if force:
                    crud.update_result(db, result.id, dict(metric_status="finished"))

    if (
        force and len(set(rr.experiment_ids + rr.result_ids + rr.unfinished_experiment_ids + rr.unfinished_result_ids)) == 0
    ):  # fmt: skip
        for exp in experimentset.experiments:
            crud.update_experiment(db, exp.id, dict(experiment_status="finished"))

    dispatch_retries(db, rr)
    return rr


#
# LeaderBoard
#


@router.get("/leaderboard", response_model=schemas.Leaderboard, tags=["leaderboard"])
def read_leaderboard(
    metric_name: str = "judge_notator",
    dataset_name: str = None,
    judge_model: str = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return crud.get_leaderboard(
        db, metric_name=metric_name, dataset_name=dataset_name, judge_model=judge_model, limit=limit
    )


#
# Ops
#


@router.get("/ops_metrics", response_model=schemas.OpsMetrics, tags=["ops"])
def read_ops_metrics(db: Session = Depends(get_db)):
    return crud.get_ops_metrics(db)


@router.get("/ops_eco", response_model=schemas.OpsEcoGlobal, tags=["ops"])
def read_ops_eco(db: Session = Depends(get_db)):
    answers_data = crud.get_ops_eco_answers(db)
    observations_data = crud.get_ops_eco_observation_table(db)
    return {
        "answers": answers_data,
        "observation_table": observations_data,
    }


#
# Config Products
#

@router.get("/config", response_model=dict, tags=["config"])
def read_config():
    try:
        with open(CONFIG_FILE_PATH, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Config file not found")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing config file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/config", response_model=dict, tags=["config"])
def write_config(config: dict, admin_check=Depends(admin_only)):
    try:
        with open(CONFIG_FILE_PATH, 'w') as file:
            yaml.safe_dump(config, file)
        return {"status": "success", "detail": "Config file updated"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Config file not found")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Error writing to config file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


#
# Generate
#


class GenerateInput(BaseModel):
    model: str
    query: str
    system_prompt: str | None = None
    sampling_params: dict | None = None
    extra_params: dict | None = None


def _generate(input: GenerateInput):
    # MCP Bridge client initalization
    try:
        mcp_bridge = MCPBridgeClient()
    except Exception as e:
        logger.warning(
            "MCP bridge is not responding, MCP will not be used."
            f"Reason: {e}"
        )  # fmt: skip
        mcp_bridge = None

    # Build Smpling params
    sampling_params_plus = (input.sampling_params or {}) | (input.extra_params or {})

    # Build tools input
    _tools = sampling_params_plus.pop("_tools_", None)
    if _tools and mcp_bridge:
        tools = mcp_bridge.tools2openai(_tools)
        sampling_params_plus["tools"] = tools

    # Build messages
    messages = [{"role": "user", "content": input.query}]
    if input.system_prompt:
        messages = [{"role": "system", "content": input.system_prompt}] + messages

    _aiclient = LlmClient()
    url, headers = _aiclient.get_url_and_headers(input.model)
    api_key = headers["Authorization"].split()[-1]
    result, steps = multi_step_generate(
        model_base_url=url,
        model_api_key=api_key,
        model_name=input.model,
        messages=messages,
        sampling_params=sampling_params_plus,
        mcp_bridge=mcp_bridge,
    )

    return result, steps


@router.post("/generate", response_model=schemas.Answer, tags=["generate"])
async def generate(input: GenerateInput, db: Session = Depends(get_db)):
    answer = None
    think = None
    error_msg = None
    result, steps = _generate(input)
    try:
        answer = result.choices[0].message.content
        if answer:
            think, tag, answer = answer.partition("</think>")
            if tag:
                think = (think + tag).strip()
            else:
                answer = think.strip()
                think = None
    except Exception as e:
        error_msg = str(e)

    return schemas.Answer(
        id=-1,
        created_at=datetime.now(),
        answer=answer,
        think=think,
        num_line=0,
        error_msg=error_msg,
        execution_time=-1,
        tool_steps=steps,
    )


#
# LOCUST
#


@router.post(
    "/locust",
    response_model=schemas.LocustRun,
    description="""Save a locust run.

    To format the locust CSV as a dataframe, here is how you must convert it to a dataframe:
    ```
    import pandas as pd
    import requests
    stats_df = pd.read_csv("stats.csv").to_json()

    # Then you can just pass the data in the POST request along the other parameters.
    ```
    """,
    tags=["locust"],
)
def create_locustrun(run: schemas.LocustRunCreate, db: Session = Depends(get_db)):
    try:
        db_run = crud.create_locustrun(db, run)
        return db_run

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/locust/runs",
    response_model=list[schemas.LocustRun],
    description="Get the list of locust runs",
    tags=["locust"],
)
def get_locustruns(skip: int = 0, limit: int = 100, backward: bool = True, db: Session = Depends(get_db)):
    try:
        db_runs = crud.get_locustruns(db, skip=skip, limit=limit, backward=backward)
        return db_runs

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/locust/{id}",
    response_model=schemas.LocustRunFull,
    description="Get locust run with data.",
    tags=["locust"],
)
def get_locustrun(run_id: int, db: Session = Depends(get_db)):
    try:
        db_run = crud.get_locustrun(db, run_id)
        return db_run

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


#
# Load Testing
#


@router.post(
    "/loadtesting",
    response_model=schemas.LoadTesting,
    description="""Save a loadtesting run.

    To format the loadtesting CSV as a dataframe, here is how you must convert it to a dataframe:
    ```
    import pandas as pd
    import requests
    stats_df = pd.read_csv("stats.csv").to_json()

    # Then you can just pass the data in the POST request along the other parameters.
    ```
    """,
    tags=["loadtesting"],
)
def create_loadtesting(run: schemas.LoadTestingCreate, db: Session = Depends(get_db)):
    try:
        db_run = crud.create_loadtesting(db, run)
        return db_run

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/loadtesting/runs",
    response_model=list[schemas.LoadTesting],
    description="Get the list of loadtesting runs",
    tags=["loadtesting"],
)
def get_loadtestings(skip: int = 0, limit: int = 100, backward: bool = True, db: Session = Depends(get_db)):
    try:
        db_runs = crud.get_loadtestings(db, skip=skip, limit=limit, backward=backward)
        return db_runs

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/loadtesting/{run_id}",
    response_model=schemas.LoadTestingFull,
    description="Get loadtesting run with data.",
    tags=["loadtesting"],
)
def get_loadtesting(run_id: int, db: Session = Depends(get_db)):
    try:
        db_run = crud.get_loadtesting(db, run_id)
        return db_run

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.delete(
    "/loadtesting/{run_id}",
    description="Remove loadtesting run.",
    tags=["loadtesting"],
)
def remove_loadtesting(run_id: int, db: Session = Depends(get_db)):
    try:
        if not crud.remove_loadtesting(db, run_id):
            raise HTTPException(status_code=404, detail="LoadTesting not found")
        return "ok"

    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e
