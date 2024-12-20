from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, create_model, Field
from sqlalchemy.orm import Session

import api.models as models
from api.clients.llm import LlmApiModels
from api.errors import SchemaError
from api.metrics import metric_registry
from api.utils import build_param_grid


#
# Custom BaseModel
#
class EgBaseModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
        json_encoders={
            # np.nan is not serializable !
            float: lambda v: None if np.isnan(v) else v,
        },
        extra="forbid",  # Do not allow unknow field
    )

    def recurse_table_init(self, db: Session) -> dict:
        obj = self.model_dump()
        for k, v in obj.items():
            sub_schema = getattr(self, k)
            if hasattr(sub_schema, "to_table_init"):
                obj[k] = sub_schema.to_table_init(db)
            elif isinstance(sub_schema, list):
                obj[k] = [
                    o.recurse_table_init(db) if isinstance(o, BaseModel) else o for o in sub_schema
                ]

        return obj

    def to_table_init(self, db: Session):
        return self.model_dump()


#
# Enum
#

MetricEnum = Enum(
    "MetricEnum", {name: name for name in metric_registry.get_metric_names()}, type=str
)


class ExperimentStatus(str, Enum):
    pending = "pending"
    running_answers = "running_answers"
    running_metrics = "running_metrics"
    finished = "finished"


class MetricStatus(str, Enum):
    pending = "pending"
    running = "running"
    finished = "finished"


#
# Dataset
#


class DatasetBase(EgBaseModel):
    name: str
    readme: str


class DatasetCreate(DatasetBase):
    df: str  # from_json

    def to_table_init(self, db: Session) -> dict:
        obj = self.recurse_table_init(db)

        # Handle dataframe
        try:
            df = pd.read_json(StringIO(self.df))
        except ValueError:
            raise SchemaError("'df' should be a readable dataframe. Use df.to_json()...")

        has_query = "query" in df.columns
        has_output = "output" in df.columns
        has_output_true = "output_true" in df.columns

        if not (has_query or has_output):
            raise SchemaError("Your dataset needs at least a column 'query' or 'ouput'.")

        return {
            "has_query": has_query,
            "has_output": has_output,
            "has_output_true": has_output_true,
            "size": len(df),
            "columns": list(df.columns),
            **obj,
        }


class Dataset(DatasetBase):
    id: int
    created_at: datetime
    has_query: bool
    has_output: bool
    has_output_true: bool
    size: int
    columns: list[str]


class DatasetFull(DatasetBase):
    df: str  # from_json


DatasetUpdate = create_model(
    "DatasetUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in Dataset.__fields__.items()
    },
    __base__=Dataset,
)


# Do not allow to update the content of a dataset to prevent breaking
# expriment alignement with num_line
DatasetPatch = create_model(
    "DatasetPatch",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in Dataset.__fields__.items()
    },
    __base__=DatasetBase,
)

#
# Model
#


class ModelBase(EgBaseModel):
    name: str
    base_url: str
    prompt_system: str | None = None
    sampling_params: dict | None = None
    extra_params: dict | None = None


class ModelCreate(ModelBase):
    api_key: str


class Model(ModelBase):
    id: int


class ModelWithKeys(Model):
    api_key: str


#
# Result
#


class Answer(EgBaseModel):
    id: int
    created_at: datetime
    answer: str | None
    num_line: int
    error_msg: str | None
    execution_time: int | None


class Observation(EgBaseModel):
    id: int
    created_at: datetime
    score: float | None
    observation: str | dict | list | None  # json
    num_line: int
    error_msg: str | None
    execution_time: int | None


class ResultBase(EgBaseModel):
    metric_name: MetricEnum


class ResultCreate(ResultBase):
    experiment_id: int | None = None

    def to_table_init(self, db):
        obj = self.recurse_table_init(db)
        return {"metric_status": "pending", **obj}


class Result(ResultBase):
    id: int
    created_at: datetime
    observation_table: list[Observation]
    num_try: int
    num_success: int
    experiment_id: int
    metric_status: MetricStatus


ResultUpdate = create_model(
    "ResultUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in Result.__fields__.items()
    },
    __base__=Result,
)

#
# Experiment
#


class ExperimentBase(EgBaseModel):
    name: str
    readme: str | None = None
    experiment_set_id: int | None = None
    judge_model: Literal[*LlmApiModels.openai] | None = None


class ExperimentCreate(ExperimentBase):
    metrics: list[MetricEnum]
    dataset: DatasetCreate | str
    model: ModelCreate | None = None

    def to_table_init(self, db: Session) -> dict:
        obj = self.recurse_table_init(db)

        # Handle Dataset
        if isinstance(self.dataset, str):
            dataset = db.query(models.Dataset).filter_by(name=self.dataset).first()
            if not dataset:
                raise SchemaError("Dataset not found")
        else:
            dataset = models.Dataset(**obj["dataset"])
        obj["dataset"] = dataset
        if dataset.has_output:
            df = pd.read_json(StringIO(dataset))
            obj["num_try"] = dataset.size
            obj["num_success"] = df["output"].count()

        # Handle Results
        results = []
        for metric_name in self.metrics:
            results.append(ResultCreate(metric_name=metric_name).to_table_init(db))
        obj["results"] = results

        # Validate Model and metric compatibility
        mr = metric_registry
        needs_query = any("query" in mr.get_metric(m).require for m in self.metrics)
        needs_output = any("output" in mr.get_metric(m).require for m in self.metrics)
        needs_output_true = any("output_true" in mr.get_metric(m).require for m in self.metrics)
        if needs_query and not dataset.has_query:
            raise SchemaError("You need to provide a query for this metric. ")
        if needs_output and not self.model and not dataset.has_output:
            raise SchemaError(
                "You need to provide an answer for this metric. "
                "Either set a model to generate it or provide a dataset with the 'output' field."
            )
        if needs_output and not dataset.has_output and not dataset.has_query:
            raise SchemaError(
                "You need to provide an answer for this metric. "
                "Either provide a dataset with the 'query' field to generate the answer or with an 'output' field if have generated it yourself."
            )

        if needs_output_true and not dataset.has_output_true:
            raise SchemaError(
                "You need to provide a ground truth for this metric. "
                "Your dataset needs to have an 'output_true' field."
            )

        if dataset.has_output and self.model:
            raise SchemaError(
                "You can't give at the same time a model and a dataset with an answer ('output' column). "
                "Gives either one or the other."
            )

        return {
            "experiment_status": "pending",
            **obj,
        }


class Experiment(ExperimentBase):
    id: int
    created_at: datetime
    experiment_status: ExperimentStatus
    num_try: int = Field(description="How many output/answers were attempted to be generated.")
    num_success: int = Field(description="How many output/answers were successfully generated.")
    num_observation_try: int = Field(
        description="How many metric observations were attempted to be generated."
    )
    num_observation_success: int = Field(
        description="How many metric observations were successfully generated."
    )

    dataset: Dataset
    model: Model | None


class ExperimentWithResults(Experiment):
    results: list[Result] | None


class ExperimentWithAnswers(Experiment):
    answers: list[Answer] | None


class ExperimentFull(Experiment):
    answers: list[Answer] | None = None
    results: list[Result] | None = None


class ExperimentFullWithDataset(ExperimentFull):
    dataset: DatasetFull | None = None


# For the special `metrics` input
class ExperimentExtra(Experiment, ExperimentCreate):
    pass


ExperimentUpdate = create_model(
    "ExperimentUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in ExperimentExtra.__fields__.items()
    },
    __base__=ExperimentExtra,
)


class ExperimentPatch(ExperimentUpdate):
    rerun_answers: bool = False
    rerun_metrics: bool = False  # metrics are automatically run after answers atm


#
# Experiment Set
#


class GridCV(BaseModel):
    common_params: dict[str, Any]
    grid_params: dict[str, list[Any]]
    repeat: int = 1


class ExperimentSetBase(EgBaseModel):
    name: str
    readme: str


class ExperimentSetCreate(ExperimentSetBase):
    experiments: list[ExperimentCreate] | None = None
    cv: GridCV | None = None

    def to_table_init(self, db: Session) -> dict:
        obj = self.recurse_table_init(db)

        if self.experiments is not None and self.cv is not None:
            raise SchemaError("Please, give either an expriments or a cv parameter but not both.")

        # Handle Experiments
        if self.cv is not None:
            experiments = []
            i = 0
            for experiment in build_param_grid(self.cv.common_params, self.cv.grid_params):
                for _ in range(self.cv.repeat):
                    experiment["name"] = f"{self.name}__{i}"
                    experiments.append(ExperimentCreate(**experiment).to_table_init(db))
                    i += 1
            obj["experiments"] = experiments
        elif self.experiments is None:
            obj.pop("experiments")

        return obj


class ExperimentSet(ExperimentSetBase):
    id: int
    created_at: datetime
    experiments: list[Experiment] | None


# For the special `metrics` input
class ExperimentSetExtra(ExperimentSet, ExperimentSetCreate):
    pass


ExperimentSetUpdate = create_model(
    "ExperimentSetUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in ExperimentSet.__fields__.items()
    },
    __base__=ExperimentSetExtra,
)


class ExperimentSetPatch(ExperimentSetUpdate):
    pass


class RetryRuns(EgBaseModel):
    # List of experiment to retry
    experiment_ids: list[int]
    # List of results/metrics to retry
    result_ids: list[int]



#
# LeaderBoard
#


class LeaderboardEntry(EgBaseModel):
    experiment_id: int
    model_name: str
    dataset_name: str
    main_metric_score: float | None
    other_metrics: dict[str, float | None]

class Leaderboard(EgBaseModel):
    entries: list[LeaderboardEntry]


