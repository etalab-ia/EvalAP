from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import RelationshipProperty, class_mapper, relationship
from sqlalchemy.sql import func

from api.schemas import EgBaseModel

Base = declarative_base()

# By convention, we reserve the String type for Enum defined in the schema scope.


def is_relationship(model, attribute_name):
    mapper = class_mapper(model)
    property = mapper.attrs.get(attribute_name)
    return isinstance(property, RelationshipProperty)


def create_object_from_dict(db, model, data):
    """Create a SQL object from a nested dictionary with relationship"""
    if isinstance(data, Base):
        return data

    # Separate relationships from simple attributes
    relationships = {}
    attributes = {}
    for k, v in data.items():
        if hasattr(model, k):
            if is_relationship(model, k):
                if v is None:
                    continue
                relationships[k] = v
            else:
                attributes[k] = v

    # Create the main object
    obj = model(**attributes)

    # Handle relationships
    for rel_name, rel_data in relationships.items():
        rel_model = getattr(model, rel_name).property.mapper.class_
        if isinstance(rel_data, Base):  # Relationship is already a db model
            setattr(obj, rel_name, rel_data)
        elif isinstance(rel_data, list):  # Handle one-to-many or many-to-many
            related_objs = [create_object_from_dict(db, rel_model, item) for item in rel_data]
            setattr(obj, rel_name, related_objs)
        elif isinstance(rel_data, (dict, EgBaseModel)):  # Handle one-to-one or many-to-one
            if isinstance(rel_data, EgBaseModel):
                rel_data = vars(rel_data)
            related_obj = create_object_from_dict(db, rel_model, rel_data)
            setattr(obj, rel_name, related_obj)
        else:
            raise NotImplementedError(
                f"Unknonw type for relationtioship {rel_name}:{type(rel_data)}"
            )

    return obj


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    name = Column(Text, unique=True)
    df = Column(JSON)  # df
    columns = Column(JSON)  # list[str]
    has_query = Column(Boolean)
    has_output = Column(Boolean)
    size = Column(Integer)
    readme = Column(Text)
    default_metric = Column(Text)


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(Text)
    base_url = Column(Text)
    api_key = Column(Text)
    prompt_system = Column(Text)
    # prompt_template = Column(Text) # rag, composition, multiagents ?
    sampling_params = Column(JSON)  # dict
    extra_params = Column(JSON)  # dict


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    metric_name = Column(String)
    num_try = Column(Integer, default=0)
    num_success = Column(Integer, default=0)
    metric_status = Column(String)

    # One
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    experiment = relationship("Experiment", back_populates="results")
    # Many
    observation_table = relationship(
        "ObservationTable", back_populates="result", cascade="all, delete-orphan"
    )


class ObservationTable(Base):
    __tablename__ = "observation_table"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    score = Column(Float)
    observation = Column(JSON)
    num_line = Column(Integer)
    error_msg = Column(Text)
    execution_time = Column(Integer)

    # One
    result_id = Column(Integer, ForeignKey("results.id"))
    result = relationship("Result", back_populates="observation_table")

    __table_args__ = (
        UniqueConstraint("num_line", "result_id", name="_metric_num_line_unique_constraint"),
    )


class Answer(Base):
    __tablename__ = "answers"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    answer = Column(Text)
    num_line = Column(Integer)
    error_msg = Column(Text)
    execution_time = Column(Integer)
    nb_tokens_prompt = Column(Integer)
    nb_tokens_completion = Column(Integer)
    retrieval_context = Column(JSON)  # list[str]

    # One
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    experiment = relationship("Experiment", back_populates="answers")

    __table_args__ = (
        UniqueConstraint("num_line", "experiment_id", name="_answer_num_line_unique_constraint"),
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(Text)
    readme = Column(Text)
    is_archived = Column(Boolean, default=False)  # do not allow user to remove without IAM.
    created_at = Column(DateTime, server_default=func.now())
    experiment_status = Column(String)
    judge_model = Column(String)
    num_try = Column(Integer, default=0)
    num_success = Column(Integer, default=0)

    @property
    def num_observation_try(self):
        return sum(result.num_try for result in self.results)

    @property
    def num_observation_success(self):
        return sum(result.num_success for result in self.results)

    @property
    def num_metrics(self):
        return len(self.results)

    # One
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    dataset = relationship("Dataset")
    model_id = Column(Integer, ForeignKey("models.id"))
    model = relationship("Model", cascade="all, delete-orphan", single_parent=True)
    experiment_set_id = Column(Integer, ForeignKey("experiment_sets.id"))
    experiment_set = relationship("ExperimentSet", back_populates="experiments")
    # Many
    results = relationship("Result", back_populates="experiment")  # len == #metrics
    answers = relationship("Answer", back_populates="experiment")  # len == #dataset.df

    __table_args__ = (
        UniqueConstraint("experiment_set_id", "name", name="_expset_name_unique_constraint"),
    )


class ExperimentSet(Base):
    __tablename__ = "experiment_sets"

    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)
    readme = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    is_archived = Column(Boolean, default=False)  # do not allow user to remove without IAM.

    # Many
    experiments = relationship("Experiment", back_populates="experiment_set")
