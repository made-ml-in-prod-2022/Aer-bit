import yaml
from typing import Optional
from dataclasses import dataclass, field

from marshmallow_dataclass import class_schema


@dataclass
class SplitParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)
    
    
@dataclass
class TrainParams:
    n_estimators: int = field(default=100)
    lr: float = field(default=0.01)
    max_depth: int = field(default=10)
    random_state: int = field(default=42)


@dataclass
class PipelineParameters:
    input_data_path: str
    train_data_path: str
    test_data_path: str
    output_model_path: str
    output_data_path: str
    splitting_params: SplitParams
    train_params: TrainParams
    feature_cols: list
    target_col: list
    
    
PipelineParametersSchema = class_schema(PipelineParameters)


def get_pipeline_parameters(path: str) -> PipelineParameters:
    with open(path, 'r') as input_stream:
        schema = PipelineParametersSchema()
        return schema.load(yaml.safe_load(input_stream))