from typing import Any, Dict, Union
import torch
import numpy as np

DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
# TODO: verify the correctness
Params = Dict[str, torch.nn.Parameter]