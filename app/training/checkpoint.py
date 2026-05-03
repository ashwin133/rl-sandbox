"""Self-describing checkpoint schema for a discrete-action policy network.

Captures everything needed to reconstruct a trained policy:
    - `algorithm`: which training algorithm produced the weights (so a
      loader can refuse a file written by a different algorithm).
    - `env_id`: which gym env the weights are valid for.
    - `obs_size`, `action_size`, `hidden_size`: architecture sizing the
      loader needs to rebuild PolicyNetwork before applying state_dict.

Algorithm names are free-form strings rather than a Literal-constrained
enum — the schema deliberately makes no commitment to a closed set, so
adding a new algorithm doesn't require editing this module. Canonical
algorithm names are defined here (e.g. `ALGORITHM_REINFORCE`) so writer
and reader import the same string and typos surface at import time.

On disk the payload is a plain `dict` (via `model_dump()`) of allowlisted
types (Tensor, str, int, dict), so `torch.load(..., weights_only=True)`
loads it safely. The Pydantic class lives only in-process.
"""

import torch
from pydantic import BaseModel, ConfigDict

ALGORITHM_REINFORCE: str = "reinforce"


class Checkpoint(BaseModel):
    """Self-describing checkpoint for a discrete-action policy network.

    Attributes:
        state_dict: PolicyNetwork weights, keyed by layer parameter name.
        algorithm: name of the training algorithm that produced these
            weights. Each algorithm's loader verifies this matches the
            algorithm it knows how to consume; a mismatch raises rather
            than silently loading incompatible weights into the wrong
            network shape.
        env_id: gym env id the model was trained on; load refuses on mismatch.
        obs_size: input dimension — the loader needs this to construct the
            network with matching input layer shape before applying weights.
        action_size: output dimension (number of discrete actions).
        hidden_size: width of the two hidden layers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_dict: dict[str, torch.Tensor]
    algorithm: str
    env_id: str
    obs_size: int
    action_size: int
    hidden_size: int
