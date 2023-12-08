from functools import partial
import numpy as np
from flax.training.train_state import TrainState

def _sample_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn(params, observations)
    return dist.sample()

def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn(params, observations)
    return dist.mode()


class TrainState:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def apply_gradients(self, grads):
        for param, grad in zip(self.model.parameters(), grads):
            param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()

    # You can add more methods as needed, for example, to save and load state, etc.

class Agent():
    def __init__(self, actor_model):
        self.actor = TrainState

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=self.rng)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions)