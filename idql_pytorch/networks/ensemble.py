from typing import Type

import torch.nn as nn
import jax
import numpy as jnp


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)

# TODO: find equivalent of jax.tree_util.tree_map
def subsample_ensemble(params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jnp.random.choice(a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params

# potential solution:
# from typing import Callable, Any

# def is_leaf_default(node):
#     return not isinstance(node, (list, tuple, dict))

# def tree_map(f: Callable[..., Any], tree: Any, *rest: Any, is_leaf: Callable[[Any], bool] = None) -> Any:
#     """Maps a function over elements of nested structures (lists, tuples, dicts)"""
#     is_leaf = is_leaf if is_leaf else is_leaf_default

#     if is_leaf(tree):
#         return f(tree, *(r[0] if isinstance(r, (list, tuple)) else r for r in rest))

#     if isinstance(tree, (list, tuple)):
#         return type(tree)(tree_map(f, t, *(r[i] if isinstance(r, (list, tuple)) else r for r in rest), is_leaf=is_leaf)
#                          for i, t in enumerate(tree))
#     elif isinstance(tree, dict):
#         return {k: tree_map(f, v, *(r[k] if isinstance(r, dict) else r for r in rest), is_leaf=is_leaf)
#                 for k, v in tree.items()}
#     else:
#         raise ValueError("Input must be a nested structure of lists, tuples, or dicts")
