import numpy as np
from dm_env.specs import DiscreteArray


def num_from_spec(spec: DiscreteArray) -> int:
    if spec.num_values:
        return spec.num_values
    else:
        np.prod(spec.shape)
