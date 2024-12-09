import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from typing import Union
import jax.numpy as jnp
from transformers import (
    FlaxDistilBertForTokenClassification,
    AutoConfig,
)
import transformers


class Embed(nn.Module):                    # create a Flax Module dataclass
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, ip):
        op = nn.Embed(num_embeddings=self.num_embeddings,
                      features=self.features)(ip)
        return op

def model_repr_dict(d: Union[dict, FrozenDict],):
    num_params = 0
    repr_dict = dict()

    for k, v in d.items():
        if (isinstance(v, dict)
            or isinstance(v, FrozenDict)):
            num_params_new, repr_new = model_repr_dict(v)
            num_params += num_params_new
            repr_dict[k] = repr_new
        else:
            num_params += v.size
            repr_dict[k] = v.shape

    return num_params, repr_dict

def model_repr(d: Union[dict, FrozenDict],
               indent: int=0,
               indent_step: int=2):
    num_params = 0
    repr_str = ''

    for k, v in d.items():
        if (isinstance(v, dict)
            or isinstance(v, FrozenDict)):
            repr_str += ' ' * indent + f'{k}:\n'
            num_params_new, repr_new = model_repr(v, indent=indent+indent_step)
            repr_str += repr_new
            num_params += num_params_new
        else:
            num_params += v.size
            repr_str += ' ' * indent + f'{k}: {v.shape}\n'

    return num_params, repr_str

def model_desc(d: Union[dict, FrozenDict],
               indent: int=0,
               indent_step: int=2):
    num_params, repr_str = model_repr(d, indent=indent, indent_step=indent_step)
    repr_str = 'Total number of parameters: {:,}\n'.format(num_params) + repr_str
    return repr_str

class Linear(nn.Module):
  out_dims: int

  @nn.compact
  def __call__(self, x):
    *batch_dims, n_ip_dims = x.shape
    x = x.reshape((-1, n_ip_dims))
    x = nn.Dense(self.out_dims)(x)
    return x.reshape((*batch_dims, self.out_dims))

class MLP(nn.Module):
  out_dims: int
  hidden_size: int = 128

  @nn.compact
  def __call__(self, x):
    *batch_dims, n_ip_dims = x.shape
    x = x.reshape((-1, n_ip_dims))
    x = nn.Dense(self.hidden_size)(x)
    x = nn.elu(x)
    x = nn.Dense(self.out_dims)(x)
    return x.reshape((*batch_dims, self.out_dims))
