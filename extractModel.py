# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render script."""

import concurrent.futures
import functools
import glob
import os
import time

from absl import app
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import utils
import jax
from jax import random
from matplotlib import cm
import mediapy as media
import numpy as np
import jax.numpy as jnp                                   

configs.define_common_flags()
jax.config.parse_flags_with_absl()



def main(unused_argv):

  print("Extracting Model from NeRF started.")

  config = configs.load_config(save_config=False)

  dataset = datasets.load_dataset('test', config.data_dir, config)

  key = random.PRNGKey(20200823)
  model, state, _, _, _ = train_utils.setup_model(config, key)

  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  step = int(state.step)
  print(f'Extracting Model at checkpoint step {step}.')

  out_name = f'extractedModel_step_{step}'
  base_dir = config.extractModel_dir
  if base_dir is None:
    base_dir = os.path.join(config.checkpoint_dir, 'extractModel')
  out_dir = os.path.join(base_dir, out_name)
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)




  #model.setupMLPforModelextraction()

  mean = jnp.array([[[0,0,0],[1,1,1]]])
  cov = jnp.array([[[1,1,1],[1,1,1]]])

  gaussians = (mean, cov)
  
  print("blub")

  #results = model.apply(key, rays = None, train_frac = None, gaussians = gaussians)

  #print("results sind da!!!!!!!!!!")
  #print("density: ", results['density'])
  #print("rgb: ", results['rgb'])





  

  

















  
  # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
  x = jax.numpy.ones([jax.local_device_count()])
  x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
  print(x)


if __name__ == '__main__':
  with gin.config_scope('eval'):  # Use the same scope as eval.py
    app.run(main)
