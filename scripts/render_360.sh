#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --tmp=8000
#SBATCH --job-name=render360
#SBATCH --output=output/render360.out
#SBATCH --error=output/render360.err
#SBATCH --gpus=rtx_2080_ti:4

export CUDA_VISIBLE_DEVICES=0

SCENE=living_room_reduced
EXPERIMENT=360
DATA_DIR=/cluster/work/riner/users/PLR-2023/jgaber/PLR-multinerf/data/"$SCENE"
CHECKPOINT_DIR=/cluster/work/riner/users/PLR-2023/jgaber/PLR-multinerf/temp/nerf_results/"$EXPERIMENT"/"$SCENE"

python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 300" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 30" \
  --logtostderr
