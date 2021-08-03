# coding=utf-8
# Copyright 2021 Google LLC..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gps_babel_tower.tasks.nearest_neighbor import NearestNeighbor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()])

# Models: https://huggingface.co/models?search=dpr
nn = NearestNeighbor(
    model_id='facebook/dpr-ctx_encoder-single-nq-base',
    data_file_path='nn_sample.txt',
    index_file_path='/tmp/nn_index.faiss')

print(nn.get_nearest_neighbors(query='iPhone 12', top_k=3))
print(nn.get_nearest_neighbors(query='xbox one', top_k=3))
