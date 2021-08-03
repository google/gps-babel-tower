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

from gps_babel_tower.models.text_embedding import TextEncoder

encoder = TextEncoder()

query = ['iPhone 12', 'iPhone 8']

candidates = [
    'iPhone 12 Pro Max',
    'iPhone 10',
    'iPhone 6s',
    'Xbox One X',
    'Xbox One S',
    'Xbox Series X',
    'Sony Playstation 4',
    'Sony PS4',
    'Nintendo Switch',
    'Nintendo Switch Lite',
]

result = encoder.batch_similarity(query, candidates)

for idx_q, q in enumerate(query):
  for idx_c, c in enumerate(candidates):
    print(f'similarity {q} to {c}:', result[idx_q][idx_c])

query = '小米手机8'

candidates = [
    '小米手机2s',
    '小米手机3',
    '小米手机mix2',
    '小米电视',
    '小米空气净化器',
    '小米盒子',
    '索尼手机',
    '索尼电视',
]

result = encoder.batch_similarity(query, candidates)

print(f'similarity to {query}:')
for (cand, score) in sorted(
    zip(candidates, result), key=lambda x: x[1], reverse=True):
  print(cand, score)
