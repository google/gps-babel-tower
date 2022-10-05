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

from typing import List, Mapping
import collections
from gps_babel_tower.models.text_embedding_pt import MT5TextEncoder
from sklearn.cluster import KMeans


class KMeansCluster:
  def __init__(self, **kwargs):
    self.text_encoder = MT5TextEncoder()
    self.kmeans = KMeans(**kwargs)
    
  def cluster(self, text_list: List[str]) -> Mapping[int, List[str]]:
    X = self.text_encoder.encode(text_list)
    classes = self.kmeans.fit_predict(X)
    
    result = collections.defaultdict(lambda: [])
    for text, text_class in zip (text_list, classes):
      result[text_class].append(text)
    
    return result