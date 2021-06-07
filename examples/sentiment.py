# Copyright 2021 Google LLC
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

from gps_babel_tower.tasks.sentiment import SentimentClient


examples = [
  'I hate you',
  "It's ok",
  'I like this app but it has minor issues'
]

# Explore models here:
# https://huggingface.co/models?search=sentiment
models = [
  'distilbert-base-uncased-finetuned-sst-2-english',
  'nlptown/bert-base-multilingual-uncased-sentiment',
  'cardiffnlp/twitter-roberta-base-sentiment',
  'm3hrdadfi/albert-fa-base-v2-sentiment-multi'
]

# Using HuggingFace models (fast, cheap)
s = SentimentClient(model='nlptown/bert-base-multilingual-uncased-sentiment',use_fast=True)
print('local model')
for ex in examples:
  print(ex, s.score(ex))

# Using GCP API (more accurate)
s = SentimentClient(engine='gcp')
print('gcp')
for ex in examples:
  print(ex, s.score(ex, score_range=(0,100)))
