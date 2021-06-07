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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from google.cloud import language_v1 as language
import logging

# Explore models here:
# https://huggingface.co/models?search=sentiment


class SentimentClient:
  def __init__(self, engine='hf', use_gpu=False, model='distilbert-base-uncased-finetuned-sst-2-english', use_fast=True):
    self.engine = engine
    if engine == 'hf':
      self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast)
      self.model = AutoModelForSequenceClassification.from_pretrained(model)
      if use_gpu:
        self.model.cuda()
      self.use_gpu = use_gpu
    elif engine == 'gcp':
      self.client = language.LanguageServiceClient()
      
    else:
      raise RuntimeError(f'engine {engine} not supported yet')

  def score(self, sentence, score_range=(-1,1)):
    if self.engine == 'hf':
      tokens = self.tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors="pt")
      if self.use_gpu:
        tokens = {k: v.cuda() for k,v in tokens.items()}
      with torch.no_grad():
        logits = self.model(**tokens).logits
      probs = torch.softmax(logits, dim=-1)
      max_score = probs.shape[1]-1
      scores_arange = torch.arange(max_score+1)
      if self.use_gpu:
        scores_arange = scores_arange.cuda()
      score = torch.sum(scores_arange * probs, axis=-1) # (0, max_score)
      score = score_range[0] + (score_range[1]-score_range[0]) * score / max_score
      if self.use_gpu:
        score = score.cpu()
      score = score.numpy()
      if score.size == 1:
        score = score.item()
      return score
    elif self.engine == 'gcp':
      document = {
        'content':sentence,
        "type_":language.Document.Type.PLAIN_TEXT
        }
      encoding_type = language.EncodingType.UTF8
      
      request = {'document': document, 'encoding_type': encoding_type}
      try:
        # Detects the sentiment of the text
        response = self.client.analyze_sentiment(request = request)
      except:
        logging.exception('unable to get score from gcp')
        return 0
      score = response.document_sentiment.score
      # score origin range(-1,1)
      score = score_range[0] + (score_range[1]-score_range[0]) * ((score + 1) / 2)
      return score
    else:
      raise RuntimeError(f'Unsupported engine {self.engine}')