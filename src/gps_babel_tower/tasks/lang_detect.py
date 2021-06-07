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

import fasttext
import langid
import cld3
import os
import urllib.request
import logging


class LangDetect:
  FAST_TEXT_MODEL_URL = 'https://storage.googleapis.com/babel-tower-exp/model/fasttext/langdetect/lid.176.ftz'
  
  def __init__(self, model_path = '/tmp/lid.176.ftz'):
    if not os.path.exists(model_path):
      print(f'downloading model from {self.FAST_TEXT_MODEL_URL} => {model_path}')
      with urllib.request.urlopen(self.FAST_TEXT_MODEL_URL) as f:
        with open(model_path, 'wb') as output:
          output.write(f.read())      
    self.ft_model = fasttext.load_model(model_path)
  
  def fasttext_get_language(self, text):
    labels, scores = self.ft_model.predict(text, k=5)
    return [(label.replace('__label__', ''), score) for (label, score) in zip(labels, scores)]

  def langid_get_language(self, text):
    lang, score = langid.classify(text)
    return [(lang, 1.0)]

  def cld3_get_language(self, text):
    result = cld3.get_language(text)
    if result:
      return [(result.language, result.probability)]
    else:
      logging.warn('cld3 failed to get predictions for text: %s', text)
      return []

  def get_language(self, text, return_details=False):
    if '\n' in text:
      text = text.split('\n')[0]
    
    detail_results = [
      self.langid_get_language(text),
      self.fasttext_get_language(text),
      self.cld3_get_language(text)
    ]
    scores = {}
    for result in detail_results:
      for lang, score in result:
        scores[lang] = scores.get(lang, 0) + score 

    result = None
    max_score = 0
    for lang, score in scores.items():
      if score > max_score:
        result = lang
        max_score = score
    
    if return_details:
      return result, detail_results
    else:
      return result