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

import pandas as pd
# also try: dpr-question_encoder-multiset-base
from transformers import AutoTokenizer, DPRQuestionEncoder, TFDPRQuestionEncoder
from transformers import XLMTokenizer, TFXLMModel
import tensorflow as tf
from typing import Sequence, Union
import enum


class EncoderType(enum.Enum):
  DPR = enum.auto()
  XLM = enum.auto()


class TextEncoder:
  # Ref: https://huggingface.co/transformers/multilingual.html
  def __init__(self, model=EncoderType.XLM, max_length=32):
    if model == EncoderType.DPR:
      model_key = 'facebook/dpr-question_encoder-single-nq-base'
      self.tokenizer = AutoTokenizer.from_pretrained(model_key)
      self.model = TFDPRQuestionEncoder.from_pretrained(model_key)
    elif model == EncoderType.XLM:
      model_key = 'xlm-mlm-100-1280'
      self.tokenizer = XLMTokenizer.from_pretrained(model_key)
      self.model = TFXLMModel.from_pretrained(model_key)
    print('using model', model_key)
    self.max_length = max_length
    self.emb_cache = {}

  def encode(self, words, cache_key=None, max_length=None):
    if cache_key and cache_key in self.emb_cache:
      return self.emb_cache[cache_key]
    
    max_length = max_length or self.max_length
    tokens = self.tokenizer(words, padding='max_length', max_length=max_length, truncation=True, return_tensors='tf')
    outputs = self.model(**tokens)
    if hasattr(outputs, 'pooler_output'):
      result = outputs.pooler_output
    else:
      # use first token output as sentence encoding
      # TODO: try average pooling
      result = outputs.last_hidden_state[:,0,:]
      
    if cache_key:
      self.emb_cache[cache_key] = result
    
    return result
  
  def batch_similarity(self,
                       query: Union[str, Sequence[str]],
                       candidates: Sequence[str],
                       cache_key1=None,
                       cache_key2=None,
                       similarity='cosine',
                       return_type='list'):
    """Compute 1 vs n similarity from query to a list of candidates."""
    return_single = False
    if isinstance(query, str):
      query = [query]
      return_single = True
    
    # (query_count, dim)
    query_embedding = self.encode(query, cache_key=cache_key1)
    # (candidates_count, dim)
    candidates_embedding = self.encode(candidates, cache_key=cache_key2)
    
    if similarity == 'cosine':
      sim_func = cosine_similarity
    elif similarity == 'l2':
      sim_func = l2_distance
    else:
      raise Exception(f'unknown similarity function {similarity}')
      
    # (candidates_count,)
    result = sim_func(query_embedding, candidates_embedding)
    
    if return_type == 'raw':
      pass
    elif return_type == 'list':
      result = list(result.numpy())
    elif return_type == 'numpy':
      result = result.numpy()
    else:
      raise Exception(f'unknown return type {return_type}')
      
    if return_single:
      result = result[0]
      
    return result

  
def cosine_similarity(v1, v2, eps=1e-8):
  """v1:  (n1, k)  vector
     v2:  (n2, k)  vector
     Returns:  (n1, n2)  cosine similarities
  """
  p = tf.matmul(v1, v2, transpose_b=True)
  n = tf.math.maximum(tf.expand_dims(tf.norm(v1, axis=-1), axis=-1) * tf.expand_dims(tf.norm(v2, axis=-1), axis=0), eps)
  scores = p / n
  return scores


def l2_distance(v1, v2):
  """v1:  (n1, k)  vector
     v2:  (n2, k)  vector
     Returns:  (n1, n2)  l2 distances
  """
  return tf.norm(tf.expand_dims(v1, 1) - tf.expand_dims(v2, 0), axis=-1)
  
