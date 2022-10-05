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

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MT5EncoderModel, T5Tokenizer


class MT5TextEncoder:
  def __init__(self, model_path='google/mt5-base', device='cuda'):
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = MT5EncoderModel.from_pretrained(model_path).to(device)
    self.device = device

  @torch.no_grad()
  def encode(self, texts, batch_size=256, return_type='numpy', max_length=None):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
      text_batch = texts[i:i+batch_size]
      emb_batch = self._encode_batch(text_batch, max_length=max_length)
      embs.append(emb_batch)
    
    result = torch.cat(embs)
    
    if return_type == 'numpy':
      return result.detach().cpu().numpy()
    return result
  
  @torch.no_grad()
  def _encode_batch(self, text_batch, max_length=None):
    tokens = self.tokenizer(
      text_batch,
      padding='longest',
      truncation=True if max_length else False,
      max_length=max_length,
      return_tensors='pt').to(self.device)

    outputs = self.model(**tokens)
    
    embs = outputs.last_hidden_state.mean(axis=1)

    return embs