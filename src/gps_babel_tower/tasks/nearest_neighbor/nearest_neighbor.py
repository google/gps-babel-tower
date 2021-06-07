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

import torch
import os
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, DPRContextEncoder


_EMB_COL = 'embedding'
_TEXT_COL = 'text'


class NearestNeighbor:
  def __init__(self,
               model_id,
               data_file_path,
               force_rebuild_index=False,
               index_file_path=None,
              ):
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model = DPRContextEncoder.from_pretrained(model_id)
    
    ds = load_dataset('text', data_files={'train': data_file_path})['train']
    if (not force_rebuild_index) and index_file_path and os.path.exists(index_file_path):
      logging.info('Load index file from %s', index_file_path)
      ds.load_faiss_index('embedding', index_file_path)
    else:
      logging.info('Building index')
      ds = ds.map(lambda example: {_EMB_COL: self.encode(example[_TEXT_COL])})
      ds.add_faiss_index(column=_EMB_COL)
      if index_file_path:
        logging.info('Save index to %s', index_file_path)
        ds.save_faiss_index(_EMB_COL, index_file_path)
        
    self.dataset = ds

  def encode(self, text):
    tokens = self.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
      result = self.model(**tokens).pooler_output.numpy()[0]
    return result
  
  def get_nearest_neighbors(self, query, top_k=10):
    q_emb = self.encode(query)
    scores, retrieved_examples = self.dataset.get_nearest_examples(_EMB_COL, q_emb, k=top_k)
    return retrieved_examples[_TEXT_COL]
