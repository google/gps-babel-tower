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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langdetect import detect

import six
from google.cloud import translate_v2 as translate


# TODO: support cloud translation API


# From langdetect language to huggingface model language
# ref: https://pypi.org/project/langdetect/
# ref: https://huggingface.co/models?filter=translation
LANG_MAPPING = {
  'zh-cn': 'zh',
  'zh-tw': 'zh',
  'jp': 'jap',
}


class TranslationClient:
  def __init__(self, engine='hf'):
    """Initializes translation client.
    
    Args: 
      engine: translation engine. valid options are:
        'hf' - using hugging face pipeline
        'gcp' - using Google Cloud translation API
    """
    self.engine = engine
    self.model_cache = {}
    
  def _get_hf_model(self, src_lang, target_lang):
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
    if model_id not in self.model_cache:
      print('loading model:', model_id)
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
      self.model_cache[model_id] = (tokenizer, model)
    
    return self.model_cache[model_id]
    
  def _translate_hf(self, sentence, src_lang, target_lang):
    tokenizer, model = self._get_hf_model(src_lang, target_lang)
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0]).replace('<pad>', '')

  def translate(self, sentence, target_lang, src_lang=None):
    """Translates a sentence.
    
    Args:
      sentence: The sentence to translate.
      target_lang: translate to this language.
      src_lang: translate from this language. If none, automatically determine source language.
    """
    if self.engine == 'hf':
      if src_lang is None:
        src_lang = detect(sentence)
      
      src_lang = LANG_MAPPING.get(src_lang, src_lang)
      target_lang = LANG_MAPPING.get(target_lang, target_lang)
      
      if src_lang == target_lang:
        # skip translation if src and target are the same
        return sentence
      
      try:
        result = self._translate_hf(sentence, src_lang, target_lang)
      except:
        en_sentence = self._translate_hf(sentence, src_lang, 'en')
        print(src_lang, '->', 'en:', sentence, en_sentence)
        result = self._translate_hf(en_sentence, 'en', target_lang)
        print('en', '->', target_lang, en_sentence, result)
      return result
    elif self.engine == 'gcp':
      translate_client = translate.Client()
      if isinstance(sentence, six.binary_type):
        sentence = sentence.decode("utf-8")
      # Text can also be a sequence of strings, in which case this method
      # will return a sequence of results for each text.
      try:
        result = translate_client.translate(sentence, target_language=target_lang)
        return result["translatedText"]
      except:
        print('Unsupported language', target_lang)
        return
    else:
      raise RuntimeError('Unsupported')