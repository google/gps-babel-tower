# coding=utf-8
# Copyright 2020 Google LLC..
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

# Lint as: python3
"""Keywords extraction component."""
import re
import logging
import string

import nltk
from nltk.corpus import stopwords
import spacy
import spacy.cli.download


class KeywordExtraction:

  def __init__(self):
    try:
      self.nlp = spacy.load('en_core_web_sm')
    except:
      logging.exception('load model failed, download automatically') 
      spacy.cli.download('en_core_web_sm')
      self.nlp = spacy.load('en_core_web_sm')
    
    nltk.download('stopwords')
    self.stopwords_en = set(stopwords.words('english'))

  def tokenize(self, text, stop_words=None, word_normalize_map=None, mode=None):
    sentence_token_limit = 1
    lda_tokens = []
    tokens = self.nlp(self.format_sentence(text))

    if stop_words:
      self.stopwords_en.update(stop_words)

    if len(tokens) >= sentence_token_limit:
      root_word = ''
      for token in tokens:
        if token.orth_.isspace():
          continue
        if token.lower_ in self.stopwords_en:
          continue
        words = None
        if mode == 'entity_vector':
          if token.dep_ in ('nsubj', 'nsubjpass'):
            words = token.vector
        elif mode == 'entity_name':
          if token.dep_ in ('nsubj', 'nsubjpass'):
            words = token.lower_
        else:
          if token.dep_ in ('compound', 'dobj', 'pobj', 'xcomp', 'nsubj',
                            'nsubjpass', 'advcl', 'amod', 'acomp'):
            word = self.normalize_word(word_normalize_map, token)
            if token.dep_ in ('dobj', 'pobj', 'xcomp'):
              if token.dep_ in ('pobj',):
                words = self.normalize_word(word_normalize_map,
                                            token.head.head) + ' ' + word
              else:
                words = self.normalize_word(word_normalize_map,
                                            token.head) + ' ' + word
            elif token.dep_ in ('compound', 'advcl', 'nsubj', 'nsubjpass',
                                'amod', 'acomp'):
              if token.head.lemma_ == 'be':
                if token.dep_ in ('nsubj', 'nsubjpass'):
                  root_word = word
                  root_word_head_idx = token.head.idx
                elif (root_word and
                      root_word_head_idx == token.head.idx and
                      token.dep_ == 'acomp'):
                  words = word + ' ' + root_word
              elif token.dep_ in ('compound', 'advcl', 'nsubj', 'nsubjpass',
                                  'amod'):
                words = word + ' ' + self.normalize_word(
                    word_normalize_map,
                    token.head)
        if words is not None:
          lda_tokens.append(words)
    return lda_tokens

  def format_sentence(self, text, stop_words=None, strip_all=False):
    if isinstance(text, str):
      emoji_pattern = re.compile(
          '['
          u'\U0001F600-\U0001F64F'  # emoticons
          u'\U0001F300-\U0001F5FF'  # symbols & pictographs
          u'\U0001F680-\U0001F6FF'  # transport & map symbols
          u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
          ']+',
          flags=re.UNICODE)
      text = emoji_pattern.sub(r'', text)
      if stop_words:
        text = ' '.join([
            word for word in nltk.word_tokenize(text)
            if word.lower() not in stop_words
        ])
      if strip_all:
        del_str = string.punctuation + string.digits  # ASCII 标点符号
      else:
        del_str = string.digits
      identify = str.maketrans('', '', del_str)
      clean_line = text.translate(identify)
      return clean_line
    else:
      print('text format error:', text)
      return ''

  def normalize_word(self, words_map, token):
    if words_map is not None and token.text in words_map:
      normalized_word = words_map[token.text].lower()
    else:
      normalized_word = token.lemma_
    return normalized_word

