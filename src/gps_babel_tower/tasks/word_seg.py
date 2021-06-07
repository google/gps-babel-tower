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

import jieba
import nagisa

class WordSegment:
  def tokenize(self, text, lang, join_by=None):
    if lang == 'zh':
      words = jieba.cut(text)
    elif lang == 'ja':
      words = nagisa.tagging(text)
      words = words.words
    else:
      raise Exception(f'Unknown language {lang}')
    
    if join_by:
      return join_by.join(words)
    