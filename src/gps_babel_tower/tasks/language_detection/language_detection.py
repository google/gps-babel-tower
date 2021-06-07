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

from google.cloud import translate_v2 as translate


class LanguageDetection():
  def __init__(self, engine='gcp'):
    self.engine = engine
    if self.engine == 'gcp':
      self.translate_client = translate.Client()
    else:
      raise RuntimeError(f'engine {engine} not supported yet')
    return

  def detect(self, text):
    language_code = None
    if self.engine == 'gcp':
      result = self.translate_client.detect_language(text)
      # print("Confidence: {}".format(result["confidence"]))
      language_code = result["language"]
    else:
      raise RuntimeError('Unsupported')
    return language_code

