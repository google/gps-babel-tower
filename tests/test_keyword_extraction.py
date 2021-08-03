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

import unittest

from gps_babel_tower.tasks.keyword_extraction import KeywordExtraction


class TestKeywordExtraction(unittest.TestCase):

  def test_rake(self):
    ke = KeywordExtraction('rake')
    text = 'Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, file storage, and YouTube.'
    tokens = ke.extract_keywords(text, max_results=5)
    self.assertEqual(len(tokens), 5)
      

if __name__ == '__main__':
  unittest.main()