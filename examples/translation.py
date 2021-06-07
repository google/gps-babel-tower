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

from gps_babel_tower.tasks.translation import TranslationClient

t = TranslationClient()
# Translate from Chinese to English
print('English:', t.translate('你好', src_lang='zh', target_lang='en'))

# Source language can also be auto detected
print('English:', t.translate('你好', target_lang='en'))
print('French:', t.translate('你好', target_lang='fr'))

# When there is not a model from zh -> jp, will try zh -> en -> jp
print('Japanese:', t.translate('我的名字叫巴巴', src_lang='zh', target_lang='jp'))

t = TranslationClient(engine='gcp')
# Translate from Chinese to English
print('English:', t.translate('你好', target_lang='jp'))

# Source language can also be auto detected
print('English:', t.translate('你好', target_lang='en'))
print('French:', t.translate('你好', target_lang='fr'))

print('Japanese:', t.translate('我的名字叫巴巴',  target_lang='ja'))

