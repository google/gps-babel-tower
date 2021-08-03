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


from gps_babel_tower.utils import data_util
import json
import os
import tensorflow as tf


gcs_bucket = os.environ['GCS_BUCKET']
project = os.environ['GCP_PROJECT']

gcs_path = f'gs://{gcs_bucket}/tmp/test_data.json'
bq_table = f'{project}.test_dataset.test_table'

test_data = [
  {'item': idx, 'content': f'content{idx}'} for idx in range(100)
]


with tf.io.gfile.GFile(gcs_path, 'w') as file:
  file.write('\n'.join(json.dumps(item) for item in test_data))

data_util.load_data_gcs_to_bq(gcs_path, bq_table)
print(f'loaded data from {gcs_path} to {bq_table}')

print(f'Iterating from {gcs_path}')
for batch in data_util.iterate_data(gcs_path=gcs_path, batch_size=7, limit=50):
  print('batch: size:', len(batch), batch)

gcs_glob_path = 'gs://{gcs_bucket}/tmp/*.json'
print(f'Iterating from {gcs_glob_path}')
for batch in data_util.iterate_data(gcs_path=gcs_glob_path, batch_size=7, limit=50):
  print('batch: size:', len(batch), batch)
  
print(f'Iterating from {bq_table}')
for batch in data_util.iterate_data(bq_table=bq_table, batch_size=6, limit=50):
  print('batch: size:', len(batch), batch)