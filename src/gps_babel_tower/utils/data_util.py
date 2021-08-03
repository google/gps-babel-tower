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

from google.cloud import bigquery
import google.api_core.exceptions
import tensorflow as tf
import csv
import json
import logging


def read_data(gcs_path=None, bq_table=None, bq_sql=None, limit=None):
  """Read data from BigQuery/GCS.

  If GCS path is in json or csv extensions, parse them to python objects.
  """
  if bq_table or bq_sql:
    if bq_table:
      bq_sql = f"""
        SELECT *
        FROM `{bq_table}`
        LIMIT {limit}
      """
    client = bigquery.Client()
    for idx, r in enumerate(client.query(bq_sql)):
      if limit is not None and idx <= limit:
        yield dict(r)
      else:
        break
  elif gcs_path:
    for path in tf.io.gfile.glob(gcs_path):
      with tf.io.gfile.GFile(path) as file:
        if gcs_path.endswith('.csv'):
          csv_reader = csv.DictReader(file)
          for idx, r in enumerate(csv_reader):
            if limit is not None and idx <= limit:
              yield r
            else:
              break
        elif gcs_path.endswith('.json'):
          for idx, line in enumerate(file):
            if limit is not None and idx <= limit:
              yield json.loads(line.strip())
            else:
              break


def iterate_data(batch_size, **kwargs):
  """Iterate data in batches from BigQuery/GCS."""
  batch = []
  for item in read_data(**kwargs):
    batch.append(item)
    if len(batch) == batch_size:
      yield batch
      batch = []

  if batch:
    yield batch


def extract_bq_to_gcs(sql, tmp_table_id, gcs_path):
  client = bigquery.Client()

  query_job = client.query(
      sql,
      job_config=bigquery.QueryJobConfig(
          destination=tmp_table_id,
          write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE))
  query_job.result()
  logging.info(f'extract data using sql: {sql}')

  extract_job = client.extract_table(
      bigquery.TableReference.from_string(tmp_table_id),
      gcs_path,
      job_config=bigquery.ExtractJobConfig(
          destination_format=bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON))
  extract_job.result()  # Waits for job to complete.

  client.delete_table(tmp_table_id, not_found_ok=True)
  logging.info(f'Loaded data to {gcs_path}.')


def load_data_gcs_to_bq(gcs_path, bq_table,
                        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE):
  client = bigquery.Client()

  # bq_table is `project.dataset.table`, remove the table part to get dataset id.
  dataset_id = '.'.join(bq_table.split('.')[:-1])
  dataset = bigquery.Dataset(dataset_id)
  try:
    dataset = client.create_dataset(dataset)  # Make an API request.
    logging.info('Created dataset %s', dataset_id)
  except google.api_core.exceptions.Conflict:
    # ignore if already exists
    logging.info('dataset already exists %s', dataset_id)

  job_config = bigquery.LoadJobConfig(
    autodetect=True,
    source_format=source_format,
    write_disposition=write_disposition,
  )
  job = client.load_table_from_uri(
      gcs_path, bq_table, job_config=job_config
  )
  logging.info('Loading result from {gcs_path} to {bq_table}')
  job.result()  # Waits for the job to complete.