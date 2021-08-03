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

"""1.enable sheets and drive api in cloud console 2.create service account and mv service_account.json to ~/.config/gspread 3.share the sheets edit permission to service account email address
"""
import gspread
from gspread_dataframe import get_as_dataframe


class SheetUtil():

  def __init__(self, service_account_filename=None):
    """Google Sheet Util for read and write
    """
    # Todo
    # the spreadsheets credentials is different from cloud bigquery credentials
    if not service_account_filename:
      self.gc = gspread.service_account()
    else:
      self.gc = gspread.service_account(filename=service_account_filename)

    return

  def read(self, sheet_id_or_url, worksheet=None):
    if 'http' in sheet_id_or_url:
      sh = self.gc.open_by_url(sheet_id_or_url)
    else:
      sh = self.gc.open_by_key(sheet_id_or_url)

    if not worksheet:
      worksheet = sh.get_worksheet(0)
    else:
      worksheet = sh.worksheet(worksheet)
    # worksheet.get_all_records()
    # return dict add .to_dict(orient='records')
    res = get_as_dataframe(worksheet, parse_dates=True, evaluate_formulas=True)
    # delete all NaN columns and rows
    res = res.dropna(axis=0, how='all').dropna(axis=1, how='all')
    return res


if __name__ == '__main__':
  su = SheetUtil('default.json')
  df = su.read('1ck8Aeij33kQ4lDR36KGPMsRqcorPzt4iP9v0fGJOyq0', 'Wildcat Output')
  print(df)
