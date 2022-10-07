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

#!/bin/bash
# Usage
# PROJECT_ID=your_gcp_project_id ./build_docker.sh

set -euxo pipefail

IMAGE_URI="gcr.io/$PROJECT_ID/babel_example:v1"
docker build -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI