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
# Usage:
# PROJECT_ID=your_project_id ./run_on_ai_platform.sh --args="--output_path,gs://your_bucket/output/path"
set -euo pipefail

REGION=us-west1
JOB_ID=babel_example_$(date +"%Y%m%d_%H%M%S")
DOCKER_IMAGE="gcr.io/$PROJECT_ID/babel_example:v1"

gcloud ai custom-jobs create \
  --region=$REGION \
  --project=$PROJECT_ID \
  --worker-pool-spec=replica-count=1,machine-type='n1-standard-8',container-image-uri="$DOCKER_IMAGE",accelerator-type="NVIDIA_TESLA_P100",accelerator-count=1 \
  --display-name=$JOB_ID \
  "$@"
  
#echo "Go to https://pantheon.corp.google.com/ai-platform/jobs?project=$PROJECT_ID to see your job status."
echo "Go to https://pantheon.corp.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID to see your job status."