#!/bin/bash

set -euxo pipefail
IMAGE_URI=codescv/gps-babel-tower-gpu:v11

docker build -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI
