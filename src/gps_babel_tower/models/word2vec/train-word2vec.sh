#!/bin/bash

# get data:
# curl -O https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

echo "started:" $(date)
gcloud ai-platform local train --package-path word2vec --module-name word2vec.train --job-dir tmp/word2vec -- \
  --input-path ~/shakespeare.txt \
  --min-count 2 \
  --epochs 1 \
  --log-per-steps 10
echo "ended:" $(date)
