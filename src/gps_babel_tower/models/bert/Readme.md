# Installing depedencies
```bash
pip install tensorflow tensorboard tf-models-official
```

# Getting official bert models
Get official bert model checkpoint from here:
https://github.com/tensorflow/models/tree/master/official/nlp/bert#access-to-pretrained-checkpoints

By default there are a few models already stored in babel's GCP bucket `gs://babel-tower-exp/model`


# Pretraining(on domain text)
## Create pretraining data
To prepare data for pretraining bert(with domain-specific text), use the following command:
```bash
python -m official.nlp.data.create_pretraining_data \
  --input_file=gs://babel-tower-exp/data/shakespeare.txt \
  --output_file=gs://babel-tower-exp/data/shakespeare.tfrecord \
  --vocab_file=gs://babel-tower-exp/model/bert/uncased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1
```
  
## Pretraining
```bash
python -m official.nlp.bert.run_pretraining \
  --input_files=gs://babel-tower-exp/data/shakespeare.tfrecord \
  --model_dir=gs://babel-tower-exp/model/bert_pretrain_shakespeare \
  --bert_config_file=gs://babel-tower-exp/model/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=gs://babel-tower-exp/model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_epochs=2 \
  --num_steps_per_epoch=30 \
  --warmup_steps=10 \
  --steps_per_loop=1 \
  --learning_rate=2e-5
```

## Pretraining on Cloud TPU
Run this on Cloud Shell to set up a Cloud TPU:
```bash
ctpu up --project $PROJECT --zone asia-east1-c
```

Run this on cloud TPU VM to pretrain (When in Cloud TPU VM, TPU_NAME is automatically set up):
```bash
PYTHONPATH=/usr/share/models python3 -m official.nlp.bert.run_pretraining \
  --input_files=gs://babel-tower-exp/data/shakespeare.tfrecord \
  --model_dir=gs://babel-tower-exp/model/bert_pretrain_shakespeare \
  --bert_config_file=gs://babel-tower-exp/model/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=gs://babel-tower-exp/model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=256 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_epochs=5 \
  --num_steps_per_epoch=10000 \
  --warmup_steps=1000 \
  --steps_per_loop=500 \
  --tpu=$TPU_NAME \
  --distribution_strategy=tpu \
  --learning_rate=2e-5
```