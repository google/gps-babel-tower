from gps_babel_tower.tasks.sentiment import SentimentClient


examples = [
  'I hate you',
  "It's ok",
  'I like this app but it has minor issues'
]

# Explore models here:
# https://huggingface.co/models?search=sentiment
models = [
  'distilbert-base-uncased-finetuned-sst-2-english',
  'nlptown/bert-base-multilingual-uncased-sentiment',
  'cardiffnlp/twitter-roberta-base-sentiment',
  'm3hrdadfi/albert-fa-base-v2-sentiment-multi'
]

# Using HuggingFace models (fast, cheap)
s = SentimentClient(model='nlptown/bert-base-multilingual-uncased-sentiment',use_fast=True)
print('local model')
for ex in examples:
  print(ex, s.score(ex))

# Using GCP API (more accurate)
s = SentimentClient(engine='gcp')
print('gcp')
for ex in examples:
  print(ex, s.score(ex, score_range=(0,100)))
