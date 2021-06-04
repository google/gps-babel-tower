from gps_babel_tower.models.text_embedding import TextEncoder


encoder = TextEncoder()

query = ['iPhone 12', 'iPhone 8']

candidates = [
  'iPhone 12 Pro Max',
  'iPhone 10',
  'iPhone 6s',
  'Xbox One X',
  'Xbox One S',
  'Xbox Series X',
  'Sony Playstation 4',
  'Sony PS4',
  'Nintendo Switch',
  'Nintendo Switch Lite',
]

result = encoder.batch_similarity(query, candidates)

for idx_q, q in enumerate(query):
  for idx_c, c in enumerate(candidates):
    print(f'similarity {q} to {c}:', result[idx_q][idx_c])
  
query = '小米手机8'

candidates = [
  '小米手机2s',
  '小米手机3',
  '小米手机mix2',
  '小米电视',
  '小米空气净化器',
  '小米盒子',
  '索尼手机',
  '索尼电视',
]

result = encoder.batch_similarity(query, candidates)

print(f'similarity to {query}:')
for (cand, score) in sorted(zip(candidates, result), key=lambda x: x[1], reverse=True):
  print(cand, score)