from gps_babel_tower.tasks.language_detection import LanguageDetection

examples = [
'Hi, this is Jim.',
'你好呀，我是巴巴',
'おはよう、私わ田中です。',
]

l = LanguageDetection()

for e in examples:
  print(e, l.detect(e))