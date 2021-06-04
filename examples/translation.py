from gps_babel_tower.tasks.translation import TranslationClient

t = TranslationClient()
# Translate from Chinese to English
print('English:', t.translate('你好', src_lang='zh', target_lang='en'))

# Source language can also be auto detected
print('English:', t.translate('你好', target_lang='en'))
print('French:', t.translate('你好', target_lang='fr'))

# When there is not a model from zh -> jp, will try zh -> en -> jp
print('Japanese:', t.translate('我的名字叫巴巴', src_lang='zh', target_lang='jp'))

t = TranslationClient(engine='gcp')
# Translate from Chinese to English
print('English:', t.translate('你好', target_lang='jp'))

# Source language can also be auto detected
print('English:', t.translate('你好', target_lang='en'))
print('French:', t.translate('你好', target_lang='fr'))

print('Japanese:', t.translate('我的名字叫巴巴',  target_lang='ja'))

