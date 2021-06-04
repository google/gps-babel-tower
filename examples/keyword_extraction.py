from gps_babel_tower.tasks.keyword_extraction import KeywordExtraction

ke = KeywordExtraction('spacy_bigram')

tokens = ke.extract_keywords('Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, file storage, and YouTube.')
print('spacy_bigram:', tokens)


ke = KeywordExtraction('rake')
tokens = ke.extract_keywords('Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, file storage, and YouTube.')
print('rake:', tokens)


ke = KeywordExtraction('keybert')
tokens = ke.extract_keywords("""「人工知能」という名前は1956年にダートマス会議でジョン・マッカーシーにより命名された。
現在では、記号処理を用いた知能の記述を主体とする情報処理や研究でのアプローチという意味あいでも使われている。
日常語としての「人工知能」という呼び名は非常に曖昧なものになっており、多少気の利いた家庭用電気機械器具の制御システムやゲームソフトの思考ルーチンなどがこう呼ばれることもある。""")
print('keybert ja:', tokens)

ke = KeywordExtraction('rake')
tokens = ke.extract_keywords("""「人工知能」という名前は1956年にダートマス会議でジョン・マッカーシーにより命名された。
現在では、記号処理を用いた知能の記述を主体とする情報処理や研究でのアプローチという意味あいでも使われている。
日常語としての「人工知能」という呼び名は非常に曖昧なものになっており、多少気の利いた家庭用電気機械器具の制御システムやゲームソフトの思考ルーチンなどがこう呼ばれることもある。""")
print('rake ja:', tokens)


ke = KeywordExtraction('keybert')
tokens = ke.extract_keywords('什么时候才能够添加修改微信号的窗口呢？')
print('keybert cn:', tokens)

ke = KeywordExtraction('rake')
tokens = ke.extract_keywords('什么时候才能够添加修改微信号的窗口呢？')
print('rake cn:', tokens)
