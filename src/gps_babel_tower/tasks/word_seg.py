import jieba
import nagisa

class WordSegment:
  def tokenize(self, text, lang, join_by=None):
    if lang == 'zh':
      words = jieba.cut(text)
    elif lang == 'ja':
      words = nagisa.tagging(text)
      words = words.words
    else:
      raise Exception(f'Unknown language {lang}')
    
    if join_by:
      return join_by.join(words)
    