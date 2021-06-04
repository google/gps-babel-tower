from . import spacy_keyword_extraction
from rake_nltk import Rake
from keybert import KeyBERT
from gps_babel_tower.tasks.lang_detect import LangDetect
from gps_babel_tower.tasks.word_seg import WordSegment
import logging
from . import rake_ja
import unidic_lite


class KeywordExtraction:
  def __init__(self, kw_model, min_len=1, max_len=3):
    self.kw_model = kw_model
    self.min_len = min_len
    self.max_len = max_len
    self.langdetect = LangDetect()
    self.word_seg = WordSegment()
    
    self.ja_tok = rake_ja.Tokenizer(rawargs=f'-r /dev/null -d {unidic_lite.DICDIR}')
    self.ja_rake = rake_ja.JapaneseRake()
    
    if kw_model == 'rake':
      self.model = Rake(min_length=min_len, max_length=max_len)
    elif kw_model == 'keybert':
      self.model = KeyBERT('xlm-r-100langs-bert-base-nli-mean-tokens')
    elif kw_model == 'spacy_bigram':
      self.model = spacy_keyword_extraction.KeywordExtraction()
    else:
      raise Exception(f'Unsupported keyword extraction model type: {kw_model}, can only be rake|keybert|spacy_bigram')
      
  
  def extract_keywords(self, text, lang=None, max_results=-1):
    if not lang:
      lang = self.langdetect.get_language(text)

    if self.kw_model == 'rake':
      if lang == 'ja':
        tokens = self.ja_tok.tokenize(text)
        self.ja_rake.extract_keywords_from_text(tokens)
        keywords = self.ja_rake.get_ranked_phrases()
      else:
        self.model.extract_keywords_from_text(text)
        keywords = self.model.get_ranked_phrases()
    elif self.kw_model == 'keybert':
      if lang == 'ja' or lang == 'zh':
        text = self.word_seg.tokenize(text, lang=lang, join_by=' ')
      keywords_and_score = self.model.extract_keywords(text, keyphrase_ngram_range=(self.min_len, self.max_len))
      keywords = [keyword for keyword, _score in keywords_and_score]
    elif self.kw_model == 'spacy_bigram':
      if lang != 'en':
        logging.error(f'spacy_bigram only supports English! language detected: {lang} for text {text}')
        return []
      keywords = self.model.tokenize(text)
      
    return keywords[:max_results]