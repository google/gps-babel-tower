from google.cloud import translate_v2 as translate


class LanguageDetection():
  def __init__(self, engine='gcp'):
    self.engine = engine
    if self.engine == 'gcp':
      self.translate_client = translate.Client()
    else:
      raise RuntimeError(f'engine {engine} not supported yet')
    return

  def detect(self, text):
    language_code = None
    if self.engine == 'gcp':
      result = self.translate_client.detect_language(text)
      # print("Confidence: {}".format(result["confidence"]))
      language_code = result["language"]
    else:
      raise RuntimeError('Unsupported')
    return language_code

