import unittest

from gps_babel_tower.tasks.keyword_extraction import KeywordExtraction


class TestKeywordExtraction(unittest.TestCase):

  def test_rake(self):
    ke = KeywordExtraction('rake')
    text = 'Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, file storage, and YouTube.'
    tokens = ke.extract_keywords(text, max_results=5)
    self.assertEqual(len(tokens), 5)
      

if __name__ == '__main__':
  unittest.main()