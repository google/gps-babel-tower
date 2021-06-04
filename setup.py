import os
import re
import shutil

from setuptools import find_packages, setup


install_requires = [
  'nltk~=3.2.5',
  'transformers~=4.6.1',
  'tensorflow~=2.4.1',
  'torch~=1.8.1',
  'langdetect>=1.0.8',
  'sentencepiece>=0.1.85',
  'datasets~=1.5.0',
  'faiss-cpu~=1.7.0',
  'google-cloud-translate>=3.0.2',
  'google-cloud-language>=2.0.0',
  'fasttext>=0.9.2',
  'langid>=1.1.6',
  'pycld3>=0.20',
  'gspread>=3.6.0',
  'gspread-dataframe>=3.2.0',
  'spacy~=3.0.5',
  'rake-nltk~=1.0.4',
  'keybert~=0.3.0',
  'mecab-python3>=0.7',
  'jieba~=0.42.1',
  'nagisa~=0.2.7',
  'unidic-lite~=1.0.8',
  'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm',
]


setup(
    name="gps-babel-tower",
    version="0.0.1", # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="chii",
    author_email="chii@google.com",
    description="Babel Tower NLP Platform",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require={},
    python_requires=">=3.6.0",
    install_requires=install_requires,
)