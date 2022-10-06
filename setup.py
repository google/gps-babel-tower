# coding=utf-8
# Copyright 2021 Google LLC..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil

from setuptools import find_packages, setup
from setuptools.command.install import install
from subprocess import getoutput

_deps = [
    'nltk>=3.2.5',
    'transformers>=4.6.1',
    'tensorflow>=2.4.1',
    'torch>=1.8.1',
    'langdetect>=1.0.8',
    'sentencepiece>=0.1.85',
    'datasets>=1.5.0',
    'faiss-cpu>=1.7.0',
    'google-cloud-translate>=3.0.2',
    'google-cloud-language>=2.0.0',
    'fasttext>=0.9.2',
    'langid>=1.1.6',
    'pycld3>=0.20',
    'gspread>=3.6.0',
    'gspread-dataframe>=3.2.0',
    'spacy>=3.0.5',
    'rake-nltk>=1.0.4',
    'keybert>=0.3.0',
    'jieba>=0.42.1',
    'nagisa>=0.2.7',
    'rake-ja>=0.0.1',
    'scipy>=1.7.0',
    'ftfy>=6.0.0',
    'diffusers>=0.3.0',
    'sacremoses>=0.0.53',
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

extras = {}
extras['torch'] = deps_list('torch')
extras['tensorflow'] = deps_list('tensorflow')
extras['langdetect'] = deps_list('langid', 'fasttext', 'pycld3', 'langdetect')
extras['keyword_extraction'] = deps_list('rake-nltk', 'keybert', 'jieba', 'nagisa', 'rake-ja') + extras['langdetect']
extras['nlp'] = deps_list('transformers', 'spacy', 'nltk', 'sentencepiece', 'datasets', 'faiss-cpu', 'sacremoses') + extras['keyword_extraction'] + extras['langdetect']
extras['google'] = deps_list('google-cloud-translate', 'google-cloud-language', 'gspread', 'gspread-dataframe')
extras['image'] = deps_list('diffusers', 'transformers', 'ftfy', 'scipy')


setup(
    name='gps-babel-tower',
    version='0.1.1',  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author='Chi Zhang',
    author_email='chii@google.com',
    description='Babel Tower NLP Library',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='Apache',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    extras_require=extras,
    python_requires='>=3.6.0',
    install_requires=[])
