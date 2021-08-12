#
# Copyright 2016 The BigDL Authors.
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
#

import nltk
import os
import itertools
import sys
from nltk.tokenize import word_tokenize

def read_localfile(fileName):
    lines = []
    with open(fileName) as f:
        for line in f:
            lines.append(line)
    f.close()
    return lines

def sentences_split(line):
    nltk.data.path.append(os.environ.get('PWD'))
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    sentenized = sent_tokenizer.tokenize(line)
    return sentenized

def sentences_bipadding(sent):
    return "SENTENCESTART " + sent + " SENTENCEEND"

def sentence_tokenizer(sentences):
    tokenized_sents = nltk.word_tokenize(sentences)
    return tokenized_sents
