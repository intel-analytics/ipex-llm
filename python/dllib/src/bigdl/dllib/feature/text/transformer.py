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

import sys
import six
from bigdl.dllib.feature.common import Preprocessing
from bigdl.dllib.feature.text import TextFeature
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


class TextTransformer(Preprocessing):
    """
    Base class of Transformers that transform TextFeature.
    """

    def __init__(self, bigdl_type="float", *args):
        super(TextTransformer, self).__init__(bigdl_type, *args)

    def transform(self, text_feature):
        """
        Transform a TextFeature.
        """
        res = callZooFunc(self.bigdl_type, "transformTextFeature", self.value, text_feature.value)
        return TextFeature(jvalue=res)


class Tokenizer(TextTransformer):
    """
    Transform text to array of string tokens.

    >>> tokenizer = Tokenizer()
    creating: createTokenizer
    """

    def __init__(self, bigdl_type="float"):
        super(Tokenizer, self).__init__(bigdl_type)


class Normalizer(TextTransformer):
    """
    Removes all dirty characters (non English alphabet) from tokens and converts words to
    lower case. Need to tokenize first.
    Original tokens will be replaced by normalized tokens.

    >>> normalizer = Normalizer()
    creating: createNormalizer
    """

    def __init__(self, bigdl_type="float"):
        super(Normalizer, self).__init__(bigdl_type)


class WordIndexer(TextTransformer):
    """
    Given a wordIndex map, transform tokens to corresponding indices.
    Those words not in the map will be aborted.
    Need to tokenize first.

    # Arguments
    map: Dict with word (string) as its key and index (int) as its value.

    >>> word_indexer = WordIndexer(map={"it": 1, "me": 2})
    creating: createWordIndexer
    """

    def __init__(self, map, bigdl_type="float"):
        super(WordIndexer, self).__init__(bigdl_type, map)


class SequenceShaper(TextTransformer):
    """
    Shape the sequence of indices to a fixed length.
    If the original sequence is longer than the target length, it will be truncated from
    the beginning or the end.
    If the original sequence is shorter than the target length, it will be padded to the end.
    Need to word2idx first.
    The original indices sequence will be replaced by the shaped sequence.

    # Arguments
    len: Positive int. The target length.
    trunc_mode: Truncation mode. String. Either 'pre' or 'post'. Default is 'pre'.
                If 'pre', the sequence will be truncated from the beginning.
                If 'post', the sequence will be truncated from the end.
    pad_element: Int. The element to be padded to the sequence if the original length is
                 smaller than the target length.
                 Default is 0 with the convention that we reserve index 0 for unknown words.
    >>> sequence_shaper = SequenceShaper(len=6, trunc_mode="post", pad_element=10000)
    creating: createSequenceShaper
    """

    def __init__(self, len, trunc_mode="pre", pad_element=0, bigdl_type="float"):
        invalidInputError(isinstance(pad_element, int), "pad_element should be an int")
        super(SequenceShaper, self).__init__(bigdl_type, len, trunc_mode, pad_element)


class TextFeatureToSample(TextTransformer):
    """
    Transform indexedTokens and label (if any) of a TextFeature to a BigDL Sample.
    Need to word2idx first.

    >>> to_sample = TextFeatureToSample()
    creating: createTextFeatureToSample
    """

    def __init__(self, bigdl_type="float"):
        super(TextFeatureToSample, self).__init__(bigdl_type)
