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

import pytest

from bigdl.dllib.feature.text import *
from test.bigdl.test_zoo_utils import ZooTestCase

text = "Hello my friend, please annotate my text"


class TestTextFeature(ZooTestCase):

    def test_text_feature_with_label(self):
        feature = TextFeature(text, 1)
        assert feature.get_text() == text
        assert feature.get_label() == 1
        assert feature.has_label()
        assert set(feature.keys()) == {'text', 'label'}
        assert feature.get_tokens() is None
        assert feature.get_sample() is None

    def test_text_feature_without_label(self):
        feature = TextFeature(text)
        assert feature.get_text() == text
        assert feature.get_label() == -1
        assert not feature.has_label()
        assert feature.keys() == ['text']
        feature.set_label(0.)
        assert feature.get_label() == 0
        assert feature.has_label()
        assert set(feature.keys()) == {'text', 'label'}
        assert feature.get_tokens() is None
        assert feature.get_sample() is None

    def test_text_feature_transformation(self):
        feature = TextFeature(text, 0)
        tokenizer = Tokenizer()
        tokenized = tokenizer.transform(feature)
        assert tokenized.get_tokens() == \
            ['Hello', 'my', 'friend,', 'please', 'annotate', 'my', 'text']
        normalizer = Normalizer()
        normalized = normalizer.transform(tokenized)
        assert normalized.get_tokens() == \
            ['hello', 'my', 'friend', 'please', 'annotate', 'my', 'text']
        word_index = {"my": 1, "please": 2, "friend": 3}
        indexed = WordIndexer(word_index).transform(normalized)
        shaped = SequenceShaper(5).transform(indexed)
        transformed = TextFeatureToSample().transform(shaped)
        assert set(transformed.keys()) == {'text', 'label', 'tokens', 'indexedTokens', 'sample'}
        sample = transformed.get_sample()
        assert list(sample.feature.storage) == [1., 3., 2., 1., 0.]
        assert list(sample.label.storage) == [0.]

    def test_text_feature_with_uri(self):
        feature = TextFeature(uri="A1")
        assert feature.get_text() is None
        assert feature.get_uri() == "A1"


if __name__ == "__main__":
    pytest.main([__file__])
