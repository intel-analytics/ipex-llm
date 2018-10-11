#
# Copyright 2018 Analytics Zoo Authors.
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

import six
from bigdl.util.common import JavaValue, callBigDlFunc
from pyspark import RDD


class TextSet(JavaValue):
    """
    TextSet wraps a set of texts with status.
    """
    def __init__(self, jvalue, bigdl_type="float", *args):
        super(TextSet, self).__init__(jvalue, bigdl_type, *args)

    def is_local(self):
        """
        Whether it is a LocalTextSet.

        :return: Boolean
        """
        return callBigDlFunc(self.bigdl_type, "textSetIsLocal", self.value)

    def is_distributed(self):
        """
        Whether it is a DistributedTextSet.

        :return: Boolean
        """
        return callBigDlFunc(self.bigdl_type, "textSetIsDistributed", self.value)

    def to_distributed(self, sc=None, partition_num=4):
        """
        Convert to a DistributedTextSet.

        Need to specify SparkContext to convert a LocalTextSet to a DistributedTextSet.
        In this case, you may also want to specify partition_num, the default of which is 4.

        :return: DistributedTextSet
        """
        if self.is_distributed():
            jvalue = self.value
        else:
            assert sc, "sc cannot be null to transform a LocalTextSet to a DistributedTextSet"
            jvalue = callBigDlFunc(self.bigdl_type, "textSetToDistributed", self.value,
                                   sc, partition_num)
        return DistributedTextSet(jvalue=jvalue)

    def to_local(self):
        """
        Convert to a LocalTextSet.

        :return: LocalTextSet
        """
        if self.is_local():
            jvalue = self.value
        else:
            jvalue = callBigDlFunc(self.bigdl_type, "textSetToLocal", self.value)
        return LocalTextSet(jvalue=jvalue)

    def get_word_index(self):
        """
        Get the word index dictionary of the TextSet.
        If the TextSet hasn't been transformed from word to index, None will be returned.

        :return: Dictionary {word: id}
        """
        return callBigDlFunc(self.bigdl_type, "textSetGetWordIndex", self.value)

    def generate_word_index_map(self, remove_topN=0, max_words_num=-1):
        """
        Generate word_index map based on sorted word frequencies in descending order.
        Return the result dictionary, which can also be retrieved by 'get_word_index()'.
        Make sure you call this after tokenize. Otherwise you will get an error.
        See word2idx for more details.

        :return: Dictionary {word: id}
        """
        return callBigDlFunc(self.bigdl_type, "textSetGenerateWordIndexMap", self.value,
                             remove_topN, max_words_num)

    def get_texts(self):
        """
        Get the text contents of a TextSet.

        :return: List of String for LocalTextSet.
                 RDD of String for DistributedTextSet.
        """
        return callBigDlFunc(self.bigdl_type, "textSetGetTexts", self.value)

    def get_labels(self):
        """
        Get the labels of a TextSet (if any).
        If a text doesn't have a label, its corresponding position will be -1.

        :return: List of int for LocalTextSet.
                 RDD of int for DistributedTextSet.
        """
        return callBigDlFunc(self.bigdl_type, "textSetGetLabels", self.value)

    def get_predicts(self):
        """
        Get the prediction results of a TextSet (if any).
        If a text hasn't been predicted by a model, its corresponding position will be None.

        :return: List of list of numpy array for LocalTextSet.
                 RDD of list of numpy array for DistributedTextSet.
        """
        predicts = callBigDlFunc(self.bigdl_type, "textSetGetPredicts", self.value)
        if isinstance(predicts, RDD):
            return predicts.map(lambda predict: _process_predict_result(predict))
        else:
            return [_process_predict_result(predict) for predict in predicts]

    def get_samples(self):
        """
        Get the BigDL Sample representations of a TextSet (if any).
        If a text hasn't been transformed to Sample, its corresponding position will be None.

        :return: List of Sample for LocalTextSet.
                 RDD of Sample for DistributedTextSet.
        """
        return callBigDlFunc(self.bigdl_type, "textSetGetSamples", self.value)

    def random_split(self, weights):
        """
        Randomly split into list of TextSet with provided weights.
        Only available for DistributedTextSet for now.

        :param weights: List of float indicating the split portions.
        """
        jvalues = callBigDlFunc(self.bigdl_type, "textSetRandomSplit", self.value, weights)
        return [TextSet(jvalue=jvalue) for jvalue in list(jvalues)]

    def tokenize(self):
        """
        Do tokenization on original text.
        See Tokenizer for more details.

        :return: TextSet after tokenization.
        """
        jvalue = callBigDlFunc(self.bigdl_type, "textSetTokenize", self.value)
        return TextSet(jvalue=jvalue)

    def normalize(self):
        """
        Do normalization on tokens.
        See Normalizer for more details.

        :return: TextSet after normalization.
        """
        jvalue = callBigDlFunc(self.bigdl_type, "textSetNormalize", self.value)
        return TextSet(jvalue=jvalue)

    def shape_sequence(self, len, mode="pre"):
        """
        Shape the sequence of tokens to a fixed length. Padding element will be "##".
        See SequenceShaper for more details.

        :return: TextSet after sequence shaping.
        """
        jvalue = callBigDlFunc(self.bigdl_type, "textSetShapeSequence", self.value, len, mode)
        return TextSet(jvalue=jvalue)

    def word2idx(self, remove_topN=0, max_words_num=-1):
        """
        Map word tokens to indices.
        Index will start from 1 and corresponds to the occurrence frequency of each word sorted
        in descending order.
        See WordIndexer for more details.

        :param remove_topN: Int. Remove the topN words with highest frequencies in the case
                            where those are treated as stopwords.
                            Default is 0, namely remove nothing.
        :param max_words_num: Int. The maximum number of words to be taken into consideration.
                              Default is -1, namely all words will be considered.
        :return: TextSet after word2idx.
        """
        jvalue = callBigDlFunc(self.bigdl_type, "textSetWord2idx", self.value,
                               remove_topN, max_words_num)
        return TextSet(jvalue=jvalue)

    def generate_sample(self):
        """
        Generate BigDL Sample.
        See TextFeatureToSample for more details.

        :return: TextSet with Samples.
        """
        jvalue = callBigDlFunc(self.bigdl_type, "textSetGenerateSample", self.value)
        return TextSet(jvalue=jvalue)

    def transform(self, transformer, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "transformTextSet", transformer, self.value)
        return self

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, bigdl_type="float"):
        """
        Read text files as TextSet.
        If sc is defined, read texts as DistributedTextSet from local file system or HDFS.
        If sc is None, read texts as LocalTextSet from local file system.

        :param path: String. Folder path to texts.
               The folder structure is expected to be the following:
                   path
                     |dir1 - text1, text2, ...
                     |dir2 - text1, text2, ...
                     |dir3 - text1, text2, ...
               Under the target path, there ought to be N subdirectories (dir1 to dirN). Each
               subdirectory represents a category and contains all texts that belong to such
               category. Each category will be a given a label according to its position in the
               ascending order sorted among all subdirectories.
               All texts will be given a label according to the subdirectory where it is located.
               Labels start from 0.
        :param sc: An instance of SparkContext if any. Default is None.
        :param min_partitions: A suggestion value of the minimal partition number.
                               Int. Default is 1. Only need to specify this when sc is not None.
        :return: TextSet.
        """
        jvalue = callBigDlFunc(bigdl_type, "readTextSet", path, sc, min_partitions)
        return TextSet(jvalue=jvalue)


class LocalTextSet(TextSet):
    """
    LocalTextSet is comprised of lists.
    """
    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type="float"):
        """
        Create a LocalTextSet using texts and labels.

        # Arguments:
        texts: List of String. Each element is the content of a text.
        labels: List of int or None if texts don't have labels.
        """
        if texts is not None:
            assert all(isinstance(text, six.string_types) for text in texts),\
                "texts for LocalTextSet should be list of string"
        if labels is not None:
            labels = [int(label) for label in labels]
        super(LocalTextSet, self).__init__(jvalue, bigdl_type, texts, labels)


class DistributedTextSet(TextSet):
    """
    DistributedTextSet is comprised of RDDs.
    """
    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type="float"):
        """
        Create a DistributedTextSet using texts and labels.

        # Arguments:
        texts: RDD of String. Each element is the content of a text.
        labels: RDD of int or None if texts don't have labels.
        """
        if texts is not None:
            assert isinstance(texts, RDD), "texts for DistributedTextSet should be RDD of String"
        if labels is not None:
            assert isinstance(labels, RDD), "labels for DistributedTextSet should be RDD of int"
            labels = labels.map(lambda x: int(x))
        super(DistributedTextSet, self).__init__(jvalue, bigdl_type, texts, labels)


def _process_predict_result(predict):
    # 'predict' is a list of JTensors or None
    # convert to a list of ndarray
    if predict is not None:
        return [res.to_ndarray() for res in predict]
    else:
        return None
