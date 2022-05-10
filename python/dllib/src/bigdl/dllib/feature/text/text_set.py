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

import six
from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.file_utils import callZooFunc
from pyspark import RDD
from bigdl.dllib.utils.log4Error import *


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
        return callZooFunc(self.bigdl_type, "textSetIsLocal", self.value)

    def is_distributed(self):
        """
        Whether it is a DistributedTextSet.

        :return: Boolean
        """
        return callZooFunc(self.bigdl_type, "textSetIsDistributed", self.value)

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
            invalidInputError(sc,
                              "sc cannot be null to transform a LocalTextSet to a"
                              " DistributedTextSet")
            jvalue = callZooFunc(self.bigdl_type, "textSetToDistributed", self.value,
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
            jvalue = callZooFunc(self.bigdl_type, "textSetToLocal", self.value)
        return LocalTextSet(jvalue=jvalue)

    def get_word_index(self):
        """
        Get the word_index dictionary of the TextSet.
        If the TextSet hasn't been transformed from word to index, None will be returned.

        :return: Dictionary {word: id}
        """
        return callZooFunc(self.bigdl_type, "textSetGetWordIndex", self.value)

    def save_word_index(self, path):
        """
        Save the word_index dictionary to text file, which can be used for future inference.
        Each separate line will be "word id".

        For LocalTextSet, save txt to a local file system.
        For DistributedTextSet, save txt to a local or distributed file system (such as HDFS).

        :param path: The path to the text file.
        """
        callZooFunc(self.bigdl_type, "textSetSaveWordIndex", self.value, path)

    def load_word_index(self, path):
        """
        Load the word_index map which was saved after the training, so that this TextSet can
        directly use this word_index during inference.
        Each separate line should be "word id".

        Note that after calling `load_word_index`, you do not need to specify any argument when
        calling `word2idx` in the preprocessing pipeline as now you are using exactly the loaded
        word_index for transformation.

        For LocalTextSet, load txt from a local file system.
        For DistributedTextSet, load txt from a local or distributed file system (such as HDFS).

        :return: TextSet with the loaded word_index.
        """
        jvalue = callZooFunc(self.bigdl_type, "textSetLoadWordIndex", self.value, path)
        return TextSet(jvalue=jvalue)

    def set_word_index(self, vocab):
        """
        Assign a word_index dictionary for this TextSet to use during word2idx.
        If you load the word_index from the saved file, you are recommended to use `load_word_index`
        directly.

        :return: TextSet with the word_index set.
        """
        jvalue = callZooFunc(self.bigdl_type, "textSetSetWordIndex", self.value, vocab)
        return TextSet(jvalue=jvalue)

    def generate_word_index_map(self, remove_topN=0, max_words_num=-1,
                                min_freq=1, existing_map=None):
        """
        Generate word_index map based on sorted word frequencies in descending order.
        Return the result dictionary, which can also be retrieved by 'get_word_index()'.
        Make sure you call this after tokenize. Otherwise you will get an error.
        See word2idx for more details.

        :return: Dictionary {word: id}
        """
        return callZooFunc(self.bigdl_type, "textSetGenerateWordIndexMap", self.value,
                           remove_topN, max_words_num, min_freq, existing_map)

    def get_texts(self):
        """
        Get the text contents of a TextSet.

        :return: List of String for LocalTextSet.
                 RDD of String for DistributedTextSet.
        """
        return callZooFunc(self.bigdl_type, "textSetGetTexts", self.value)

    def get_uris(self):
        """
        Get the identifiers of a TextSet.
        If a text doesn't have a uri, its corresponding position will be None.

        :return: List of String for LocalTextSet.
                 RDD of String for DistributedTextSet.
        """
        return callZooFunc(self.bigdl_type, "textSetGetURIs", self.value)

    def get_labels(self):
        """
        Get the labels of a TextSet (if any).
        If a text doesn't have a label, its corresponding position will be -1.

        :return: List of int for LocalTextSet.
                 RDD of int for DistributedTextSet.
        """
        return callZooFunc(self.bigdl_type, "textSetGetLabels", self.value)

    def get_predicts(self):
        """
        Get the prediction results (if any) combined with uris (if any) of a TextSet.
        If a text doesn't have a uri, its corresponding uri will be None.
        If a text hasn't been predicted by a model, its corresponding prediction will be None.

        :return: List of (uri, prediction as a list of numpy array) for LocalTextSet.
                 RDD of (uri, prediction as a list of numpy array) for DistributedTextSet.
        """
        predicts = callZooFunc(self.bigdl_type, "textSetGetPredicts", self.value)
        if isinstance(predicts, RDD):
            return predicts.map(lambda predict: (predict[0], _process_predict_result(predict[1])))
        else:
            return [(predict[0], _process_predict_result(predict[1])) for predict in predicts]

    def get_samples(self):
        """
        Get the BigDL Sample representations of a TextSet (if any).
        If a text hasn't been transformed to Sample, its corresponding position will be None.

        :return: List of Sample for LocalTextSet.
                 RDD of Sample for DistributedTextSet.
        """
        return callZooFunc(self.bigdl_type, "textSetGetSamples", self.value)

    def random_split(self, weights):
        """
        Randomly split into list of TextSet with provided weights.
        Only available for DistributedTextSet for now.

        :param weights: List of float indicating the split portions.
        """
        jvalues = callZooFunc(self.bigdl_type, "textSetRandomSplit", self.value, weights)
        return [TextSet(jvalue=jvalue) for jvalue in list(jvalues)]

    def tokenize(self):
        """
        Do tokenization on original text.
        See Tokenizer for more details.

        :return: TextSet after tokenization.
        """
        jvalue = callZooFunc(self.bigdl_type, "textSetTokenize", self.value)
        return TextSet(jvalue=jvalue)

    def normalize(self):
        """
        Do normalization on tokens.
        Need to tokenize first.
        See Normalizer for more details.

        :return: TextSet after normalization.
        """
        jvalue = callZooFunc(self.bigdl_type, "textSetNormalize", self.value)
        return TextSet(jvalue=jvalue)

    def word2idx(self, remove_topN=0, max_words_num=-1, min_freq=1, existing_map=None):
        """
        Map word tokens to indices.
        Important: Take care that this method behaves a bit differently for training and inference.

        ---------------------------------------Training--------------------------------------------
        During the training, you need to generate a new word_index dictionary according to the texts
        you are dealing with. Thus this method will first do the dictionary generation and then
        convert words to indices based on the generated dictionary.

        You can specify the following arguments which pose some constraints when generating
        the dictionary.
        In the result dictionary, index will start from 1 and corresponds to the occurrence
        frequency of each word sorted in descending order.
        Here we adopt the convention that index 0 will be reserved for unknown words.
        After word2idx, you can get the generated word_index dictionary by calling 'get_word_index'.
        Also, you can call `save_word_index` to save this word_index dictionary to be used in
        future training.

        :param remove_topN: Non-negative int. Remove the topN words with highest frequencies
                            in the case where those are treated as stopwords.
                            Default is 0, namely remove nothing.
        :param max_words_num: Int. The maximum number of words to be taken into consideration.
                              Default is -1, namely all words will be considered.
                              Otherwise, it should be a positive int.
        :param min_freq: Positive int. Only those words with frequency >= min_freq will be taken
                         into consideration.
                         Default is 1, namely all words that occur will be considered.
        :param existing_map: Existing dictionary of word_index if any.
                             Default is None and in this case a new dictionary with index starting
                             from 1 will be generated.
                             If not None, then the generated dictionary will preserve the word_index
                             in existing_map and assign subsequent indices to new words.

        ---------------------------------------Inference--------------------------------------------
        During the inference, you are supposed to use exactly the same word_index dictionary as in
        the training stage instead of generating a new one.
        Thus please be aware that you do not need to specify any of the above arguments.
        You need to call `load_word_index` or `set_word_index` beforehand for dictionary loading.

        Need to tokenize first.
        See WordIndexer for more details.

        :return: TextSet after word2idx.
        """
        jvalue = callZooFunc(self.bigdl_type, "textSetWord2idx", self.value,
                             remove_topN, max_words_num, min_freq, existing_map)
        return TextSet(jvalue=jvalue)

    def shape_sequence(self, len, trunc_mode="pre", pad_element=0):
        """
        Shape the sequence of indices to a fixed length.
        Need to word2idx first.
        See SequenceShaper for more details.

        :return: TextSet after sequence shaping.
        """
        invalidInputError(isinstance(pad_element, int), "pad_element should be an int")
        jvalue = callZooFunc(self.bigdl_type, "textSetShapeSequence", self.value,
                             len, trunc_mode, pad_element)
        return TextSet(jvalue=jvalue)

    def generate_sample(self):
        """
        Generate BigDL Sample.
        Need to word2idx first.
        See TextFeatureToSample for more details.

        :return: TextSet with Samples.
        """
        jvalue = callZooFunc(self.bigdl_type, "textSetGenerateSample", self.value)
        return TextSet(jvalue=jvalue)

    def transform(self, transformer):
        return TextSet(callZooFunc(self.bigdl_type, "transformTextSet",
                                   transformer, self.value), self.bigdl_type)

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, bigdl_type="float"):
        """
        Read text files with labels from a directory.
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

        :param path: The folder path to texts. Local or distributed file system (such as HDFS)
                     are supported. If you want to read from a distributed file system, sc
                     needs to be specified.
        :param sc: An instance of SparkContext.
                   If specified, texts will be read as a DistributedTextSet.
                   Default is None and in this case texts will be read as a LocalTextSet.
        :param min_partitions: Int. A suggestion value of the minimal partition number for input
                               texts. Only need to specify this when sc is not None. Default is 1.

        :return: TextSet.
        """
        jvalue = callZooFunc(bigdl_type, "readTextSet", path, sc, min_partitions)
        return TextSet(jvalue=jvalue)

    @classmethod
    def read_csv(cls, path, sc=None, min_partitions=1, bigdl_type="float"):
        """
        Read texts with id from csv file.
        Each record is supposed to contain the following two fields in order:
        id(string) and text(string).
        Note that the csv file should be without header.

        :param path: The path to the csv file. Local or distributed file system (such as HDFS)
                     are supported. If you want to read from a distributed file system, sc
                     needs to be specified.
        :param sc: An instance of SparkContext.
                   If specified, texts will be read as a DistributedTextSet.
                   Default is None and in this case texts will be read as a LocalTextSet.
        :param min_partitions: Int. A suggestion value of the minimal partition number for input
                               texts. Only need to specify this when sc is not None. Default is 1.

        :return: TextSet.
        """
        jvalue = callZooFunc(bigdl_type, "textSetReadCSV", path, sc, min_partitions)
        return TextSet(jvalue=jvalue)

    @classmethod
    def read_parquet(cls, path, sc, bigdl_type="float"):
        """
        Read texts with id from parquet file.
        Schema should be the following:
        "id"(string) and "text"(string).

        :param path: The path to the parquet file.
        :param sc: An instance of SparkContext.

        :return: DistributedTextSet.
        """
        jvalue = callZooFunc(bigdl_type, "textSetReadParquet", path, sc)
        return DistributedTextSet(jvalue=jvalue)

    @classmethod
    def from_relation_pairs(cls, relations, corpus1, corpus2, bigdl_type="float"):
        """
        Used to generate a TextSet for pairwise training.

        This method does the following:
        1. Generate all RelationPairs: (id1, id2Positive, id2Negative) from Relations.
        2. Join RelationPairs with corpus to transform id to indexedTokens.
        Note: Make sure that the corpus has been transformed by SequenceShaper and WordIndexer.
        3. For each pair, generate a TextFeature having Sample with:
        - feature of shape (2, text1Length + text2Length).
        - label of value [1 0] as the positive relation is placed before the negative one.

        :param relations: List or RDD of Relation.
        :param corpus1: TextSet that contains all id1 in relations. For each TextFeature in corpus1,
                        text must have been transformed to indexedTokens of the same length.
        :param corpus2: TextSet that contains all id2 in relations. For each TextFeature in corpus2,
                        text must have been transformed to indexedTokens of the same length.
        Note that if relations is a list, then corpus1 and corpus2 must both be LocalTextSet.
        If relations is RDD, then corpus1 and corpus2 must both be DistributedTextSet.

        :return: TextSet.
        """
        if isinstance(relations, RDD):
            relations = relations.map(lambda x: x.to_tuple())
        elif isinstance(relations, list):
            relations = [relation.to_tuple() for relation in relations]
        else:
            invalidInputError(False, "relations should be RDD or list of Relation")
        jvalue = callZooFunc(bigdl_type, "textSetFromRelationPairs", relations, corpus1, corpus2)
        return TextSet(jvalue=jvalue)

    @classmethod
    def from_relation_lists(cls, relations, corpus1, corpus2, bigdl_type="float"):
        """
        Used to generate a TextSet for ranking.

        This method does the following:
        1. For each id1 in relations, find the list of id2 with corresponding label that
        comes together with id1.
        In other words, group relations by id1.
        2. Join with corpus to transform each id to indexedTokens.
        Note: Make sure that the corpus has been transformed by SequenceShaper and WordIndexer.
        3. For each list, generate a TextFeature having Sample with:
        - feature of shape (list_length, text1_length + text2_length).
        - label of shape (list_length, 1).

        :param relations: List or RDD of Relation.
        :param corpus1: TextSet that contains all id1 in relations. For each TextFeature in corpus1,
                        text must have been transformed to indexedTokens of the same length.
        :param corpus2: TextSet that contains all id2 in relations. For each TextFeature in corpus2,
                        text must have been transformed to indexedTokens of the same length.
        Note that if relations is a list, then corpus1 and corpus2 must both be LocalTextSet.
        If relations is RDD, then corpus1 and corpus2 must both be DistributedTextSet.

        :return: TextSet.
        """
        if isinstance(relations, RDD):
            relations = relations.map(lambda x: x.to_tuple())
        elif isinstance(relations, list):
            relations = [relation.to_tuple() for relation in relations]
        else:
            invalidInputError(False, "relations should be RDD or list of Relation")
        jvalue = callZooFunc(bigdl_type, "textSetFromRelationLists", relations, corpus1, corpus2)
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
            invalidInputError(all(isinstance(text, six.string_types) for text in texts),
                              "texts for LocalTextSet should be list of string")
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
            invalidInputError(isinstance(texts, RDD),
                              "texts for DistributedTextSet should be RDD of String")
        if labels is not None:
            invalidInputError(isinstance(labels, RDD),
                              "labels for DistributedTextSet should be RDD of int")
            labels = labels.map(lambda x: int(x))
        super(DistributedTextSet, self).__init__(jvalue, bigdl_type, texts, labels)


def _process_predict_result(predict):
    # 'predict' is a list of JTensors or None
    # convert to a list of ndarray
    if predict is not None:
        return [res.to_ndarray() for res in predict]
    else:
        return None
