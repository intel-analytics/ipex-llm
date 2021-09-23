# #
# # Copyright 2016 The BigDL Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#
# import pytest
# import shutil
#
# from bigdl.dllib.optim.optimizer import SGD
# from bigdl.dllib.feature.common import ChainedPreprocessing, Relations
# from bigdl.dllib.feature.text import *
# from bigdl.dllib.utils.nncontext import *
# from zoo.models.textclassification import TextClassifier
# from zoo.models.textmatching import KNRM
# from zoo.pipeline.api.keras.models import Sequential
# from zoo.pipeline.api.keras.layers import TimeDistributed
# from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy
#
#
# class TestTextSet:
#
#     def setup_method(self, method):
#         """ setup any state tied to the execution of the given method in a
#         class.  setup_method is invoked for every test method of a class.
#         """
#         self.sc = init_nncontext(init_spark_conf().setMaster("local[1]")
#                                  .setAppName("test text set"))
#         text1 = "Hello my friend, please annotate my text"
#         text2 = "hello world, this is some sentence for my test"
#         text3 = "another text for test"
#         self.texts = [text1, text2, text3]
#         self.labels = [0., 1, 1]
#         resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
#         self.path = os.path.join(resource_path, "news20")
#         self.glove_path = os.path.join(resource_path, "glove.6B/glove.6B.50d.txt")
#         self.qa_path = os.path.join(resource_path, "qa")
#
#     def teardown_method(self, method):
#         """ teardown any state that was previously setup with a setup_method
#         call.
#         """
#         self.sc.stop()
#
#     def test_textset_without_label(self):
#         local_set = LocalTextSet(self.texts)
#         assert local_set.get_labels() == [-1, -1, -1]
#         distributed_set = DistributedTextSet(self.sc.parallelize(self.texts))
#         assert distributed_set.get_labels().collect() == [-1, -1, -1]
#
#     def test_textset_convertion(self):
#         local_set = LocalTextSet(self.texts, self.labels)
#         local1 = local_set.to_local()
#         distributed1 = local_set.to_distributed(self.sc)
#         assert local1.is_local()
#         assert distributed1.is_distributed()
#         assert local1.get_texts() == distributed1.get_texts().collect()
#
#         texts_rdd = self.sc.parallelize(self.texts)
#         labels_rdd = self.sc.parallelize(self.labels)
#         distributed_set = DistributedTextSet(texts_rdd, labels_rdd)
#         local2 = distributed_set.to_local()
#         distributed2 = distributed_set.to_distributed()
#         assert local2.is_local()
#         assert distributed2.is_distributed()
#         assert local2.get_texts() == distributed2.get_texts().collect()
#
#     def test_local_textset_integration(self):
#         local_set = LocalTextSet(self.texts, self.labels)
#         assert local_set.is_local()
#         assert not local_set.is_distributed()
#         assert local_set.get_texts() == self.texts
#         assert local_set.get_labels() == self.labels
#         tokenized = ChainedPreprocessing([Tokenizer(), Normalizer()])(local_set)
#         word_index = tokenized.generate_word_index_map(max_words_num=10)
#         transformed = ChainedPreprocessing([WordIndexer(word_index), SequenceShaper(10),
#                                             TextFeatureToSample()])(tokenized)
#         assert transformed.is_local()
#         word_index = transformed.get_word_index()
#         assert len(word_index) == 10
#         assert word_index["my"] == 1
#         samples = transformed.get_samples()
#         assert len(samples) == 3
#         for sample in samples:
#             assert sample.feature.shape[0] == 10
#
#         vocab_file = create_tmp_path() + ".txt"
#         transformed.save_word_index(vocab_file)
#         local_set2 = LocalTextSet(self.texts, self.labels)
#         local_set2.load_word_index(vocab_file)
#         transformed2 = local_set2.tokenize().normalize().word2idx()\
#             .shape_sequence(10).generate_sample()
#         samples2 = transformed2.get_samples()
#         for s1, s2 in zip(samples, samples2):
#             assert np.allclose(s1.feature.to_ndarray(), s2.feature.to_ndarray())
#         os.remove(vocab_file)
#
#         model = TextClassifier(5, self.glove_path, word_index, 10)
#         model.compile("adagrad", "sparse_categorical_crossentropy", ['accuracy'])
#         tmp_log_dir = create_tmp_path()
#         tmp_checkpoint_path = create_tmp_path()
#         os.mkdir(tmp_checkpoint_path)
#         model.set_tensorboard(tmp_log_dir, "textclassification")
#         model.set_checkpoint(tmp_checkpoint_path)
#         model.fit(transformed, batch_size=2, nb_epoch=2, validation_data=transformed)
#         acc = model.evaluate(transformed, batch_size=2)
#         res_set = model.predict(transformed, batch_per_thread=2)
#         predicts = res_set.get_predicts()
#
#         # Test for loaded model predict on TextSet
#         tmp_path = create_tmp_path() + ".bigdl"
#         model.save_model(tmp_path, over_write=True)
#         loaded_model = TextClassifier.load_model(tmp_path)
#         loaded_res_set = loaded_model.predict(transformed, batch_per_thread=2)
#         loaded_predicts = loaded_res_set.get_predicts()
#         assert len(predicts) == len(loaded_predicts)
#
#         for i in range(0, len(predicts)):  # (uri, prediction)
#             assert not predicts[i][0]
#             assert not loaded_predicts[i][0]  # uri is not recorded and thus None
#             assert len(predicts[i][1]) == 1
#             assert len(loaded_predicts[i][1]) == 1
#             assert predicts[i][1][0].shape == (5, )
#             assert np.allclose(predicts[i][1][0], loaded_predicts[i][1][0])
#         shutil.rmtree(tmp_log_dir)
#         shutil.rmtree(tmp_checkpoint_path)
#         os.remove(tmp_path)
#
#     def test_distributed_textset_integration(self):
#         texts_rdd = self.sc.parallelize(self.texts)
#         labels_rdd = self.sc.parallelize(self.labels)
#         distributed_set = DistributedTextSet(texts_rdd, labels_rdd)
#         assert distributed_set.is_distributed()
#         assert not distributed_set.is_local()
#         assert distributed_set.get_texts().collect() == self.texts
#         assert distributed_set.get_labels().collect() == self.labels
#
#         sets = distributed_set.random_split([0.5, 0.5])
#         train_texts = sets[0].get_texts().collect()
#         test_texts = sets[1].get_texts().collect()
#         assert set(train_texts + test_texts) == set(self.texts)
#
#         tokenized = Tokenizer()(distributed_set)
#         transformed = tokenized.normalize().word2idx().shape_sequence(5).generate_sample()
#         word_index = transformed.get_word_index()
#         assert len(word_index) == 14
#         samples = transformed.get_samples().collect()
#         assert len(samples) == 3
#         for sample in samples:
#             assert sample.feature.shape[0] == 5
#
#         vocab_file = create_tmp_path() + ".txt"
#         transformed.save_word_index(vocab_file)
#         distributed_set2 = DistributedTextSet(texts_rdd, labels_rdd)
#         distributed_set2.load_word_index(vocab_file)
#         transformed2 = distributed_set2.tokenize().normalize().word2idx()\
#             .shape_sequence(5).generate_sample()
#         samples2 = transformed2.get_samples().collect()
#         for s1, s2 in zip(samples, samples2):
#             assert np.allclose(s1.feature.to_ndarray(), s2.feature.to_ndarray())
#         os.remove(vocab_file)
#
#         model = TextClassifier(5, self.glove_path, word_index, 5, encoder="lstm")
#         model.compile(SGD(), SparseCategoricalCrossEntropy())
#         model.fit(transformed, batch_size=2, nb_epoch=2)
#         res_set = model.predict(transformed, batch_per_thread=2)
#         predicts = res_set.get_predicts().collect()
#         for predict in predicts:  # (uri, prediction)
#             assert not predict[0]  # uri is not recorded and thus None
#             assert len(predict[1]) == 1
#             assert predict[1][0].shape == (5, )
#
#         tmp_path = create_tmp_path() + ".bigdl"
#         model.save_model(tmp_path, over_write=True)
#         loaded_model = TextClassifier.load_model(tmp_path)
#         loaded_res_set = loaded_model.predict(transformed, batch_per_thread=2)
#         loaded_predicts = loaded_res_set.get_predicts().collect()
#         assert len(loaded_predicts) == len(predicts)
#         os.remove(tmp_path)
#
#     def test_read_local(self):
#         local_set = TextSet.read(self.path)
#         assert local_set.is_local()
#         assert not local_set.get_word_index()  # should be None
#         assert len(local_set.get_texts()) == 3
#         assert local_set.get_labels() == [0, 0, 1]
#         assert local_set.get_samples() == [None, None, None]
#         assert local_set.get_predicts() == [(uri, None) for uri in local_set.get_uris()]
#
#     def test_read_distributed(self):
#         distributed_set = TextSet.read(self.path, self.sc, 4)
#         assert distributed_set.is_distributed()
#         assert not distributed_set.get_word_index()
#         assert len(distributed_set.get_texts().collect()) == 3
#         assert sorted(distributed_set.get_labels().collect()) == [0, 0, 1]
#         assert distributed_set.get_samples().collect() == [None, None, None]
#         assert distributed_set.get_predicts().collect() ==\
#             [(uri, None) for uri in distributed_set.get_uris().collect()]
#
#     def test_read_csv_parquet(self):
#         text_set = TextSet.read_csv(self.qa_path + "/question_corpus.csv", self.sc)
#         text_set2 = TextSet.read_csv(self.qa_path + "/question_corpus.csv")
#         text_set3 = TextSet.read_parquet(self.qa_path + "/question_corpus.parquet", self.sc)
#         assert text_set.is_distributed()
#         assert text_set2.is_local()
#         assert text_set3.is_distributed()
#
#     def test_qaranker_distributed_integration(self):
#         relations = Relations.read(self.qa_path+"/relations.txt", self.sc)
#         assert relations.count() == 4
#         text_set = TextSet.read_csv(self.qa_path+"/question_corpus.csv", self.sc)
#         assert text_set.get_uris().collect() == ["Q1", "Q2"]
#         transformed = text_set.tokenize().normalize().word2idx().shape_sequence(5)
#         relation_pairs = TextSet.from_relation_pairs(relations, transformed, transformed)
#         pair_samples = relation_pairs.get_samples().collect()
#         assert len(pair_samples) == 2
#         for sample in pair_samples:
#             assert list(sample.feature.shape) == [2, 10]
#             assert np.allclose(sample.label.to_ndarray(), np.array([[1.0], [0.0]]))
#         relation_lists = TextSet.from_relation_lists(relations, transformed, transformed)
#         relation_samples = relation_lists.get_samples().collect()
#         assert len(relation_samples) == 2
#         for sample in relation_samples:
#             assert list(sample.feature.shape) == [2, 10]
#             assert list(sample.label.shape) == [2, 1]
#         knrm = KNRM(5, 5, self.glove_path, word_index=transformed.get_word_index())
#         model = Sequential().add(TimeDistributed(knrm, input_shape=(2, 10)))
#         model.compile("sgd", "rank_hinge")
#         model.fit(relation_pairs, batch_size=2, nb_epoch=2)
#         print(knrm.evaluate_ndcg(relation_lists, 3))
#         print(knrm.evaluate_map(relation_lists))
#
#     def test_qaranker_local_integration(self):
#         relations = Relations.read(self.qa_path+"/relations.txt")
#         assert len(relations) == 4
#         text_set = TextSet.read_csv(self.qa_path+"/question_corpus.csv")
#         assert text_set.get_uris() == ["Q1", "Q2"]
#         transformed = text_set.tokenize().normalize().word2idx().shape_sequence(5)
#         relation_pairs = TextSet.from_relation_pairs(relations, transformed, transformed)
#         pair_samples = relation_pairs.get_samples()
#         assert len(pair_samples) == 2
#         for sample in pair_samples:
#             assert list(sample.feature.shape) == [2, 10]
#             assert np.allclose(sample.label.to_ndarray(), np.array([[1.0], [0.0]]))
#         relation_lists = TextSet.from_relation_lists(relations, transformed, transformed)
#         relation_samples = relation_lists.get_samples()
#         assert len(relation_samples) == 2
#         for sample in relation_samples:
#             assert list(sample.feature.shape) == [2, 10]
#             assert list(sample.label.shape) == [2, 1]
#         knrm = KNRM(5, 5, self.glove_path, word_index=transformed.get_word_index())
#         model = Sequential().add(TimeDistributed(knrm, input_shape=(2, 10)))
#         model.compile("sgd", "rank_hinge")
#         model.fit(relation_pairs, batch_size=2, nb_epoch=2)
#         print(knrm.evaluate_ndcg(relation_lists, 3))
#         print(knrm.evaluate_map(relation_lists))
#
#
# if __name__ == "__main__":
#     pytest.main([__file__])
