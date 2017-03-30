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

# Still in experimental stage!

import itertools
import re
from optparse import OptionParser

from dataset import news20
from nn.layer import *
from nn.criterion import *
from optim.optimizer import *
from util.common import *
from util.common import Sample


def text_to_words(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words


def analyze_texts(data_rdd):
    return data_rdd.flatMap(lambda (text, label): text_to_words(text)) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda (w, c): - c).zipWithIndex() \
        .map(lambda ((w, c), i): (w, (i + 1, c))).collect()


# pad([1, 2, 3, 4, 5], 0, 6)
def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l


def to_vec(token, b_w2v, embedding_dim):
    if token in b_w2v:
        return b_w2v[token]
    else:
        return pad([], 0, embedding_dim)


def to_sample(vectors, label, embedding_dim):
    # flatten nested list
    flatten_features = list(itertools.chain(*vectors))
    features = np.array(flatten_features, dtype='float').reshape(
        [sequence_len, embedding_dim])

    if model_type.lower() == "cnn":
        features = features.transpose(1, 0)
    return Sample.from_ndarray(features, np.array(label))


def build_model(class_num):
    model = Sequential()

    if model_type.lower() == "cnn":
        model.add(Reshape([embedding_dim, 1, sequence_len]))
        model.add(SpatialConvolution(embedding_dim, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(SpatialConvolution(128, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(Reshape([128]))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128)))
        model.add(Select(2, -1))
    else:
        raise ValueError('model can only be cnn, lstm, or gru')

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model


def train(sc,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split):
    print('Processing text dataset')
    texts = news20.get_news20()
    data_rdd = sc.parallelize(texts, 2)

    word_to_ic = analyze_texts(data_rdd)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic[10: max_words])
    bword_to_ic = sc.broadcast(word_to_ic)

    w2v = news20.get_glove_w2v(dim=embedding_dim)
    filtered_w2v = {w: v for w, v in w2v.items() if w in word_to_ic}
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda (text, label):
                              ([w for w in text_to_words(text) if
                                w in bword_to_ic.value], label))
    padded_tokens_rdd = tokens_rdd.map(
        lambda (tokens, label): (pad(tokens, "##", sequence_len), label))
    vector_rdd = padded_tokens_rdd.map(lambda (tokens, label):
                                       ([to_vec(w, bfiltered_w2v.value,
                                                embedding_dim) for w in
                                         tokens], label))
    sample_rdd = vector_rdd.map(
        lambda (vectors, label): to_sample(vectors, label, embedding_dim))

    train_rdd, val_rdd = sample_rdd.randomSplit(
        [training_split, 1-training_split])

    state = {"learningRate": 0.01,
             "learningRateDecay": 0.0002}

    optimizer = Optimizer(
        model=build_model(news20.CLASS_NUM),
        training_rdd=train_rdd,
        criterion=ClassNLLCriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method="Adagrad",
        state=state)

    optimizer.setvalidation(
        batch_size=batch_size,
        val_rdd=val_rdd,
        trigger=EveryEpoch(),
        val_method=["Top1Accuracy"]
    )
    train_model = optimizer.optimize()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")  # noqa
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="15")
    parser.add_option("--model", dest="model_type", default="cnn")

    (options, args) = parser.parse_args(sys.argv)
    if options.action == "train":
        batch_size = int(options.batchSize)
        embedding_dim = int(options.embedding_dim)
        max_epoch = int(options.max_epoch)
        model_type = options.model_type
        sequence_len = 50
        max_words = 1000
        training_split = 0.8
        sc = SparkContext(appName="text_classifier",
                          conf=create_spark_conf())
        init_engine()
        train(sc,
              batch_size,
              sequence_len, max_words, embedding_dim, training_split)
    elif options.action == "test":
        pass
