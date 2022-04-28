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

import itertools
import re
from optparse import OptionParser

from bigdl.dllib.feature.dataset import news20
from bigdl.dllib.nn.layer import *
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.utils.common import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.utils.utils import detect_conda_env_name
import os
import datetime as dt
from bigdl.dllib.utils.log4Error import *


def text_to_words(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words


def analyze_texts(data_rdd):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))

    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0])) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda w_c: - w_c[1]).zipWithIndex() \
        .map(lambda w_c_i: index(w_c_i)).collect()


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

    return Sample.from_ndarray(features, np.array(label))


def build_model(class_num):
    model = Sequential()

    if model_type.lower() == "cnn":
        model.add(TemporalConvolution(embedding_dim, 256, 5)) \
            .add(ReLU()) \
            .add(TemporalMaxPooling(sequence_len - 5 + 1)) \
            .add(Squeeze(2))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 256, p)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 256, p)))
        model.add(Select(2, -1))

    model.add(Linear(256, 128)) \
        .add(Dropout(0.2)) \
        .add(ReLU()) \
        .add(Linear(128, class_num)) \
        .add(LogSoftMax())

    return model


def train(sc, data_path,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split):
    print('Processing text dataset')
    texts = news20.get_news20(source_dir=data_path)
    data_rdd = sc.parallelize(texts, 2)

    word_to_ic = analyze_texts(data_rdd)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic[10: max_words])
    bword_to_ic = sc.broadcast(word_to_ic)

    w2v = news20.get_glove_w2v(source_dir=data_path, dim=embedding_dim)
    filtered_w2v = dict((w, v) for w, v in w2v.items() if w in word_to_ic)
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda text_label:
                              ([w for w in text_to_words(text_label[0]) if
                                w in bword_to_ic.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(
        lambda tokens_label: (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                       ([to_vec(w, bfiltered_w2v.value,
                                                embedding_dim) for w in
                                         tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))

    train_rdd, val_rdd = sample_rdd.randomSplit(
        [training_split, 1 - training_split])

    optimizer = Optimizer.create(
        model=build_model(news20.CLASS_NUM),
        training_set=train_rdd,
        criterion=ClassNLLCriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method=Adagrad(learningrate=learning_rate, learningrate_decay=0.001))

    optimizer.set_validation(
        batch_size=batch_size,
        val_rdd=val_rdd,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )

    logdir = '/tmp/.bigdl/'
    app_name = 'adam-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_summary = TrainSummary(log_dir=logdir, app_name=app_name)
    train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
    val_summary = ValidationSummary(log_dir=logdir, app_name=app_name)
    optimizer.set_train_summary(train_summary)
    optimizer.set_val_summary(val_summary)

    train_model = optimizer.optimize()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-l", "--learning_rate", dest="learning_rate", default="0.05")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="300")  # noqa
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="30")
    parser.add_option("--model", dest="model_type", default="cnn")
    parser.add_option("-p", "--p", dest="p", default="0.0")
    parser.add_option("-d", "--data_path", dest="data_path", default="/tmp/news20/")
    parser.add_option("--optimizerVersion", dest="optimizerVersion", default="optimizerV1")
    parser.add_option("--on-yarn", action="store_true", dest="onYarn", default=False)

    (options, args) = parser.parse_args(sys.argv)
    if options.action == "train":
        batch_size = int(options.batchSize)
        embedding_dim = int(options.embedding_dim)
        learning_rate = float(options.learning_rate)
        max_epoch = int(options.max_epoch)
        p = float(options.p)
        model_type = options.model_type
        sequence_len = 500
        max_words = 5000
        training_split = 0.8
        data_path = options.data_path
        if options.onYarn:
            hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
            invalidInputError(hadoop_conf,
                              "Directory path to hadoop conf not found for yarn-client"
                              " mode.", "Please either specify argument hadoop_conf or"
                                        "set the environment variable HADOOP_CONF_DIR")
            conda_env_name = detect_conda_env_name()
            sc = init_spark_on_yarn(hadoop_conf=hadoop_conf,
                                    conda_name=conda_env_name,
                                    num_executors=2,
                                    executor_cores=2,
                                    executor_memory="20g",
                                    driver_memory="10g")
        else:
            conf = {"spark.driver.memory": "40g"}
            sc = init_spark_on_local(cores=4, conf=conf)

        set_optimizer_version(options.optimizerVersion)
        train(sc, data_path,
              batch_size,
              sequence_len, max_words, embedding_dim, training_split)
        sc.stop()
    elif options.action == "test":
        pass
