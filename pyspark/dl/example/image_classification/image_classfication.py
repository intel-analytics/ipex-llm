#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Still in experimental stage!


from optparse import OptionParser
from distutils.util import strtobool

from dataset import imagenet
from dataset.transformer import *
from nn.layer import *
from util.common import *
from pyspark.sql import *
from ml.dl_classifier import *
from pyspark.mllib.linalg import Vectors

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--isHdfs", dest="hdfs", default=False)
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default=32)
    parser.add_option("-f", "--folder", type=str, dest="folder", default="./")
    parser.add_option("--modelPath", type=str, dest="model_path", default="")
    parser.add_option("--modelType", type=str, dest="model_type", default="")
    parser.add_option("--showNum", type=int, dest="show_num", default=100)


    (options, args) = parser.parse_args(sys.argv)
    sc = SparkContext(appName="image_classification", conf=create_spark_conf())
    init_engine()
    image_size = 224
    ishdfs = strtobool(options.hdfs)
    # read dataset
    if ishdfs:
        data_rdd = imagenet.read_seq_file(sc, options.folder, 255.0, has_name=True)
    else:
        data_rdd = imagenet.read_local_with_name(sc, options.folder, 255.0, False)
    # transform to vectors
    val_rdd = data_rdd.map(lambda data: (crop(image_size, image_size, "random")(data[0]), data[1])).map(
        lambda data: (channel_normalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)(data[0]), data[1])).map(
        lambda data: (transpose()(data[0]), data[1])).map(
        lambda data: (Vectors.dense(data[0].features.ravel().tolist()), data[1]))
    # create data frame
    sqlContext = SQLContext(sc)
    val_df = sqlContext.createDataFrame(val_rdd, ['features', 'image_name'])

    # get model
    if options.model_type == 'torch':
        model = Model.load_torch(options.model_path)
    elif options.model_type == 'bigdl':
        model = Model.load(options.model_path)
    else:
        print("Unsupported model type")
        sys.exit(0)
    # get DL classifier
    classifier = DLClassifier().set_input_col("features").set_output_col("predict")
    # set paramsMap
    paramMap = {"model_train": model,
                "batch_shape": (options.batchSize, 3, image_size, image_size)
                }
    # predict
    classifier.transform(val_df, paramMap)\
        .select("image_name", "predict") \
        .show(options.show_num, False)
    sc.stop()