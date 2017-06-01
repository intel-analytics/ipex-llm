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


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--isHdfs", dest="hdfs", default=False)
    parser.add_option("-f", "--folder", type=str, dest="folder", default="./")
    parser.add_option("--modelPath", type=str, dest="model_path", default="")
    parser.add_option("--modelType", type=str, dest="model_type", default="")
    parser.add_option("--showNum", type=int, dest="show_num", default=100)


    (options, args) = parser.parse_args(sys.argv)
    sc = SparkContext(appName="image_classification", conf=create_spark_conf())
    version = str(sc.version)
    init_engine()
    image_size = 224
    ishdfs = strtobool(options.hdfs)
    # read dataset
    if ishdfs:
        data_rdd = imagenet.read_seq_file(sc, options.folder, 255.0, has_name=True)
    else:
        data_rdd = imagenet.read_local_with_name(sc, options.folder, 255.0, False)
    # transform to vectors
    val_transformer = ImgTransformer([Crop(image_size, image_size, "random"),
                                   ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                   TransposeToTensor()
                                   ])
    if version.startswith("1.5") or version.startswith("1.6"):
        from pyspark.mllib.linalg import Vectors
    else:
        from pyspark.ml.linalg import Vectors

    val_rdd = data_rdd.map(
        lambda features_label_name: (val_transformer(features_label_name[0]), features_label_name[2])).map(
        lambda features_name: (Vectors.dense(features_name[0].ravel().tolist()), features_name[1]))
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
    classifier = DLClassifier(model, (3, image_size, image_size)).setInputCol("features").setOutputCol("predict")
    # predict
    classifier.transform(val_df)\
        .select("image_name", "predict") \
        .show(options.show_num, False)
    sc.stop()