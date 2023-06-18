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

from optparse import OptionParser

from dataset.transformer import *
from dataset import imagenet
from models.inception import inception
from models.loadmodel.alexnet import alexnet
from nn.layer import *
from util.common import *

def get_data(folder, file_type="local", normalize=255.0):
    if "hdfs" == file_type:
        return imagenet.read_seq_file(sc, folder, normalize)
    elif "local" == file_type:
        return imagenet.read_local(sc, folder, 255.0)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--folder", dest="folder", default="./")
    parser.add_option("-m", "--modelName", dest="modelName", default="")
    parser.add_option("-t", "--modelType", dest="modelType", default="")
    parser.add_option("--caffeDefPath", dest="caffeDefPath", default="")
    parser.add_option("--modelPath", dest="modelPath", default="")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="32")
    parser.add_option("--meanFile", dest="meanFile", default="imagenet_mean.binaryproto")

    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="image_classifier",
                      conf=create_spark_conf())
    init_engine()

    if options.modelType == "caffe":
        if options.modelName == "alexnet":
            image_size = 227
            model = Model.load_caffe(alexnet.alexnet(1000), options.caffeDefPath, options.modelPath)
            means = imagenet.load_mean_file(options.meanFile)
            validate_dataset = get_data(options.folder, "hdfs").map(
                pixel_normalizer(means)).map(
                crop(image_size, image_size, "center")).map(
                transpose(False))

        elif options.modelName == "inception":
            image_size = 224
            model = Model.load_caffe(inception.Inception_v1_NoAuxClassifier(1000), options.caffeDefPath, options.modelPath)
            validate_dataset = get_data(options.folder, "hdfs", 1.0).map(
                crop(image_size, image_size, "center")).map(
                channel_normalizer(123, 117, 104, 1, 1, 1)).map(
                transpose(False))

    elif options.modelType == "torch":
        if options.modelName == "resnet":
            image_size = 224
            model = Model.load_torch(options.modelPath)
            validate_dataset = get_data(options.folder, "hdfs").map(
                crop(image_size, image_size, "center")).map(
                channel_normalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)).map(
                transpose(True))

    results = model.test(validate_dataset, int(options.batchSize), ["Top1Accuracy"])
    for result in results:
        print result
    sc.stop()
