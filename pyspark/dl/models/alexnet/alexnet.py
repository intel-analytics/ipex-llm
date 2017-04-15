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

import sys
from optparse import OptionParser

from dataset import imagenet
from dataset.transformer import *
from nn.layer import *
from nn.criterion import *
from optim.optimizer import *
from util.common import *


def build_model(class_num):
    model = Sequential()
    model.add(SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, 1, False).set_name("conv1"))
    model.add(ReLU(True).set_name("relu1"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("norm1"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).set_name("pool1"))
    model.add(SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2).set_name("conv2"))
    model.add(ReLU(True).set_name("relu2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).set_name("pool2"))
    model.add(SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1).set_name("conv3"))
    model.add(ReLU(True).set_name("relu3"))
    model.add(SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2).set_name("conv4"))
    model.add(ReLU(True).set_name("relu4"))
    model.add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2).set_name("conv5"))
    model.add(ReLU(True).set_name("relu5"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).set_name("pool5"))
    model.add(View([256, 6, 6]))
    model.add(Linear(256 * 6 * 6, 4096).set_name("fc6"))
    model.add(ReLU(True).set_name("relu6"))
    model.add(Dropout(0.5).set_name("drop6"))
    model.add(Linear(4096, 4096).set_name("fc7"))
    model.add(ReLU(True).set_name("relu7"))
    model.add(Dropout(0.5).set_name("drop7"))
    model.add(Linear(4096, class_num).set_name("fc8"))
    model.add(LogSoftMax().set_name("loss"))
    return model


def get_alex_data(folder, file_type="image", data_type="train"):
    path = os.path.join(folder, data_type)
    if "seq" == file_type:
        return imagenet.read_seq_file(sc, path)
    elif "image" == file_type:
        return imagenet.read_local(sc, path, 255.0)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-c", "--coreNum", dest="coreNum", default="4")
    parser.add_option("-n", "--nodeNum", dest="nodeNum", default="1")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("-f", "--folder", dest="folder", default="./")
    parser.add_option("-t", "--fileType", dest="type", default="image")
    parser.add_option("-m", "--meanFile", dest="meanFile", default="imagenet_mean.binaryproto")

    (options, args) = parser.parse_args(sys.argv)
    sc = SparkContext(appName="alexnet", conf=create_spark_conf())
    init_engine()
    image_size = 227
    means = imagenet.load_mean_file(options.meanFile)
    if options.action == "train":
        # train_data = get_alex_data(options.folder, "image", "train").map(
        #     pixel_normalizer(means)).map(crop(image_size, image_size, "random"))
        train_data = get_alex_data(options.folder, "seq", "train").map(
            pixel_normalizer(means)).map(
            crop(image_size, image_size, "random")).map(
            transpose(False))
        test_data = get_alex_data(options.folder, "seq", "val").map(
            pixel_normalizer(means)).map(
            crop(image_size, image_size, "center")).map(
            transpose(False))
        state = {
            "learningRate": 0.01,
            "weightDecay": 0.0005,
            "momentum": 0.9,
            "learingRateSchedule": Step(100000, 0.1)}
        optimizer = Optimizer(
            model=build_model(10),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method="SGD",
            state=state,
            end_trigger=MaxEpoch(2),
            batch_size=int(options.batchSize))
        optimizer.setvalidation(
            batch_size=4,
            val_rdd=test_data,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy"]
        )
        optimizer.setcheckpoint(EveryEpoch(), "/tmp/alexnet/")
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
    elif options.action == "test":
        model = Model.loadCaffe(build_model(1000),
                                "/home/jwang/model/alexnet/train_val.prototxt",
                                "/home/jwang/model/alexnet/bvlc_alexnet.caffemodel", False)
        test_data = get_alex_data(options.folder, "image", "val").map(
            bgr_pixel_normalizer(means)).map(crop(image_size, image_size, "center"))
        # # TODO: Pass model path through external parameter
        # model = Model.from_path("/tmp/lenet5/model.431")
        # validator = Validator(model, test_data, batch_size=32)
        # result = validator.test(["top1", "top5"])
        result = model.test(test_data, 4, ["Top1Accuracy"])
        print result
