from nn.layer import *
from optparse import OptionParser
from nn.criterion import *
from optim.optimizer import *
from util.common import *
from dataset import imagenet
from dataset.transformer import *


def alexnet(class_num):
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
    model.add(View([256 * 6 * 6]))
    model.add(Linear(256 * 6 * 6, 4096).set_name("fc6"))
    model.add(ReLU(True).set_name("relu6"))
    model.add(Dropout(0.5).set_name("drop6"))
    model.add(Linear(4096, 4096).set_name("fc7"))
    model.add(ReLU(True).set_name("relu7"))
    model.add(Dropout(0.5).set_name("drop7"))
    model.add(Linear(4096, class_num).set_name("fc8"))
    model.add(LogSoftMax().set_name("loss"))
    return model