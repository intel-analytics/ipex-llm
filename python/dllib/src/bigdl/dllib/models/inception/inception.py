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


from bigdl.dllib.nn.layer import *
from optparse import OptionParser
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.nn.initialization_method import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.feature.transform.vision.image import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.utils.utils import detect_conda_env_name
import os
from math import ceil
from bigdl.dllib.utils.log4Error import *


def t(input_t):
    if type(input_t) is list:
        # insert into index 0 spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_t)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_t


def inception_layer_v1(input_size, config, name_prefix=""):
    concat = Concat(2)
    conv1 = Sequential()
    conv1.add(SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "1x1"))
    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
    concat.add(conv1)
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "3x3"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
    concat.add(conv3)
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "5x5_reduce"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "5x5"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
    concat.add(conv5)
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True).set_name(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1)
             .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
             .set_name(name_prefix + "pool_proj"))
    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
    concat.add(pool).set_name(name_prefix + "output")
    return concat


def inception_v1_no_aux_classifier(class_num, has_dropout=True):
    model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name("conv1/7x7_s2"))
    model.add(ReLU(True).set_name("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name("conv2/3x3_reduce"))
    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name("conv2/3x3"))
    model.add(ReLU(True).set_name("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    model.add(inception_layer_v1(192, t([t([64]), t(
        [96, 128]), t([16, 32]), t([32])]), "inception_3a/"))
    model.add(inception_layer_v1(256, t([t([128]), t(
        [128, 192]), t([32, 96]), t([64])]), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(inception_layer_v1(480, t([t([192]), t(
        [96, 208]), t([16, 48]), t([64])]), "inception_4a/"))
    model.add(inception_layer_v1(512, t([t([160]), t(
        [112, 224]), t([24, 64]), t([64])]), "inception_4b/"))
    model.add(inception_layer_v1(512, t([t([128]), t(
        [128, 256]), t([24, 64]), t([64])]), "inception_4c/"))
    model.add(inception_layer_v1(512, t([t([112]), t(
        [144, 288]), t([32, 64]), t([64])]), "inception_4d/"))
    model.add(inception_layer_v1(528, t([t([256]), t(
        [160, 320]), t([32, 128]), t([128])]), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(inception_layer_v1(832, t([t([256]), t(
        [160, 320]), t([32, 128]), t([128])]), "inception_5a/"))
    model.add(inception_layer_v1(832, t([t([384]), t(
        [192, 384]), t([48, 128]), t([128])]), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    if has_dropout:
        model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    model.add(View([1024], num_input_dims=3))
    model.add(Linear(1024, class_num)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
              .set_name("loss3/classifier"))
    model.add(LogSoftMax().set_name("loss3/loss3"))
    model.reset()
    return model


def inception_v1(class_num, has_dropout=True):
    feature1 = Sequential()
    feature1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False)
                 .set_init_method(weight_init_method=Xavier(),
                                  bias_init_method=ConstInitMethod(0.1))
                 .set_name("conv1/7x7_s2"))
    feature1.add(ReLU(True).set_name("conv1/relu_7x7"))
    feature1.add(
        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True)
            .set_name("pool1/3x3_s2"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75)
                 .set_name("pool1/norm1"))
    feature1.add(SpatialConvolution(64, 64, 1, 1, 1, 1)
                 .set_init_method(weight_init_method=Xavier(),
                                  bias_init_method=ConstInitMethod(0.1))
                 .set_name("conv2/3x3_reduce"))
    feature1.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    feature1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
                 .set_init_method(weight_init_method=Xavier(),
                                  bias_init_method=ConstInitMethod(0.1))
                 .set_name("conv2/3x3"))
    feature1.add(ReLU(True).set_name("conv2/relu_3x3"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    feature1.add(
        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    feature1.add(inception_layer_v1(192,
                                    t([t([64]), t([96, 128]), t([16, 32]), t([32])]),
                                    "inception_3a/"))

    feature1.add(inception_layer_v1(256,
                                    t([t([128]), t([128, 192]), t([32, 96]), t([64])]),
                                    "inception_3b/"))

    feature1.add(
        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool3/3x3_s2"))

    feature1.add(inception_layer_v1(480,
                                    t([t([192]), t([96, 208]), t([16, 48]), t([64])]),
                                    "inception_4a/"))

    output1 = Sequential()
    output1.add(SpatialAveragePooling(
        5, 5, 3, 3, ceil_mode=True).set_name("loss1/ave_pool"))
    output1.add(
        SpatialConvolution(512, 128, 1, 1, 1, 1).set_name("loss1/conv"))
    output1.add(ReLU(True).set_name("loss1/relu_conv"))
    output1.add(View([128 * 4 * 4, 3]))
    output1.add(Linear(128 * 4 * 4, 1024).set_name("loss1/fc"))
    output1.add(ReLU(True).set_name("loss1/relu_fc"))
    if has_dropout:
        output1.add(Dropout(0.7).set_name("loss1/drop_fc"))
    output1.add(Linear(1024, class_num).set_name("loss1/classifier"))
    output1.add(LogSoftMax().set_name("loss1/loss"))

    feature2 = Sequential()
    feature2.add(inception_layer_v1(512,
                                    t([t([160]), t([112, 224]), t([24, 64]), t([64])]),
                                    "inception_4b/"))
    feature2.add(inception_layer_v1(512,
                                    t([t([128]), t([128, 256]), t([24, 64]), t([64])]),
                                    "inception_4c/"))
    feature2.add(inception_layer_v1(512,
                                    t([t([112]), t([144, 288]), t([32, 64]), t([64])]),
                                    "inception_4d/"))

    output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).set_name("loss2/ave_pool"))
    output2.add(
        SpatialConvolution(528, 128, 1, 1, 1, 1).set_name("loss2/conv"))
    output2.add(ReLU(True).set_name("loss2/relu_conv"))
    output2.add(View([128 * 4 * 4, 3]))
    output2.add(Linear(128 * 4 * 4, 1024).set_name("loss2/fc"))
    output2.add(ReLU(True).set_name("loss2/relu_fc"))
    if has_dropout:
        output2.add(Dropout(0.7).set_name("loss2/drop_fc"))
    output2.add(Linear(1024, class_num).set_name("loss2/classifier"))
    output2.add(LogSoftMax().set_name("loss2/loss"))

    output3 = Sequential()
    output3.add(inception_layer_v1(528,
                                   t([t([256]), t([160, 320]), t([32, 128]), t([128])]),
                                   "inception_4e/"))
    output3.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool4/3x3_s2"))
    output3.add(inception_layer_v1(832,
                                   t([t([256]), t([160, 320]), t([32, 128]), t([128])]),
                                   "inception_5a/"))
    output3.add(inception_layer_v1(832,
                                   t([t([384]), t([192, 384]), t([48, 128]), t([128])]),
                                   "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    if has_dropout:
        output3.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    output3.add(View([1024, 3]))
    output3.add(Linear(1024, class_num)
                .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
                .set_name("loss3/classifier"))
    output3.add(LogSoftMax().set_name("loss3/loss3"))

    split2 = Concat(2).set_name("split2")
    split2.add(output3)
    split2.add(output2)

    mainBranch = Sequential()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    split1 = Concat(2).set_name("split1")
    split1.add(mainBranch)
    split1.add(output1)

    model = Sequential()

    model.add(feature1)
    model.add(split1)

    model.reset()
    return model


def get_inception_data(url, sc=None, data_type="train"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)


def config_option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="folder", default="",
                      help="url of hdfs folder store the hadoop sequence files")
    parser.add_option("--model", type=str, dest="model", default="", help="model snapshot location")
    parser.add_option("--state", type=str, dest="state", default="", help="state snapshot location")
    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="",
                      help="where to cache the model")
    parser.add_option("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                      help="overwrite checkpoint files")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=0,
                      help="epoch numbers")
    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=62000,
                      help="iteration numbers")
    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01,
                      help="learning rate")
    parser.add_option("--warmupEpoch", type=int, dest="warmupEpoch", default=0,
                      help="warm up epoch numbers")
    parser.add_option("--maxLr", type=float, dest="maxLr", default=0.0, help="max Lr after warm up")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
    parser.add_option("--classNum", type=int, dest="classNum", default=1000, help="class number")
    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001,
                      help="weight decay")
    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
                      help="checkpoint interval of iterations")
    parser.add_option("--gradientMin", type=float, dest="gradientMin", default=0.0,
                      help="min gradient clipping by")
    parser.add_option("--gradientMax", type=float, dest="gradientMax", default=0.0,
                      help="max gradient clipping by")
    parser.add_option("--gradientL2NormThreshold", type=float, dest="gradientL2NormThreshold",
                      default=0.0, help="gradient L2-Norm threshold")
    parser.add_option("--executor-cores", type=int, dest="cores", default=4,
                      help="number of executor cores")
    parser.add_option("--num-executors", type=int, dest="executors", default=16,
                      help="number of executors")
    parser.add_option("--executor-memory", type=str, dest="executorMemory", default="30g",
                      help="executor memory")
    parser.add_option("--driver-memory", type=str, dest="driverMemory", default="30g",
                      help="driver memory")
    parser.add_option("--deploy-mode", type=str, dest="deployMode", default="yarn-client",
                      help="yarn deploy mode, yarn-client or yarn-cluster")

    return parser


if __name__ == "__main__":
    # parse options
    parser = config_option_parser()
    (options, args) = parser.parse_args(sys.argv)
    if not options.learningRate:
        parser.error("-l --learningRate is a mandatory opt")
    if not options.batchSize:
        parser.error("-b --batchSize is a mandatory opt")

    # init
    hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
    invalidInputError(hadoop_conf,
                      "Directory path to hadoop conf not found for yarn-client"
                      " mode.", "Please either specify argument hadoop_conf or"
                                "set the environment variable HADOOP_CONF_DIR")

    conf = create_spark_conf().set("spark.executor.memory", options.executorMemory) \
        .set("spark.executor.cores", options.cores) \
        .set("spark.executor.instances", options.executors) \
        .set("spark.driver.memory", options.driverMemory)

    sc = init_nncontext(conf, cluster_mode=options.deployMode, hadoop_conf=hadoop_conf)

    image_size = 224  # create dataset
    train_transformer = Pipeline([PixelBytesToMat(),
                                  Resize(256, 256),
                                  RandomCropper(image_size, image_size, True, "Random", 3),
                                  ChannelNormalize(123.0, 117.0, 104.0),
                                  MatToTensor(to_rgb=False),
                                  ImageFrameToSample(input_keys=["imageTensor"],
                                                     target_keys=["label"])
                                  ])
    raw_train_data = get_inception_data(options.folder, sc, "train")
    train_data = DataSet.image_frame(raw_train_data).transform(train_transformer)

    val_transformer = Pipeline([PixelBytesToMat(),
                                Resize(256, 256),
                                RandomCropper(image_size, image_size, False, "Center", 3),
                                ChannelNormalize(123.0, 117.0, 104.0),
                                MatToTensor(to_rgb=False),
                                ImageFrameToSample(input_keys=["imageTensor"],
                                                   target_keys=["label"])
                                ])
    raw_val_data = get_inception_data(options.folder, sc, "val")
    val_data = DataSet.image_frame(raw_val_data).transform(val_transformer)

    # build model
    if options.model != "":
        # load model snapshot
        inception_model = Model.loadModel(options.model)
    else:
        inception_model = inception_v1_no_aux_classifier(options.classNum)

    # set optimization method
    iterationPerEpoch = int(ceil(float(1281167) / options.batchSize))
    if options.maxEpoch:
        maxIteration = iterationPerEpoch * options.maxEpoch
    else:
        maxIteration = options.maxIteration
    warmup_iteration = options.warmupEpoch * iterationPerEpoch
    if options.state != "":
        # load state snapshot
        optim = OptimMethod.load(options.state)
    else:
        if warmup_iteration == 0:
            warmupDelta = 0.0
        else:
            if options.maxLr:
                maxlr = options.maxLr
            else:
                maxlr = options.learningRate
            warmupDelta = (maxlr - options.learningRate) / warmup_iteration
        polyIteration = maxIteration - warmup_iteration
        lrSchedule = SequentialSchedule(iterationPerEpoch)
        lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
        lrSchedule.add(Poly(0.5, maxIteration), polyIteration)
        optim = SGD(learningrate=options.learningRate, learningrate_decay=0.0,
                    weightdecay=options.weightDecay,
                    momentum=0.9, dampening=0.0, nesterov=False,
                    leaningrate_schedule=lrSchedule)

    # create triggers
    if options.maxEpoch:
        checkpoint_trigger = EveryEpoch()
        test_trigger = EveryEpoch()
        end_trigger = MaxEpoch(options.maxEpoch)
    else:
        checkpoint_trigger = SeveralIteration(options.checkpointIteration)
        test_trigger = SeveralIteration(options.checkpointIteration)
        end_trigger = MaxIteration(options.maxIteration)

    # Optimizer
    optimizer = Optimizer.create(
        model=inception_model,
        training_set=train_data,
        optim_method=optim,
        criterion=ClassNLLCriterion(),
        end_trigger=end_trigger,
        batch_size=options.batchSize
    )

    if options.checkpoint:
        optimizer.set_checkpoint(checkpoint_trigger, options.checkpoint, options.overwrite)

    if options.gradientMin and options.gradientMax:
        optimizer.set_gradclip_const(options.gradientMin, options.gradientMax)

    if options.gradientL2NormThreshold:
        optimizer.set_gradclip_l2norm(options.gradientL2NormThreshold)

    optimizer.set_validation(trigger=test_trigger,
                             val_rdd=val_data,
                             batch_size=options.batchSize,
                             val_method=[Top1Accuracy(), Top5Accuracy()])

    trained_model = optimizer.optimize()

    sc.stop()
