from nn.layer import *
from optparse import OptionParser
from nn.criterion import *
from optim.optimizer import *
from util.common import *
from dataset import imagenet
from dataset.transformer import *


def scala_T(input_T):
    if type(input_T) is list:
        # insert into index 0 spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_T)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_T


def Inception_Layer_v1(input_size, config, name_prefix=""):
    concat = Concat(2)
    conv1 = Sequential()
    conv1.add(
        SpatialConvolution(input_size,
                           config[1][1],
                           1, 1, 1, 1, init_method="Xavier")
        .set_name(name_prefix + "1x1"))
    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
    concat.add(conv1)
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1,
                                 init_method="Xavier")
              .set_name(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1], config[2][2],
                                 3, 3, 1, 1, 1, 1, init_method="Xavier")
              .set_name(name_prefix + "3x3"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
    concat.add(conv3)
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size,
                                 config[3][1], 1, 1, 1, 1,
                                 init_method="Xavier")
              .set_name(name_prefix + "5x5_reduce"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1],
                                 config[3][2], 5, 5, 1, 1, 2, 2,
                                 init_method="Xavier")
              .set_name(name_prefix + "5x5"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
    concat.add(conv5)
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1,
                               to_ceil=True).set_name(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1,
                                init_method="Xavier")
             .set_name(name_prefix + "pool_proj"))
    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
    concat.add(pool).set_name(name_prefix + "output")
    return concat


def Inception_v1_NoAuxClassifier(class_num):
    model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1,
                                 False, init_method="Xavier").set_name("conv1/7x7_s2"))
    model.add(ReLU(True).set_name("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1, init_method="Xavier")
          .set_name("conv2/3x3_reduce"))
    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1, init_method="Xavier")
          .set_name("conv2/3x3"))
    model.add(ReLU(True).set_name("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75). set_name("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, scala_T([scala_T([64]), scala_T(
        [96, 128]), scala_T([16, 32]), scala_T([32])]), "inception_3a/"))
    model.add(Inception_Layer_v1(256, scala_T([scala_T([128]), scala_T(
        [128, 192]), scala_T([32, 96]), scala_T([64])]), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(480, scala_T([scala_T([192]), scala_T(
        [96, 208]), scala_T([16, 48]), scala_T([64])]), "inception_4a/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([160]), scala_T(
        [112, 224]), scala_T([24, 64]), scala_T([64])]), "inception_4b/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([128]), scala_T(
        [128, 256]), scala_T([24, 64]), scala_T([64])]), "inception_4c/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([112]), scala_T(
        [144, 288]), scala_T([32, 64]), scala_T([64])]), "inception_4d/"))
    model.add(Inception_Layer_v1(528, scala_T([scala_T([256]), scala_T(
        [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([256]), scala_T(
        [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_5a/"))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([384]), scala_T(
        [192, 384]), scala_T([48, 128]), scala_T([128])]), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    model.add(View([1024], num_input_dims=3))
    model.add(Linear(1024, class_num, init_method="Xavier").set_name("loss3/classifier"))
    model.add(LogSoftMax().set_name("loss3/loss3"))
    model.reset()
    return model


def Inception_v1(class_num):
    feature1 = Sequential()
    feature1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False,
                                    init_method="Xavier")
                 .set_name("conv1/7x7_s2"))
    feature1.add(ReLU(True).set_name("conv1/relu_7x7"))
    feature1.add(
        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True)
        .set_name("pool1/3x3_s2"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75)
                 .set_name("pool1/norm1"))
    feature1.add(SpatialConvolution(64, 64, 1, 1, 1, 1, init_method="Xavier")
                 .set_name("conv2/3x3_reduce"))
    feature1.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    feature1.add(SpatialConvolution(64, 192, 3, 3, 1,
                                    1, 1, 1,
                                    init_method="Xavier")
                 .set_name("conv2/3x3"))
    feature1.add(ReLU(True).set_name("conv2/relu_3x3"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75). set_name("conv2/norm2"))
    feature1.add(
        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    feature1.add(Inception_Layer_v1(192,
                                    scala_T([scala_T([64]), scala_T([96, 128]),
                                            scala_T([16, 32]), scala_T([32])]),
                                    "inception_3a/"))
    feature1.add(Inception_Layer_v1(256, scala_T([
        scala_T([128]), scala_T([128, 192]), scala_T([32, 96]), scala_T([64])]),
        "inception_3b/"))
    feature1.add(
        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool3/3x3_s2"))
    feature1.add(Inception_Layer_v1(480, scala_T([
        scala_T([192]), scala_T([96, 208]), scala_T([16, 48]), scala_T([64])]),
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
    output1.add(Dropout(0.7).set_name("loss1/drop_fc"))
    output1.add(Linear(1024, class_num).set_name("loss1/classifier"))
    output1.add(LogSoftMax().set_name("loss1/loss"))

    feature2 = Sequential()
    feature2.add(Inception_Layer_v1(512,
                                    scala_T([scala_T([160]), scala_T([112, 224]),
                                            scala_T([24, 64]), scala_T([64])]),
                                    "inception_4b/"))
    feature2.add(Inception_Layer_v1(512,
                                    scala_T([scala_T([128]), scala_T([128, 256]),
                                            scala_T([24, 64]), scala_T([64])]),
                                    "inception_4c/"))
    feature2.add(Inception_Layer_v1(512,
                                    scala_T([scala_T([112]), scala_T([144, 288]),
                                            scala_T([32, 64]), scala_T([64])]),
                                    "inception_4d/"))

    output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).set_name("loss2/ave_pool"))
    output2.add(
        SpatialConvolution(528, 128, 1, 1, 1, 1).set_name("loss2/conv"))
    output2.add(ReLU(True).set_name("loss2/relu_conv"))
    output2.add(View([128 * 4 * 4, 3]))
    output2.add(Linear(128 * 4 * 4, 1024).set_name("loss2/fc"))
    output2.add(ReLU(True).set_name("loss2/relu_fc"))
    output2.add(Dropout(0.7).set_name("loss2/drop_fc"))
    output2.add(Linear(1024, class_num).set_name("loss2/classifier"))
    output2.add(LogSoftMax().set_name("loss2/loss"))

    output3 = Sequential()
    output3.add(Inception_Layer_v1(528,
                                   scala_T([scala_T([256]), scala_T([160, 320]),
                                           scala_T([32, 128]), scala_T([128])]),
                                   "inception_4e/"))
    output3.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool4/3x3_s2"))
    output3.add(Inception_Layer_v1(832,
                                   scala_T([scala_T([256]), scala_T([160, 320]),
                                           scala_T([32, 128]), scala_T([128])]),
                                   "inception_5a/"))
    output3.add(Inception_Layer_v1(832,
                                   scala_T([scala_T([384]), scala_T([192, 384]),
                                           scala_T([48, 128]), scala_T([128])]),
                                   "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    output3.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    output3.add(View([1024, 3]))
    output3.add(Linear(1024, class_num, init_method="Xavier")
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


def get_inception_data(folder, file_type="image", data_type="train", normalize=255.0):
    path = os.path.join(folder, data_type)
    if "seq" == file_type:
        return imagenet.read_seq_file(sc, path, normalize)
    elif "image" == file_type:
        return imagenet.read_local(sc, path, normalize)


def config_option_parser():
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-f","--folder", type=str, dest="folder", default="",
                      help="url of hdfs folder store the hadoop sequence files")
    parser.add_option("--model", type=str, dest="model", default="", help="model snapshot location")
    parser.add_option("--state", type=str, dest="state", default="", help="state snapshot location")
    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="", help="where to cache the model")
    parser.add_option("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                      help="overwrite checkpoint files")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=0, help="epoch numbers")
    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=62000, help="iteration numbers")
    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01, help="iteration numbers")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
    parser.add_option("--classNum", type=int, dest="classNum", default=1000, help="class number")
    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001, help="weight decay")
    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
                      help="checkpoint interval of iterations")

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
    sc = SparkContext(appName="inception v1", conf=create_spark_conf())
    init_engine()

    # build model
    inception_model = Inception_v1_NoAuxClassifier(options.classNum)

    image_size = 224

    if options.action == "train":
        # create dataset
        train_data = get_inception_data(options.folder, "seq", "train").map(
            crop(image_size, image_size, "random")).map(
            flip(0.5)).map(
            channel_normalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)).map(
            transpose(False))
        val_data = get_inception_data(options.folder, "seq", "val").map(
            crop(image_size, image_size, "center")).map(
            flip(0.5)).map(
            channel_normalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)).map(
            transpose(False))

        # TODO: Check stateSnapshot opt

        if options.maxEpoch:
            state = scala_T(
                {"learningRate": options.learningRate,
                 "weightDecay": options.weightDecay,
                 "momentum": 0.9,
                 "dampening": 0.0})
            checkpoint_trigger = EveryEpoch()
            test_trigger = EveryEpoch()
            end_trigger = MaxEpoch(options.maxEpoch)
        else:
            state = scala_T(
                {"learningRate": options.learningRate,
                 "weightDecay": options.weightDecay,
                 "momentum": 0.9,
                 "dampening": 0.0})
            checkpoint_trigger = SeveralIteration(options.checkpointIteration)
            test_trigger = SeveralIteration(options.checkpointIteration)
            end_trigger = MaxIteration(options.maxIteration)

        # Optimizer
        optimizer = Optimizer(
                model=inception_model,
                training_rdd=train_data,
                optim_method="SGD",
                criterion=ClassNLLCriterion(),
                end_trigger=end_trigger,
                batch_size=options.batchSize,
                state=state
            )
        
        if options.checkpoint:
            optimizer.setcheckpoint(checkpoint_trigger, options.checkpoint, options.overwrite)

        optimizer.setvalidation(trigger=test_trigger,
                                val_rdd=val_data,
                                batch_size=options.batchSize,
                                val_method=["Top1Accuracy", "Top5Accuracy"])

        trained_model = optimizer.optimize()

    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_inception_data(options.folder, "seq", "val").map(
            crop(image_size, image_size, "center")).map(
            flip(0.5)).map(
            channel_normalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)).map(
            transpose(False))
        model = Model.load(options.model)
        results = model.test(test_data, options.batchSize, ["Top1Accuracy", "Top5Accuracy"])
        for result in results:
            print result

