#
# Copyright 2018 Analytics Zoo Authors.
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

from math import ceil
from optparse import OptionParser
from bigdl.optim.optimizer import *
from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.keras.metrics import *
from zoo.pipeline.nnframes import *
from zoo.tfpark import TFDataset, TFOptimizer
from nets import inception_v1
import tensorflow as tf

slim = tf.contrib.slim


def get_inception_data(url, sc=None, data_type="train"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)


def config_option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="folder", default="",
                      help="url of hdf+s folder store the hadoop sequence files")
    parser.add_option("--model", type=str, dest="model", default="",
                      help="model snapshot location")
    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="",
                      help="where to cache the model")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=0,
                      help="epoch numbers")
    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=3100,
                      help="iteration numbers")
    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01,
                      help="learning rate")
    parser.add_option("--warmupEpoch", type=int, dest="warmupEpoch", default=0,
                      help="warm up epoch numbers")
    parser.add_option("--maxLr", type=float, dest="maxLr", default=0.0,
                      help="max Lr after warm up")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001,
                      help="weight decay")
    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
                      help="checkpoint interval of iterations")
    parser.add_option("--resumeTrainingCheckpoint", type=str, dest="resumeTrainingCheckpoint",
                      default=None,
                      help="an analytics zoo checkpoint path for resume training, usually contains"
                           + "a file named model.$iter_num and a file named"
                           + " optimMethod-TFParkTraining.$iter_num")
    parser.add_option("--resumeTrainingVersion", type=int, dest="resumeTrainingVersion",
                      default=None,
                      help="the version of checkpoint file, should be the $iter_num"
                           + " in model.$iter_num")
    return parser

if __name__ == "__main__":
    # parse options
    parser = config_option_parser()
    (options, args) = parser.parse_args(sys.argv)

    if not options.learningRate:
        parser.error("-l --learningRate is a mandatory opt")
    if not options.batchSize:
        parser.error("-b --batchSize is a mandatory opt")

    sc = init_nncontext("inception v1")

    image_size = 224  # create dataset
    train_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                              ImageResize(256, 256),
                                              ImageRandomCrop(image_size, image_size),
                                              ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                              ImageChannelNormalize(123.0, 117.0, 104.0),
                                              ImageMatToTensor(format="NHWC", to_RGB=False),
                                              ImageSetToSample(input_keys=["imageTensor"],
                                                               target_keys=["label"])])
    raw_train_data = get_inception_data(options.folder, sc, "train")
    train_data = FeatureSet.image_frame(raw_train_data).transform(train_transformer)

    val_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                            ImageResize(256, 256),
                                            ImageCenterCrop(image_size, image_size),
                                            ImageChannelNormalize(123.0, 117.0, 104.0),
                                            ImageMatToTensor(format="NHWC", to_RGB=False),
                                            ImageSetToSample(input_keys=["imageTensor"],
                                                             target_keys=["label"])])
    raw_val_data = get_inception_data(options.folder, sc, "val")
    val_data = FeatureSet.image_frame(raw_val_data).transform(val_transformer)
    val_data = val_data.transform(ImageFeatureToSample())

    train_data = train_data.transform(ImageFeatureToSample())

    dataset = TFDataset.from_feature_set(train_data,
                                         features=(tf.float32, [224, 224, 3]),
                                         labels=(tf.int32, [1]),
                                         batch_size=options.batchSize,
                                         validation_dataset=val_data)

    images, labels = dataset.tensors

    # As sequence file's label is one-based, so labels need to subtract 1.
    zero_based_label = labels - 1

    is_training = tf.placeholder(dtype=tf.bool, shape=())

    with slim.arg_scope(inception_v1.inception_v1_arg_scope(weight_decay=0.0,
                                                            use_batch_norm=False)):
        logits, end_points = inception_v1.inception_v1(images,
                                                       dropout_keep_prob=0.6,
                                                       num_classes=1000,
                                                       is_training=is_training)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                                 labels=zero_based_label))

    iterationPerEpoch = int(ceil(float(1281167) / options.batchSize))
    if options.maxEpoch:
        maxIteration = iterationPerEpoch * options.maxEpoch
    else:
        maxIteration = options.maxIteration
    warmup_iteration = options.warmupEpoch * iterationPerEpoch

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
                weightdecay=options.weightDecay, momentum=0.9, dampening=0.0,
                nesterov=False,
                leaningrate_schedule=lrSchedule)

    if options.maxEpoch:
        checkpoint_trigger = EveryEpoch()
        end_trigger = MaxEpoch(options.maxEpoch)
    else:
        checkpoint_trigger = SeveralIteration(options.checkpointIteration)
        end_trigger = MaxIteration(options.maxIteration)

    optimizer = TFOptimizer.from_loss(loss, optim,
                                      val_outputs=[logits],
                                      val_labels=[zero_based_label],
                                      val_method=[Accuracy(), Top5Accuracy(), Loss()],
                                      tensor_with_value={is_training: [True, False]},
                                      model_dir="/tmp/logs")

    if options.resumeTrainingCheckpoint is not None:
        assert options.resumeTrainingVersion is not None,\
            "--resumeTrainingVersion must be specified when --resumeTrainingCheckpoint is."
        optimizer.load_checkpoint(options.resumeTrainingCheckpoint,
                                  options.resumeTrainingVersion)

    optimizer.optimize(end_trigger=end_trigger, checkpoint_trigger=checkpoint_trigger)

    if options.checkpoint:
        saver = tf.train.Saver()
        saver.save(optimizer.sess, options.checkpoint)

    sc.stop()
