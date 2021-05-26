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
from optparse import OptionParser

import tensorflow as tf

from zoo import init_nncontext
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.feature.image.imageset import *
from zoo.tfpark import TFDataset, TFEstimator
from zoo.tfpark import ZooOptimizer


def main(option):
    batch_size = 16 if not option.batch_size else int(option.batch_size)
    sc = init_nncontext()

    def input_fn(mode, params):

        if mode == tf.estimator.ModeKeys.TRAIN:
            image_set = ImageSet.read(params["image_path"],
                                      sc=sc, with_label=True, one_based_label=False)
            train_transformer = ChainedPreprocessing([ImageBytesToMat(),
                                                      ImageResize(256, 256),
                                                      ImageRandomCrop(224, 224),
                                                      ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                                      ImageChannelNormalize(
                                                          0.485, 0.456, 0.406,
                                                          0.229, 0.224, 0.225),
                                                      ImageMatToTensor(to_RGB=True, format="NHWC"),
                                                      ImageSetToSample(input_keys=["imageTensor"],
                                                                       target_keys=["label"])
                                                      ])
            feature_set = FeatureSet.image_frame(image_set.to_image_frame())
            feature_set = feature_set.transform(train_transformer)
            feature_set = feature_set.transform(ImageFeatureToSample())
            dataset = TFDataset.from_feature_set(feature_set,
                                                 features=(tf.float32, [224, 224, 3]),
                                                 labels=(tf.int32, [1]), batch_size=batch_size)
        else:
            raise NotImplementedError

        return dataset

    def model_fn(features, labels, mode, params):
        from nets import inception
        slim = tf.contrib.slim
        labels = tf.squeeze(labels, axis=1)
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, end_points = inception.inception_v1(features,
                                                        num_classes=int(params["num_classes"]),
                                                        is_training=True)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
            train_op = ZooOptimizer(tf.train.AdamOptimizer()).minimize(loss)
            return tf.estimator.EstimatorSpec(mode, train_op=train_op,
                                              predictions=logits, loss=loss)
        else:
            raise NotImplementedError

    estimator = TFEstimator.from_model_fn(model_fn,
                                          params={"image_path": option.image_path,
                                                  "num_classes": option.num_classes,
                                                  "batch_size": option.batch_size})

    estimator.train(input_fn, steps=100)
    print("finished...")
    sc.stop()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--image-path", dest="image_path")
    parser.add_option("--num-classes", dest="num_classes")
    parser.add_option("--batch_size", dest="batch_size")

    (options, args) = parser.parse_args(sys.argv)
    main(options)
