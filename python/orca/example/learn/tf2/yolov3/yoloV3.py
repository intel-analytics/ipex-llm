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
# ===========================================================================
#
# This file is adapted from
# https://github.com/zzh8829/yolov3-tf2/blob/master/train.py,
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py and
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py
#
# MIT License
#
# Copyright (c) 2019 Zihao Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
    ModelCheckpoint, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Input, Lambda, \
    LeakyReLU, MaxPool2D, UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from zoo.orca.data.image.parquet_dataset import read_parquet, write_parquet
from zoo.orca.learn.tf2 import Estimator
from zoo.orca import init_orca_context, stop_orca_context
import numpy as np
import ray
import tempfile
import os
import argparse
import sys

DEFAULT_IMAGE_SIZE = 416


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    N = tf.shape(y_true)[0]
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    def outer_comp(i):
        def inner_comp(j):
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            def reduce(y_true, anchor_eq, grid_size):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = tf.stack([i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = tf.stack(
                    [box[0], box[1], box[2], box[3], tf.constant(1, dtype=tf.float32),
                     y_true[i][j][4]])
                print("updates", updates)
                # idx += 1
                return (True, indexes, updates)

            (mask, indexes, updates) = tf.cond(tf.reduce_any(anchor_eq) and
                                               not tf.equal(y_true[i][j][2], 0),
                                               lambda: reduce(y_true, anchor_eq, grid_size),
                                               lambda: (False, tf.zeros(4, tf.int32),
                                                        tf.zeros(6, tf.float32)))
            return (mask, indexes, updates)

        return tf.map_fn(inner_comp, tf.range(tf.shape(y_true)[1]),
                         dtype=(tf.bool, tf.int32, tf.float32))

    (mask, indexes, updates) = tf.map_fn(outer_comp, tf.range(N),
                                         dtype=(tf.bool, tf.int32, tf.float32))

    indexes = tf.boolean_mask(indexes, mask)
    updates = tf.boolean_mask(updates, mask)

    return tf.tensor_scatter_nd_update(y_true_out, indexes, updates)


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1],
                                                                            anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


def parse_data_train(image, label):
    x_train = tf.io.decode_jpeg(image, 3)
    x_train = tf.image.resize(x_train, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    paddings = [[0, 100 - tf.shape(label)[0]], [0, 0]]
    y_train = tf.pad(label, paddings)
    y_train = tf.convert_to_tensor(y_train, tf.float32)
    return x_train, y_train


IMAGE_FEATURE_MAP = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169), (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


# As tensorflow lite doesn't support tf.size used in tf.meshgrid,
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1], grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores, [1])
    bbox = tf.reshape(bbox, (-1, 4))
    classes = tf.argmax(dscores, 1)
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=100,
        iou_threshold=0.5,
        score_threshold=0.5,
        soft_nms_sigma=0.5
    )

    num_valid_nms_boxes = tf.shape(selected_indices)[0]

    selected_indices = tf.concat([selected_indices, tf.zeros(100 - num_valid_nms_boxes, tf.int32)],
                                 0)
    selected_scores = tf.concat([selected_scores, tf.zeros(100 - num_valid_nms_boxes, tf.float32)],
                                -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores = selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes, selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections = num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloLoss(anchors, classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def main():
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir",
                        help="Required. The path where data locates.")
    parser.add_argument("--output_data", dest="output_data", default=tempfile.mkdtemp(),
                        help="Required. The path where voc parquet data locates.")
    parser.add_argument("--data_year", dest="data_year", default="2009",
                        help="Required. The voc data date.")
    parser.add_argument("--split_name_train", dest="split_name_train", default="train",
                        help="Required. Split name.")
    parser.add_argument("--split_name_test", dest="split_name_test", default="val",
                        help="Required. Split name.")
    parser.add_argument("--names", dest="names",
                        help="Required. The path where class names locates.")
    parser.add_argument("--weights", dest="weights", default="./checkpoints/yolov3.weights",
                        help="Required. The path where weights locates.")
    parser.add_argument("--checkpoint", dest="checkpoint", default="./checkpoints/yolov3.tf",
                        help="Required. The path where checkpoint locates.")
    parser.add_argument("--checkpoint_folder", dest="checkpoint_folder", default="./checkpoints",
                        help="Required. The path where saved checkpoint locates.")
    parser.add_argument("--epochs", dest="epochs", type=int, default=2,
                        help="Required. epochs.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16,
                        help="Required. epochs.")
    parser.add_argument("--cluster_mode", dest="cluster_mode", default="local",
                        help="Required. Run on local/yarn/k8s mode.")
    parser.add_argument("--class_num", dest="class_num", type=int, default=20,
                        help="Required. class num.")
    parser.add_argument("--worker_num", type=int, default=1,
                        help="The number of slave nodes to be used in the cluster."
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--cores", type=int, default=4,
                        help="The number of cpu cores you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--memory", type=str, default="20g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--object_store_memory", type=str, default="10g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--enable_numa_binding", dest="enable_numa_binding", default=False,
                        help="enable_numa_binding")
    parser.add_argument('--k8s_master', type=str, default="",
                        help="The k8s master. "
                             "It should be k8s://https://<k8s-apiserver-host>: "
                             "<k8s-apiserver-port>.")
    parser.add_argument("--container_image", type=str, default="",
                        help="The runtime k8s image. ")
    parser.add_argument('--k8s_driver_host', type=str, default="",
                        help="The k8s driver localhost.")
    parser.add_argument('--k8s_driver_port', type=str, default="",
                        help="The k8s driver port.")
    parser.add_argument('--nfs_mount_path', type=str, default="",
                        help="nfs mount path")

    options = parser.parse_args()

    if options.cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores=options.cores, num_nodes=options.worker_num,
                          memory=options.memory, init_ray_on_spark=True,
                          object_store_memory=options.object_store_memory)
    elif options.cluster_mode == "k8s":
        init_orca_context(cluster_mode="k8s", master=options.k8s_master,
                          container_image=options.container_image,
                          init_ray_on_spark=True, enable_numa_binding=options.enable_numa_binding,
                          num_nodes=options.worker_num, cores=options.cores, memory=options.memory,
                          object_store_memory=options.object_store_memory,
                          conf={"spark.driver.host": options.driver_host,
                                "spark.driver.port": options.driver_port,
                                "spark.kubernetes.executor.volumes.persistentVolumeClaim."
                                "nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                "spark.kubernetes.executor.volumes.persistentVolumeClaim."
                                "nfsvolumeclaim.mount.path": options.nfs_mount_path,
                                "spark.kubernetes.driver.volumes.persistentVolumeClaim."
                                "nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                "spark.kubernetes.driver.volumes.persistentVolumeClaim."
                                "nfsvolumeclaim.mount.path": options.nfs_mount_path})
    elif options.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client", cores=options.cores,
                          num_nodes=options.worker_num, memory=options.memory,
                          init_ray_on_spark=True, enable_numa_binding=options.enable_numa_binding,
                          object_store_memory=options.object_store_memory)

    # convert yolov3 weights
    yolo = YoloV3(classes=80)
    load_darknet_weights(yolo, options.weights)
    yolo.save_weights(options.checkpoint)

    def model_creator(config):
        model = YoloV3(DEFAULT_IMAGE_SIZE, training=True, classes=options.class_num)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

        model_pretrained = YoloV3(
            DEFAULT_IMAGE_SIZE, training=True, classes=80)
        model_pretrained.load_weights(options.checkpoint)

        model.get_layer('yolo_darknet').set_weights(
            model_pretrained.get_layer('yolo_darknet').get_weights())
        freeze_all(model.get_layer('yolo_darknet'))

        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss = [YoloLoss(anchors[mask], classes=options.class_num)
                for mask in anchor_masks]
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=False)
        return model

    # prepare data
    class_map = {name: idx for idx, name in enumerate(
        open(options.names).read().splitlines())}
    dataset_path = os.path.join(options.data_dir, "VOCdevkit")
    voc_train_path = os.path.join(options.output_data, "train_dataset")
    voc_val_path = os.path.join(options.output_data, "val_dataset")

    write_parquet(format="voc", voc_root_path=dataset_path, output_path="file://" + voc_train_path,
                  splits_names=[(options.data_year, options.split_name_train)], classes=class_map)
    write_parquet(format="voc", voc_root_path=dataset_path, output_path="file://" + voc_val_path,
                  splits_names=[(options.data_year, options.split_name_test)], classes=class_map)

    output_types = {"image": tf.string, "label": tf.float32, "image_id": tf.string}
    output_shapes = {"image": (), "label": (None, 5), "image_id": ()}

    def train_data_creator(config, batch_size):
        train_dataset = read_parquet(format="tf_dataset", path=voc_train_path,
                                     output_types=output_types,
                                     output_shapes=output_shapes)
        train_dataset = train_dataset.map(
            lambda data_dict: (data_dict["image"], data_dict["label"]))
        train_dataset = train_dataset.map(parse_data_train)
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(lambda x, y: (
            transform_images(x, DEFAULT_IMAGE_SIZE),
            transform_targets(y, anchors, anchor_masks, DEFAULT_IMAGE_SIZE)))
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset

    def val_data_creator(config, batch_size):
        val_dataset = read_parquet(format="tf_dataset", path=voc_val_path,
                                   output_types=output_types,
                                   output_shapes=output_shapes)
        val_dataset = val_dataset.map(
            lambda data_dict: (data_dict["image"], data_dict["label"]))
        val_dataset = val_dataset.map(parse_data_train)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.map(lambda x, y: (
            transform_images(x, DEFAULT_IMAGE_SIZE),
            transform_targets(y, anchors, anchor_masks, DEFAULT_IMAGE_SIZE)))
        return val_dataset

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(options.checkpoint_folder + '/yolov3_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    trainer = Estimator.from_keras(model_creator=model_creator)

    trainer.fit(train_data_creator,
                epochs=options.epochs,
                batch_size=options.batch_size,
                steps_per_epoch=3473 // options.batch_size,
                callbacks=callbacks,
                validation_data=val_data_creator,
                validation_steps=3581 // options.batch_size)
    stop_orca_context()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
