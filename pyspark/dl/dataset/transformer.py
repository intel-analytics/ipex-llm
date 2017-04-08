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


from util.common import Sample
import random


def normalizer(mean, std):
    """
    Normalize features by standard deviation
    """
    return lambda sample: Sample.from_ndarray((sample.features - mean) / std,
                                              sample.label, sample.bigdl_type)

def bgr_pixel_normalizer(mean):
    return lambda record: Sample([feature - m for feature, m in zip(record.features, mean)],
                                 record.label, record.features_shape,
                                 record.label_shape, record.bigdl_type)


def bgr_crop(crop_width, crop_height, crop_method='random'):
    def func(record):
        h = record.features_shape[0]
        w = record.features_shape[1]
        if crop_method == 'random':
            x1 = random.randint(0, w - crop_width)
            y1 = random.randint(0, h - crop_height)
        elif crop_method == 'center':
            x1 = (w - crop_width)/2
            y1 = (h - crop_height)/2
        target = []
        start = (x1 + y1 * w) * 3
        for i in xrange(crop_width * crop_height):
            target.append(record.features[start + ((i / crop_width) * w +
                                                   (i % crop_height)) * 3])
            target.append(record.features[start + ((i / crop_width) * w +
                                                   (i % crop_height)) * 3 + 1])
            target.append(record.features[start + ((i / crop_width) * w +
                                                   (i % crop_height)) * 3 + 2])
        return Sample(target,
                      record.label,
                      [crop_height, crop_width, record.features_shape[2]],
                      record.label_shape, record.bigdl_type)
    return func


def bgr_normalizer(mean, std):
    def func(record):
        result = []
        for i in range(record.features_shape(0) * record.features_shape(1)):
            result.append((record.features[i * 3 + 0] - mean[0]) / std[0])
            result.append((record.features[i * 3 + 1] - mean[1]) / std[1])
            result.append((record.features[i * 3 + 2] - mean[0]) / std[2])
        return Sample(result,
                      record.label,
                      record.features_shape,
                      record.label_shape,
                      record.bigdl_type)
    return func


# def flip(record, threshold):
#     img = np.array(record.features).reshape(record.features_shape)
#     if random.random() > threshold:
#         np.fliplr(img)
#     return Sample([float(i) for i in img.ravel()], record.label, list(img.shape),
#                   record.label_shape, record.bigdl_type)


# def scale(record, resize_width, resize_height):
#     img = np.array(record.features).reshape(record.features_shape)
#     misc.imresize(img, (resize_width, resize_height))
#     return Sample([float(i) for i in img.ravel()], record.label, list(img.shape),
#                   record.label_shape, record.bigdl_type)


def bgr_flip(threshold):
    def f(record):
        if random.random() > threshold:
            data = record.features
            height = record.features_shape[0]
            width = record.features_shape[1]
            for y in xrange(height):
                for x in xrange(width / 2):
                    swap = data[(y * width + x) * 3]
                    data[(y * width + x) * 3] = data[(y * width + width - x - 1) * 3]
                    data[(y * width + width - x - 1) * 3] = swap

                    swap = data[(y * width + x) * 3 + 1]
                    data[(y * width + x) * 3 + 1] = data[(y * width + width - x - 1) * 3 + 1]
                    data[(y * width + width - x - 1) * 3 + 1] = swap

                    swap = data[(y * width + x) * 3 + 2]
                    data[(y * width + x) * 3 + 2] = data[(y * width + width - x - 1) * 3 + 2]
                    data[(y * width + width - x - 1) * 3 + 2] = swap
            return Sample(data, record.label, record.features_shape,
                          record.label_shape, record.bigdl_type)
        else:
            return record
    return f