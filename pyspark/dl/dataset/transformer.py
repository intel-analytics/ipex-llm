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
import numpy as np



def normalizer(mean, std):
    """
    Normalize features by standard deviation
    """
    return lambda sample: Sample.from_ndarray((sample.features - mean) / std,
                                              sample.label, sample.bigdl_type)


def pixel_normalizer(mean):
    return lambda sample: Sample.from_ndarray((sample.features - mean),
                                              sample.label, sample.bigdl_type)


def crop(crop_width, crop_height, crop_method='random'):
    def func(sample):
        img = sample.features.astype(dtype='uint8')
        h = sample.features.shape[0]
        w = sample.features.shape[1]
        if crop_method == 'random':
            x1 = random.randint(0, w - crop_width)
            y1 = random.randint(0, h - crop_height)
        elif crop_method == 'center':
            x1 = (w - crop_width)/2
            y1 = (h - crop_height)/2
        cropped = img[y1:y1+crop_height, x1:x1+crop_width]
        return Sample.from_ndarray(cropped, sample.label, sample.bigdl_type)
    return func


def channel_normalizer(mean_b, mean_g, mean_r, std_b, std_g, std_r):
    def func(sample):
        mean = np.array([mean_b, mean_g, mean_r])
        std = np.array([std_b, std_g, std_r])
        mean_sub = sample.features[:,:] - mean
        result = mean_sub[:,:] / std
        # dtype = Sample.get_dtype(sample.bigdl_type)
        # shape = sample.features.shape
        # target = np.empty(shape, dtype=dtype)
        # for i in xrange(shape(0)):
        #     for j in xrange(shape(1)):
        #         target[i, j] = (sample.features[i, j] - mean) / std
        return Sample.from_ndarray(result, sample.label, sample.bigdl_type)
    return func


def flip(threshold):
    def func(sample):
        if random.random() > threshold:
            return Sample.from_ndarray(np.flip(sample.features, 1), sample.label, sample.bigdl_type)
        else:
            return sample
    return func


def transpose(to_rgb=True):
    def func(sample):
        if to_rgb:
            result = sample.features.transpose(2,0,1)[(2,1,0),:,:]
        else:
            result = sample.features.transpose(2,0,1)
        return Sample.from_ndarray(result, sample.label,sample.bigdl_type)
    return func
