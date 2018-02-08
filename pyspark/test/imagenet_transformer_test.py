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

from nn.layer import *
from optim.optimizer import *
from util.common import *
import unittest
from dataset.imagenet import *
from dataset.transformer import *


class TestImagenetTransformer(unittest.TestCase):
    def setUp(self):
        sparkConf = create_spark_conf()
        self.sc = SparkContext(master="local[4]", appName="test imagenet",
                               conf=sparkConf)
        init_engine()

    def tearDown(self):
        self.sc.stop()

    def test_read_local(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        image_folder = os.path.join(resource_path, "imagenet/local")
        image_rdd = read_local(self.sc, image_folder)
        self.assertEqual(image_rdd.count(), 3)
        for sample in image_rdd.collect():
            features = sample[0]
            self.assertEqual(features.shape, (256, 256, 3))
            self.assertGreaterEqual(features[1, 1, 0], 0.0)
            self.assertLessEqual(features[1, 1, 0], 1.0)
            self.assertEqual(features.dtype, "float32")
            label = sample[1]
            self.assertTrue(np.allclose(label, 1.0, atol=1e-6, rtol=0))
            self.assertEqual(label.size, 1)

    def test_read_seq(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        image_folder = os.path.join(resource_path, "imagenet/seq")
        print(image_folder)
        image_rdd = read_seq_file(self.sc, "file://" + image_folder)
        self.assertEqual(image_rdd.count(), 3)
        for sample in image_rdd.collect():
            features = sample[0]
            self.assertEqual(features.shape, (256, 256, 3))
            self.assertGreaterEqual(features[1, 1, 0], 0.0)
            self.assertLessEqual(features[1, 1, 0], 1.0)
            self.assertEqual(features.dtype, "float32")
            label = sample[1]
            self.assertTrue(np.allclose(label, 1.0, atol=1e-6, rtol=0))
            self.assertEqual(label.size, 1)

    def test_transformer(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        image_folder = os.path.join(resource_path, "imagenet/test")
        features_rdd = read_local(self.sc, image_folder).map(
            lambda features_label: features_label[0])
        self.assertEqual(features_rdd.count(), 1)
        features_org = []
        for features in features_rdd.collect():
            features_org.append(features)

        # test pixel mean file
        mean_file = os.path.join(os.path.join(resource_path, "imagenet"), "ilsvrc_2012_mean.npy")
        mean = load_mean_file(mean_file)
        pixel_normailized_list = []
        pixel_normalized = features_rdd.map(PixelNormalizer(mean))
        for features in pixel_normalized.collect():
            pixel_normailized_list.append(features)
        self.assertTrue(np.allclose(pixel_normailized_list[0] + mean,
                                    features_org[0],
                                    atol=1e-6, rtol=0))

        # test channel normalizer
        channel_normailized_list = []
        channel_normalized = features_rdd.map(ChannelNormalizer(0.1, 0.2, 0.3, 0.1, 0.1, 0.1))
        for sample in channel_normalized.collect():
            channel_normailized_list.append(sample)
        self.assertTrue(
            np.allclose(channel_normailized_list[0][:, :, 0] * 0.1 + 0.3,
                        features_org[0][:, :, 0],
                        atol=1e-6, rtol=0))
        self.assertTrue(
            np.allclose(channel_normailized_list[0][:, :, 1] * 0.1 + 0.2,
                        features_org[0][:, :, 1],
                        atol=1e-6, rtol=0))
        self.assertTrue(
            np.allclose(channel_normailized_list[0][:, :, 2] * 0.1 + 0.1,
                        features_org[0][:, :, 2],
                        atol=1e-6, rtol=0))

        # test crop
        cropped1_list = []
        cropped1 = features_rdd.map(Crop(227, 227, "random"))
        for sample in cropped1.collect():
            cropped1_list.append(sample)
        self.assertEqual(cropped1_list[0].shape, (227, 227, 3))
        cropped2_list = []
        cropped2 = features_rdd.map(Crop(224, 224, "center"))
        for sample in cropped2.collect():
            cropped2_list.append(sample)
        self.assertEqual(cropped2_list[0].shape, (224, 224, 3))
        self.assertTrue(np.allclose(cropped2_list[0][0, 0, :],
                                    features_org[0][(256-224)/2, (256-224)/2, :],
                                    atol=1e-6, rtol=0))

        # test flip
        filpped1_list = []
        flipped1 = features_rdd.map(Flip(0))
        for sample in flipped1.collect():
            filpped1_list.append(sample)
        self.assertTrue(np.allclose(filpped1_list[0][0, 1, :],
                                    features_org[0][0, 254, :],
                                    atol=1e-6, rtol=0))

        filpped2_list = []
        flipped2 = features_rdd.map(Flip(1))
        for sample in flipped2.collect():
            filpped2_list.append(sample)
        self.assertTrue(np.allclose(filpped2_list[0],
                                    features_org[0],
                                    atol=1e-6, rtol=0))

        # test transpose
        transposed1_list = []
        transposed1 = features_rdd.map(TransposeToTensor())
        for sample in transposed1.collect():
            transposed1_list.append(sample)
        self.assertEqual(transposed1_list[0].shape, (3, 256, 256))
        self.assertTrue(np.allclose(transposed1_list[0][0, :, :],
                                    features_org[0][:, :, 2],
                                    atol=1e-6, rtol=0))
        transposed2_list = []
        transposed2 = features_rdd.map(TransposeToTensor(False))
        for sample in transposed2.collect():
            transposed2_list.append(sample)
        self.assertEqual(transposed2_list[0].shape, (3, 256, 256))

if __name__ == "__main__":
    unittest.main(failfast=True)
