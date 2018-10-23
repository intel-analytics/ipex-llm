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

import pytest
import os
from bigdl.util.common import *
from bigdl.transform.vision.image import *
import tempfile


class TestLayer():

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        init_engine()
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        self.image_path = os.path.join(resource_path, "pascal/000025.jpg")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_get_sample(self):
        image_frame = ImageFrame.read(self.image_path)
        transformer = Pipeline([PixelBytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample()])
        transformed = transformer(image_frame)
        transformed.get_sample()

    def transformer_test(self, transformer):
        image_frame = ImageFrame.read(self.image_path)
        transformed = transformer(image_frame)
        transformed.get_image()

        image_frame = ImageFrame.read(self.image_path, self.sc)
        transformed = transformer(image_frame)
        images = transformed.get_image()
        images.count()

    def test_get_image(self):
        image_frame = ImageFrame.read(self.image_path)
        image_frame.get_image()

    def test_get_label(self):
        image_frame = ImageFrame.read(self.image_path)
        image_frame.get_label()

    def test_is_local(self):
        image_frame = ImageFrame.read(self.image_path)
        assert image_frame.is_local() is True
        image_frame = ImageFrame.read(self.image_path, self.sc)
        assert image_frame.is_local() is False

    def test_is_distributed(self):
        image_frame = ImageFrame.read(self.image_path)
        assert image_frame.is_distributed() is False
        image_frame = ImageFrame.read(self.image_path, self.sc)
        assert image_frame.is_distributed() is True

    def test_hflip(self):
        transformer = HFlip()
        self.transformer_test(transformer)

    def test_colorjitter(self):
        color = ColorJitter(random_order_prob=1.0, shuffle=True)
        self.transformer_test(color)

    def test_resize(self):
        resize = Resize(200, 200, 1)
        self.transformer_test(resize)

    def test_brightness(self):
        brightness = Brightness(0.0, 32.0)
        self.transformer_test(brightness)

    def test_channel_order(self):
        transformer = ChannelOrder()
        self.transformer_test(transformer)

    def test_aspect_scale(self):
        transformer = AspectScale(300)
        self.transformer_test(transformer)

    def test_random_aspect_scale(self):
        transformer = RandomAspectScale([300, 400])
        self.transformer_test(transformer)

    def test_contrast(self):
        transformer = Contrast(0.5, 1.5)
        self.transformer_test(transformer)

    def test_saturation(self):
        transformer = Saturation(0.5, 1.5)
        self.transformer_test(transformer)

    def test_hue(self):
        transformer = Hue(0.5, 1.5)
        self.transformer_test(transformer)

    def test_channel_normalize(self):
        transformer = ChannelNormalize(100.0, 200.0, 300.0, 2.0, 2.0, 2.0)
        self.transformer_test(transformer)

    def test_pixel_normalize(self):
        means = [2.0] * 3 * 500 * 375
        transformer = PixelNormalize(means)
        self.transformer_test(transformer)

    def test_fixed_crop_norm(self):
        crop = FixedCrop(0.0, 0.0, 0.5, 1.0)
        self.transformer_test(crop)

    def test_fixed_crop_unnorm(self):
        crop = FixedCrop(0.0, 0.0, 200.0, 200., False)
        self.transformer_test(crop)

    def test_center_crop(self):
        crop = CenterCrop(200, 200)
        self.transformer_test(crop)

    def test_random_crop(self):
        crop = RandomCrop(200, 200)
        self.transformer_test(crop)

    def test_filler(self):
        filler = Filler(0.0, 0.0, 0.1, 0.2)
        self.transformer_test(filler)

    def test_expand(self):
        expand = Expand(means_r=123, means_g=117, means_b=104,
                        max_expand_ratio=2.0)
        self.transformer_test(expand)

    def test_fix_expand(self):
        expand = FixExpand(1000, 1000)
        self.transformer_test(expand)

    def test_random_transformer(self):
        transformer = RandomTransformer(HFlip(), 0.5)
        self.transformer_test(transformer)

    def test_pipeline(self):
        transformer = Pipeline([ColorJitter(), HFlip(), Resize(200, 200, 1)])
        self.transformer_test(transformer)

    def test_inception_preprocess(self):
        transformer = Pipeline([Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample()])
        self.transformer_test(transformer)

    def test_mat_to_floats(self):
        transformer = MatToFloats()
        self.transformer_test(transformer)

    def test_mat_to_floats_no_share(self):
        transformer = MatToFloats(share_buffer=False)
        self.transformer_test(transformer)

    def test_mat_to_tensor(self):
        transformer = MatToTensor()
        self.transformer_test(transformer)

    def testImageFrameToSample(self):
        transformer = Pipeline([MatToTensor(), ImageFrameToSample()])
        self.transformer_test(transformer)

    def test_image_frame_transform(self):
        transformer = MatToTensor()
        image_frame = ImageFrame.read(self.image_path)
        transformed = image_frame.transform(transformer)
        transformed.get_image()

    def test_empty_get_predict_local(self):
        image_frame = ImageFrame.read(self.image_path)
        image_frame.get_predict()

    def test_empty_get_predict_distributed(self):
        image_frame = ImageFrame.read(self.image_path, self.sc)
        image_frame.get_predict()

    def test_read_write_parquet(self):
        temp = tempfile.mkdtemp() + "testParquet/"
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources/pascal")
        ImageFrame.write_parquet(resource_path, temp, self.sc, 1)
        read_image_frame = ImageFrame.read_parquet(temp, self.sc)

    def test_set_label(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources/pascal")
        imageFrame = ImageFrame.read(resource_path, self.sc)
        uris = imageFrame.get_uri().collect()
        label = {}
        for uri in uris:
            label[uri] = 10
        imageFrame.set_label(label)

    def test_random_split(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources/pascal")
        imageFrame = ImageFrame.read(resource_path, self.sc)
        splits = imageFrame.random_split([1.0])

    def test_channel_scaled_normalizer(self):
        transformer = ChannelScaledNormalizer(123, 117, 104, 1.0)
        self.transformer_test(transformer)

    def test_random_alter_aspect(self):
        transformer = RandomAlterAspect(0.08, 1, 0.75, "CUBIC", 20)
        self.transformer_test(transformer)

    def test_random_cropper(self):
        transformer = RandomCropper(20, 20, True, "Random", 3)
        self.transformer_test(transformer)

    def test_random_resize(self):
        transformer = RandomResize(100, 100)
        self.transformer_test(transformer)

if __name__ == "__main__":
    pytest.main([__file__])
