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

import os.path

from zoo.orca.data.file import open_image, open_text, load_numpy


class TestFile:
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

    def test_open_local_text(self):
        file_path = os.path.join(self.resource_path, "qa/relations.txt")
        lines = open_text(file_path)
        assert lines == ["Q1,Q1,1", "Q1,Q2,0", "Q2,Q1,0", "Q2,Q2,1"]

    def test_open_s3_text(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/hyperseg/trainingData/train_tiled.txt"
            lines = open_text(file_path)
            assert lines[0] == "CONTENTAI_000001"

    def test_open_local_image(self):
        file_path = os.path.join(self.resource_path, "cat_dog/cats/cat.7000.jpg")
        image = open_image(file_path)

    def test_open_s3_image(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/dogs-vs-cats/samples/cat.7000.jpg"
            image = open_image(file_path)

    def test_load_local_numpy(self):
        file_path = os.path.join(self.resource_path, "orca/data/random.npy")
        res = load_numpy(file_path)
        assert res.shape == (2, 5)

    def test_load_s3_numpy(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/hyperseg/VGGcompression/core1.npy"
            res = load_numpy(file_path)
            assert res.shape == (32, 64, 3, 3)
