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

import os.path
import shutil
import tempfile

from bigdl.orca.data.file import open_image, open_text, load_numpy, exists, makedirs, write_text


class TestFile:
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")

    def test_open_local_text(self):
        file_path = os.path.join(self.resource_path, "qa/relations.txt")
        lines = open_text(file_path)
        assert lines == ["Q1,Q1,1", "Q1,Q2,0", "Q2,Q1,0", "Q2,Q2,1"]

    def test_open_local_text_2(self):
        file_path = os.path.join(self.resource_path, "qa/relations.txt")
        lines = open_text("file://" + file_path)
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

    def test_open_local_image_2(self):
        file_path = os.path.join(self.resource_path, "cat_dog/cats/cat.7000.jpg")
        image = open_image("file://" + file_path)

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

    def test_load_local_numpy_2(self):
        file_path = os.path.join(self.resource_path, "orca/data/random.npy")
        res = load_numpy("file://" + file_path)
        assert res.shape == (2, 5)

    def test_load_s3_numpy(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/hyperseg/VGGcompression/core1.npy"
            res = load_numpy(file_path)
            assert res.shape == (32, 64, 3, 3)

    def test_exists_local(self):
        file_path = os.path.join(self.resource_path, "orca/data/random.npy")
        assert exists(file_path)
        file_path = os.path.join(self.resource_path, "orca/data/abc.npy")
        assert not exists(file_path)

    def test_exists_local(self):
        file_path = os.path.join(self.resource_path, "orca/data/random.npy")
        assert exists("file://" + file_path)
        file_path = os.path.join(self.resource_path, "orca/data/abc.npy")
        assert not exists("file://" + file_path)

    def test_exists_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/nyc_taxi.csv"
            assert exists(file_path)
            file_path = "s3://analytics-zoo-data/abc.csv"
            assert not exists(file_path)

    def test_mkdirs_local(self):
        temp = tempfile.mkdtemp()
        path = os.path.join(temp, "dir1")
        makedirs(path)
        assert exists(path)
        path = os.path.join(temp, "dir2/dir3")
        makedirs(path)
        assert exists(path)
        shutil.rmtree(temp)

    def test_mkdirs_local_2(self):
        temp = tempfile.mkdtemp()
        path = os.path.join(temp, "dir1")
        makedirs("file://" + path)
        assert exists("file://" + path)
        path = os.path.join(temp, "dir2/dir3")
        makedirs("file://" + path)
        assert exists("file://" + path)
        shutil.rmtree(temp)

    def test_mkdirs_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/temp/abc/"
            makedirs(file_path)
            assert exists(file_path)
            import boto3
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key).client('s3', verify=False)
            s3_client.delete_object(Bucket='analytics-zoo-data', Key='temp/abc/')

    def test_write_text_local(self):
        temp = tempfile.mkdtemp()
        path = os.path.join(temp, "test.txt")
        write_text(path, "abc\n")
        text = open_text(path)
        shutil.rmtree(temp)
        assert text == ['abc']

    def test_write_text_local_2(self):
        temp = tempfile.mkdtemp()
        path = os.path.join(temp, "test.txt")
        write_text("file://" + path, "abc\n")
        text = open_text("file://" + path)
        shutil.rmtree(temp)
        assert text == ['abc']

    def test_write_text_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/test.txt"
            text = 'abc\ndef\n'
            write_text(file_path, text)
            lines = open_text(file_path)
            assert lines == ['abc', 'def']
            import boto3
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key).client('s3', verify=False)
            s3_client.delete_object(Bucket='analytics-zoo-data', Key='test.txt')
