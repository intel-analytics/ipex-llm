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

from bigdl.serving.client import InputQueue, OutputQueue


class TestClient:
    def test_default_config(self):
        input_api = InputQueue()
        output_api = OutputQueue()
        assert input_api.name == "serving_stream"
        assert input_api.host == "localhost"
        assert input_api.port == "6379"
        assert output_api.name == "serving_stream"
        assert output_api.host == "localhost"
        assert output_api.port == "6379"

    def test_input_queue_without_frontend(self):
        input_api = InputQueue(host="1.1.1.1", port="1111", name="my-test")
        assert input_api.name == "my-test"
        assert input_api.host == "1.1.1.1"
        assert input_api.port == "1111"
        assert not input_api.frontend_url

    def test_input_queue_with_frontend(self):
        input_api = InputQueue(
            host="1.1.1.1", port="1111", name="my-test", frontend_url="1.1.1.1:1")
        input_api2 = InputQueue(frontend_url="1.1.1.1:1")
        assert input_api.name == "my-test"
        assert input_api.host == "1.1.1.1"
        assert input_api.port == "1111"
        assert input_api.frontend_url == "1.1.1.1:1"
        assert input_api2.frontend_url == "1.1.1.1:1"

    def test_output_queue(self):
        output_api = OutputQueue(host="1.1.1.1", port="1111", name="my-test")
        assert output_api.name == "my-test"
        assert output_api.host == "1.1.1.1"
        assert output_api.port == "1111"
