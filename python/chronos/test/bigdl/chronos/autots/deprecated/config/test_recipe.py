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

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.autots.deprecated.config.recipe import *


class TestTimeSequencePredictor(ZooTestCase):

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_SmokeRecipe(self):
        recipe = SmokeRecipe()
        assert recipe.num_samples == 1
        assert recipe.training_iteration == 1
        assert recipe.reward_metric is None

    def test_MTNetSmokeRecipe(self):
        recipe = MTNetSmokeRecipe()
        assert recipe.num_samples == 1
        assert recipe.training_iteration == 1
        assert recipe.reward_metric is None

    def test_GridRandomRecipe(self):
        recipe = GridRandomRecipe(num_rand_samples=100)
        search_space = recipe.search_space()
        assert search_space is not None

    def test_RandomRecipe(self):
        recipe = GridRandomRecipe(num_rand_samples=100)
        search_space = recipe.search_space()
        assert search_space is not None

    def test_LSTMGridRandomRecipe(self):
        recipe = LSTMGridRandomRecipe(num_rand_samples=100)
        search_space = recipe.search_space()
        assert search_space is not None

    def test_MTNetGridRandomRecipe(self):
        recipe = MTNetGridRandomRecipe()
        search_space = recipe.search_space()
        assert search_space is not None

    def test_BayesRecipe(self):
        recipe = BayesRecipe(num_samples=10)
        search_space = recipe.search_space()
        assert search_space is not None
        assert recipe.reward_metric is not None


if __name__ == '__main__':
    pytest.main([__file__])
