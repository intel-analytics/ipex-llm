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
from unittest import TestCase

import bigdl.nano.automl as nano_automl

class TestVisualization(TestCase):

    def test_import_should_okay(self):
        try:
            from bigdl.nano.automl.hpo.visualization import plot_optimization_history
        except ImportError:
            self.fail("cannot import plot_optimization_history from nano.aotoml.hpo.visualization.")
        try:
            from bigdl.nano.automl.hpo.visualization import plot_parallel_coordinate
        except ImportError:
            self.fail("cannot import plot_parallel_coordinate from nano.aotoml.hpo.visualization.")
        try:
            from bigdl.nano.automl.hpo.visualization import plot_intermediate_values
        except ImportError:
            self.fail("cannot import plot_intermediate_values from nano.aotoml.hpo.visualization.")
        try:
            from bigdl.nano.automl.hpo.visualization import plot_contour
        except ImportError:
            self.fail("cannot import plot_contour from nano.aotoml.hpo.visualization.")
        try:
            from bigdl.nano.automl.hpo.visualization import plot_param_importances
        except ImportError:
            self.fail("cannot import plot_param_importances from nano.aotoml.hpo.visualization.")

if __name__ == '__main__':
    pytest.main([__file__])