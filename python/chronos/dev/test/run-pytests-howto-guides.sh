#!/usr/bin/env bash

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

export Chronos_Howto_Guides_dir=${ANALYTICS_ZOO_ROOT}/python/chronos/colab-notebook/howto

# limit the number of epoch to reduce test time
sed -i 's/epochs=./epochs=1/' $Chronos_Howto_Guides_dir/*.ipynb
# comment out the install commands
sed -i 's/!pip install/#!pip install/' $Chronos_Howto_Guides_dir/*.ipynb

echo "Running tests for Chronos How-to Guides"
python -m pytest -v  --nbmake --nbmake-timeout=6000 --nbmake-kernel=python3 ${Chronos_Howto_Guides_dir}

exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi

echo "Tests for Chronos How-to Guides finished."
