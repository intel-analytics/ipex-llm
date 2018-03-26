#!/bin/sh
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

#
# This script is to create python dependency libraries and submit python jobs to spark yarn cluster which need to import these modules. 
# 
# After running this script, you will get vent.zip. Use this zip file to submit spark job on yarn clusters. Please refer to pythn_submit_yarn_cluster.sh.template to submit python job.
# 

pip install virtualenv

#create package
VENV="venv"
virtualenv $VENV
virtualenv --relocatable $VENV
. $VENV/bin/activate
pip install -U -r requirements.txt
zip -q -r $VENV.zip $VENV



