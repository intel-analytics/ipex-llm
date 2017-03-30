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
# After running this script, you will get vent.zip. Use this zip file to submit spark job on yarn clusters. Please refer to pythn_submit_yarn.sh.template to submit python job.
# 

if [ "$(cat /proc/version | grep -c -i "centos")" != "0" ]; then
	SYSTEM="CentOs"
elif [ "$(cat /proc/version | grep -c -i "ubuntu")" != "0" ]; then
	SYSTEM="Ubuntu"
fi

if [ $SYSTEM = "CentOS" ] ; then
	sudo yum update
	sudo yum -y install python-setuptools python-devel
	sudo yum install -y gcc make
	sudo yum install -y zip
elif [ $SYSTEM = "Ubuntu" ] ; then
	sudo apt-get update
	sudo apt-get install -y python-setuptools python-dev
	sudo apt-get install -y gcc make
	sudo apt-get install -y zip
else 
	echo "Other OS" 
fi

sudo easy_install pip
pip install virtualenv

#create package
VENV="venv"
virtualenv $VENV
virtualenv --relocatable $VENV
. $VENV/bin/activate
pip install -U -r requirements.txt
zip -q -r $VENV.zip $VENV
rm -rf $VENV



