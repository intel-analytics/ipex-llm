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

# This is the script to add spark3 suffix or change spark2 suffix to spark3
# for all project names and dependencies.

# Add name=, == and - in pattern matching so that if the script runs twice,
# it won't change anything in the second run.

file=$1

sed -i "s/name='bigdl'/name='bigdl-spark3'/g" ${file}
sed -i "s/name='bigdl-spark2'/name='bigdl-spark3'/g" ${file}

sed -i "s/bigdl-dllib==/bigdl-dllib-spark3==/g" ${file}
sed -i "s/bigdl-dllib-spark2==/bigdl-dllib-spark3==/g" ${file}
sed -i "s/name='bigdl-dllib'/name='bigdl-dllib-spark3'/g" ${file}
sed -i "s/name='bigdl-dllib-spark2'/name='bigdl-dllib-spark3'/g" ${file}
sed -i "s/dist\/bigdl_dllib-/dist\/bigdl_dllib_spark3-/g" ${file}
sed -i "s/dist\/bigdl_dllib_spark2-/dist\/bigdl_dllib_spark3-/g" ${file}

sed -i "s/bigdl-orca==/bigdl-orca-spark3==/g" ${file}
sed -i "s/bigdl-orca-spark2==/bigdl-orca-spark3==/g" ${file}
sed -i "s/name='bigdl-orca'/name='bigdl-orca-spark3'/g" ${file}
sed -i "s/name='bigdl-orca-spark2'/name='bigdl-orca-spark3'/g" ${file}
sed -i "s/dist\/bigdl_orca-/dist\/bigdl_orca_spark3-/g" ${file}
sed -i "s/dist\/bigdl_orca_spark2-/dist\/bigdl_orca_spark3-/g" ${file}
sed -i "s/bigdl-orca\[ray\]/bigdl-orca-spark3\[ray\]/g" ${file}
sed -i "s/bigdl-orca-spark2\[ray\]/bigdl-orca-spark3\[ray\]/g" ${file}

sed -i "s/bigdl-chronos==/bigdl-chronos-spark3==/g" ${file}
sed -i "s/bigdl-chronos-spark2==/bigdl-chronos-spark3==/g" ${file}
sed -i "s/name='bigdl-chronos'/name='bigdl-chronos-spark3'/g" ${file}
sed -i "s/name='bigdl-chronos-spark2'/name='bigdl-chronos-spark3'/g" ${file}
sed -i "s/dist\/bigdl_chronos-/dist\/bigdl_chronos_spark3-/g" ${file}
sed -i "s/dist\/bigdl_chronos_spark2-/dist\/bigdl_chronos_spark3-/g" ${file}

sed -i "s/bigdl-friesian==/bigdl-friesian-spark3==/g" ${file}
sed -i "s/bigdl-friesian-spark2==/bigdl-friesian-spark3==/g" ${file}
sed -i "s/name='bigdl-friesian'/name='bigdl-friesian-spark3'/g" ${file}
sed -i "s/name='bigdl-friesian-spark2'/name='bigdl-friesian-spark3'/g" ${file}
sed -i "s/dist\/bigdl_friesian-/dist\/bigdl_friesian_spark3-/g" ${file}
sed -i "s/dist\/bigdl_friesian_spark2-/dist\/bigdl_friesian_spark3-/g" ${file}
