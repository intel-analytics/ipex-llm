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

# This is the script to remove spark2 and spark3 suffix for all project
# names and dependencies.

file=$1

sed -i "s/name='bigdl-spark2'/name='bigdl'/g" ${file}
sed -i "s/name='bigdl-spark3'/name='bigdl'/g" ${file}

sed -i "s/bigdl-dllib-spark2==/bigdl-dllib==/g" ${file}
sed -i "s/bigdl-dllib-spark3==/bigdl-dllib==/g" ${file}
sed -i "s/name='bigdl-dllib-spark2'/name='bigdl-dllib'/g" ${file}
sed -i "s/name='bigdl-dllib-spark3'/name='bigdl-dllib'/g" ${file}
sed -i "s/dist\/bigdl_dllib_spark2-/dist\/bigdl_dllib-/g" ${file}
sed -i "s/dist\/bigdl_dllib_spark3-/dist\/bigdl_dllib-/g" ${file}

sed -i "s/bigdl-orca-spark2==/bigdl-orca==/g" ${file}
sed -i "s/bigdl-orca-spark3==/bigdl-orca==/g" ${file}
sed -i "s/name='bigdl-orca-spark2'/name='bigdl-orca'/g" ${file}
sed -i "s/name='bigdl-orca-spark3'/name='bigdl-orca'/g" ${file}
sed -i "s/dist\/bigdl_orca_spark2-/dist\/bigdl_orca-/g" ${file}
sed -i "s/dist\/bigdl_orca_spark3-/dist\/bigdl_orca-/g" ${file}
sed -i "s/bigdl-orca-spark2\[ray\]/bigdl-orca\[ray\]/g" ${file}
sed -i "s/bigdl-orca-spark3\[ray\]/bigdl-orca\[ray\]/g" ${file}

sed -i "s/bigdl-chronos-spark2==/bigdl-chronos==/g" ${file}
sed -i "s/bigdl-chronos-spark3==/bigdl-chronos==/g" ${file}
sed -i "s/name='bigdl-chronos-spark2'/name='bigdl-chronos'/g" ${file}
sed -i "s/name='bigdl-chronos-spark3'/name='bigdl-chronos'/g" ${file}
sed -i "s/dist\/bigdl_chronos_spark2-/dist\/bigdl_chronos-/g" ${file}
sed -i "s/dist\/bigdl_chronos_spark3-/dist\/bigdl_chronos-/g" ${file}

sed -i "s/bigdl-friesian-spark2==/bigdl-friesian==/g" ${file}
sed -i "s/bigdl-friesian-spark3==/bigdl-friesian==/g" ${file}
sed -i "s/name='bigdl-friesian-spark2'/name='bigdl-friesian'/g" ${file}
sed -i "s/name='bigdl-friesian-spark3'/name='bigdl-friesian'/g" ${file}
sed -i "s/dist\/bigdl_friesian_spark2-/dist\/bigdl_friesian-/g" ${file}
sed -i "s/dist\/bigdl_friesian_spark3-/dist\/bigdl_friesian-/g" ${file}
