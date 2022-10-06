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
import os
import click
import fnmatch
import sys

for files in os.listdir('/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/'):
    if fnmatch.fnmatch(files, 'bigdl-ppml-*-python-api.zip'):
        sys.path.append('/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/' + files)
        sys.path.append('/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/' + files + '/bigdl/ppml/fl/nn/generated')

from bigdl.ppml.fl.fl_server import FLServer

@click.command()
@click.option('--client_num', default=2)
@click.option('--port', default=8980)
@click.option('--servermodelpath', default='/tmp/fgboost_server_model')
def run(port, client_num, servermodelpath):
    conf = open('ppml-conf.yaml', 'w')
    conf.write('serverPort: ' + str(port) + '\n')
    conf.write('clientNum: ' + str(client_num) + '\n')
    conf.write('fgBoostServerModelPath: ' + servermodelpath + '\n')
    conf.close()

    fl_server = FLServer()
    fl_server.build()
    fl_server.start()
    fl_server.block_until_shutdown()

if __name__ == '__main__':
    run()

