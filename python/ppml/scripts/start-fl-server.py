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

import sys
import os
import fnmatch
import getopt

for files in os.listdir('/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/'):
    if fnmatch.fnmatch(files, 'bigdl-ppml-*-python-api.zip'):
        sys.path.append('/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/' + files)
        sys.path.append('/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/' + files + '/bigdl/ppml/fl/nn/generated')

from bigdl.ppml.fl.nn.fl_server import FLServer

if __name__ == '__main__':

    client_num = 2
    port = 8980
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:p:", ["client-num=", "port="])
    except getopt.GetoptError:
        print("start_fl_server.py -c <client-num> -p <port>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("start_fl_server.py -c <client-num> -p <port>")
        elif opt in ("-c", "--client-num"):
            client_num = arg
        elif opt in ("-p", "--port"):
            port = arg

    fl_server = FLServer(client_num)
    fl_server.set_port(port)
    fl_server.build()
    fl_server.start()

    fl_server.wait_for_termination()
