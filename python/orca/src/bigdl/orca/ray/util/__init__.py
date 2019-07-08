#
# Copyright 2018 Analytics Zoo Authors.
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
import signal

from zoo.ray.util.utils import to_list


def gen_shutdown_per_node(pgids, node_ips=None):
    import ray.services as rservices
    pgids = to_list(pgids)

    def _shutdown_per_node(iter):
        print("Stopping pgids: {}".format(pgids))
        if node_ips:
            current_node_ip = rservices.get_node_ip_address()
            effect_pgids = [pair[0] for pair in zip(pgids, node_ips) if pair[1] == current_node_ip]
        else:
            effect_pgids = pgids
        for pgid in pgids:
            print("Stopping by pgid {}".format(pgid))
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                print("WARNING: cannot kill pgid: {}".format(pgid))

    return _shutdown_per_node


def is_local(sc):
    master = sc._conf.get("spark.master")
    return master == "local" or master.startswith("local[")
