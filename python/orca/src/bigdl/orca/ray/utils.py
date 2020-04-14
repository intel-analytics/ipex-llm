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

import re
import os
import signal


def to_list(input):
    if isinstance(input, (list, tuple)):
        return list(input)
    else:
        return [input]


def resource_to_bytes(resource_str):
    if not resource_str:
        return resource_str
    matched = re.compile("([0-9]+)([a-z]+)?").match(resource_str.lower())
    fraction_matched = re.compile("([0-9]+\\.[0-9]+)([a-z]+)?").match(resource_str.lower())
    if fraction_matched:
        raise Exception(
            "Fractional values are not supported. Input was: {}".format(resource_str))
    try:
        value = int(matched.group(1))
        postfix = matched.group(2)
        if postfix == 'b':
            value = value
        elif postfix == 'k':
            value = value * 1000
        elif postfix == "m":
            value = value * 1000 * 1000
        elif postfix == 'g':
            value = value * 1000 * 1000 * 1000
        else:
            raise Exception("Not supported type: {}".format(resource_str))
        return value
    except Exception:
        raise Exception("Size must be specified as bytes(b),"
                        "kilobytes(k), megabytes(m), gigabytes(g). "
                        "E.g. 50b, 100k, 250m, 30g")


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
        for pgid in effect_pgids:
            print("Stopping by pgid {}".format(pgid))
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                print("WARNING: cannot kill pgid: {}".format(pgid))

    return _shutdown_per_node


def is_local(sc):
    master = sc.getConf().get("spark.master")
    return master == "local" or master.startswith("local[")
