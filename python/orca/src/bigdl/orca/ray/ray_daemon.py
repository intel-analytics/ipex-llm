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
import sys
import signal
import psutil
import logging
logging.basicConfig(filename='daemon.log', level=logging.INFO)


def stop(pgid):
    logging.info(f"Stopping pgid {pgid} by ray_daemon.")
    try:
        # SIGTERM may not kill all the children processes in the group.
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        logging.error("Cannot kill pgid: {}".format(pgid))


def manager():
    pid_to_watch = int(sys.argv[1])
    pgid_to_kill = int(sys.argv[2])
    import time
    while psutil.pid_exists(pid_to_watch):
        time.sleep(1)
    stop(pgid_to_kill)


if __name__ == "__main__":
    manager()
