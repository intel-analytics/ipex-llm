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

import argparse
import errno
import glob
import json
import logging
import os
import shutil
import time
import tempfile
import re
import pickle
import subprocess
import zmq

logger = logging.getLogger(__name__)

class LogMonitor:
    """
    A monitor process for monitoring worker log files.

    """

    def __init__(self, driver_ip, driver_port, log_path, threads_stopped, partition_id):
        """Initialize the log monitor object."""
        from bigdl.dllib.utils.utils import get_node_ip
        self.ip = get_node_ip()
        self.log_path = log_path
        self.partition_id = partition_id
        self.log_filenames = set()
        self.open_file_infos = []
        self.closed_file_infos = []
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(driver_ip, driver_port))
        logger.info("Connected log server on {}:{}".format(driver_ip, driver_port))
        self.threads_stopped = threads_stopped
        self.log_fd = None

    def close_log_file(self):
        """Close open file"""
        if self.log_fd:
            self.log_fd.close()

    def check_and_open_log_file(self):
        if not self.log_fd:
            if os.path.exists(self.log_path):
                self.log_fd = open(self.log_path, "rb")

    def check_log_file_and_publish_updates(self):
        """Get any changes to the log file and push updates to zmq.

        Returns:
            True if anything was published and false otherwise.
        """
        anything_published = False
        if self.log_fd:
            assert not self.log_fd.closed

            lines_to_publish = []
            max_num_lines_to_read = 100
            for _ in range(max_num_lines_to_read):
                try:
                    next_line = self.log_fd.readline()
                    # Replace any characters not in UTF-8 with
                    # a replacement character, see
                    # https://stackoverflow.com/a/38565489/10891801
                    next_line = next_line.decode("utf-8", "replace")
                    if next_line == "":
                        break
                    if next_line[-1] == "\n":
                        next_line = next_line[:-1]
                    new_line = "[partition = {}, ip = {}] ".format(self.partition_id, self.ip) \
                                   + next_line
                    lines_to_publish.append(new_line)
                except Exception:
                    logger.error("Error: Reading file: {}, position: {} "
                                     "failed.".format(
                            self.log_path,
                            self.log_fd.tell()))
                    raise

            if len(lines_to_publish) > 0:
                message = "\n".join(lines_to_publish)
                self.socket.send_string(message)
                res = self.socket.recv().decode("utf-8")
                if res == "received":
                    anything_published = True

        return anything_published


    def run(self):
        try:
            while True:
                # Exit if we received a signal that we should stop.
                if self.threads_stopped.is_set():
                    return
                self.check_and_open_log_file()
                anything_published = self.check_log_file_and_publish_updates()
                # If nothing was published, then wait a little bit before checking
                # for logs to avoid using too much CPU.
                if not anything_published:
                    time.sleep(0.1)
        except Exception as e:
            self.socket.send_string(str(e))
            raise e
        finally:
            self.close_log_file()
