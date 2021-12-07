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

class LogFileInfo:
    def __init__(self,
                 filename=None,
                 executor_id=None,
                 size_when_last_opened=None,
                 file_position=None,
                 file_handle=None
                 ):
        assert (filename is not None and size_when_last_opened is not None
                and file_position is not None)
        self.filename = filename
        self.executor_id = executor_id
        self.size_when_last_opened = size_when_last_opened
        self.file_position = file_position
        self.file_handle = file_handle


class LogMonitor:
    """A monitor process for monitoring worker log files.

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
        print("Connected ZMQ server on {}:{}".format(driver_ip, driver_port))
        self.threads_stopped = threads_stopped
        self.log_fd = None

    def close_log_file(self):
        """Close all open files (so that we can open more)."""
        print("call close files")
        if self.log_fd:
            self.log_fd.close()

    def check_and_open_log_file(self):
        if not self.log_fd:
            if os.path.exists(self.log_path):
                self.log_fd = open(self.log_path, "rb")

    # def update_log_filenames(self):
    #     """Update the list of log files to monitor."""
    #     # output of user code is written here
    #     if os.path.exists(self.file_info_path):
    #         with open(self.file_info_path, "rb") as f:
    #             self.closed_file_infos = pickle.load(f)
    #     else:
    #         # log_file_paths = glob.glob("{}/**/stdout".format(self.logs_dir))\
    #         #                  + glob.glob("{}/**/stderr".format(self.logs_dir))
    #         log_file_paths = self.find_log_files()
    #         for file_path in log_file_paths:
    #             if os.path.isfile(file_path):
    #                 self.log_filenames.add(file_path)
    #                 executor_id = os.path.basename(os.path.dirname(file_path))
    #                 self.closed_file_infos.append(
    #                     LogFileInfo(
    #                         filename=file_path,
    #                         executor_id=executor_id,
    #                         size_when_last_opened=0,
    #                         file_position=0,
    #                         file_handle=None))
    #                 log_filename = os.path.basename(file_path)
    #                 logger.info("Beginning to track file {}".format(log_filename))

    def open_closed_files(self):
        while len(self.closed_file_infos) > 0:
            file_info = self.closed_file_infos.pop(0)
            assert file_info.file_handle is None
            try:
                f = open(file_info.filename, "rb")
            except (IOError, OSError) as e:
                if e.errno == errno.ENOENT:
                    logger.warning("Warning: The file {} was not "
                                       "found.".format(file_info.filename))
                    self.log_filenames.remove(file_info.filename)
                    continue
                else:
                    raise e

            f.seek(file_info.file_position)
            file_info.file_handle = f
            self.open_file_infos.append(file_info)

    # def check_log_files_and_publish_updates(self):
    #     """Get any changes to the log files and push updates to zmq.
    #
    #     Returns:
    #         True if anything was published and false otherwise.
    #     """
    #     anything_published = False
    #     for file_info in self.open_file_infos:
    #         assert not file_info.file_handle.closed
    #
    #         lines_to_publish = []
    #         max_num_lines_to_read = 100
    #         for _ in range(max_num_lines_to_read):
    #             try:
    #                 next_line = file_info.file_handle.readline()
    #                 # Replace any characters not in UTF-8 with
    #                 # a replacement character, see
    #                 # https://stackoverflow.com/a/38565489/10891801
    #                 next_line = next_line.decode("utf-8", "replace")
    #                 if next_line == "":
    #                     break
    #                 if next_line[-1] == "\n":
    #                     next_line = next_line[:-1]
    #                 new_line = "[executor_id = {}, ip = {}] ".format(file_info.executor_id, self.ip) \
    #                            + next_line
    #                 lines_to_publish.append(new_line)
    #             except Exception:
    #                 logger.error("Error: Reading file: {}, position: {} "
    #                              "failed.".format(
    #                     file_info.full_path,
    #                     file_info.file_info.file_handle.tell()))
    #                 raise
    #
    #         # Record the current position in the file.
    #         file_info.file_position = file_info.file_handle.tell()
    #
    #         if len(lines_to_publish) > 0:
    #             message = "\n".join(lines_to_publish)
    #             self.socket.send_string(message)
    #             res = self.socket.recv().decode("utf-8")
    #             if res == "received":
    #                 anything_published = True
    #
    #     return anything_published

    def check_log_file_and_publish_updates(self):
        """Get any changes to the log files and push updates to zmq.

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
