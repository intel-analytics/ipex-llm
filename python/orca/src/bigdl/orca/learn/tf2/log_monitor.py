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

    This class mantains a list of open files and a list of closed log files. We
    can't simply leave all files open because we'll run out of file
    descriptors.

    The "run" method of this class will cycle between doing several things:
    1. First, it will check if any new files have appeared in the log
       directory. If so, they will be added to the list of closed files.
    2. Then, if we are unable to open any new files, we will close all of the
       files.
    3. Then, we will open as many closed files as we can that may have new
       lines (judged by an increase in file size since the last time the file
       was opened).
    4. Then we will loop through the open files and see if there are any new
       lines in the file. If so, we will publish them to Redis.

    Attributes:
        host (str): The hostname of this machine. Used to improve the log
            messages published to Redis.
        logs_dir (str): The directory that the log files are in.
        redis_client: A client used to communicate with the Redis server.
        log_filenames (set): This is the set of filenames of all files in
            open_file_infos and closed_file_infos.
        open_file_infos (list[LogFileInfo]): Info for all of the open files.
        closed_file_infos (list[LogFileInfo]): Info for all of the closed
            files.
        can_open_more_files (bool): True if we can still open more files and
            false otherwise.
    """

    def __init__(self, driver_ip, driver_port, logs_dir, threads_stopped, application_id):
        """Initialize the log monitor object."""
        from bigdl.dllib.utils.utils import get_node_ip
        self.ip = get_node_ip()
        self.logs_dir = logs_dir
        self.application_id = application_id
        self.log_filenames = set()
        self.open_file_infos = []
        self.closed_file_infos = []
        # self.can_open_more_files = True
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(driver_ip, driver_port))
        print("Connected ZMQ server on {}:{}".format(driver_ip, driver_port))
        self.file_info_path = os.path.join(tempfile.gettempdir(), "{}_file_info.pkl".format(application_id))
        self.threads_stopped = threads_stopped

    def close_all_files(self):
        """Close all open files (so that we can open more)."""
        print("call close files")
        while len(self.open_file_infos) > 0:
            file_info = self.open_file_infos.pop(0)
            file_info.file_handle.close()
            file_info.file_handle = None
            self.closed_file_infos.append(file_info)
        # save file info
        # if len(self.open_file_infos) > 0:
        with open(self.file_info_path, 'wb') as f:
            pickle.dump(self.closed_file_infos, f)

    def update_log_filenames(self):
        """Update the list of log files to monitor."""
        # output of user code is written here
        if os.path.exists(self.file_info_path):
            with open(self.file_info_path, "rb") as f:
                self.closed_file_infos = pickle.load(f)
        else:
            # log_file_paths = glob.glob("{}/**/stdout".format(self.logs_dir))\
            #                  + glob.glob("{}/**/stderr".format(self.logs_dir))
            log_file_paths = self.find_log_files()
            for file_path in log_file_paths:
                if os.path.isfile(file_path):
                    self.log_filenames.add(file_path)
                    executor_id = os.path.basename(os.path.dirname(file_path))
                    self.closed_file_infos.append(
                        LogFileInfo(
                            filename=file_path,
                            executor_id=executor_id,
                            size_when_last_opened=0,
                            file_position=0,
                            file_handle=None))
                    log_filename = os.path.basename(file_path)
                    logger.info("Beginning to track file {}".format(log_filename))

    def open_closed_files(self):
        """Open some closed files if they may have new lines.

        Opening more files may require us to close some of the already open
        files.
        """
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

    def check_log_files_and_publish_updates(self):
        """Get any changes to the log files and push updates to Redis.

        Returns:
            True if anything was published and false otherwise.
        """
        anything_published = False
        for file_info in self.open_file_infos:
            assert not file_info.file_handle.closed

            lines_to_publish = []
            max_num_lines_to_read = 100
            for _ in range(max_num_lines_to_read):
                try:
                    next_line = file_info.file_handle.readline()
                    # Replace any characters not in UTF-8 with
                    # a replacement character, see
                    # https://stackoverflow.com/a/38565489/10891801
                    next_line = next_line.decode("utf-8", "replace")
                    if next_line == "":
                        break
                    if next_line[-1] == "\n":
                        next_line = next_line[:-1]
                    new_line = "[executor_id = {}, ip = {}] ".format(file_info.executor_id, self.ip) \
                               + next_line
                    lines_to_publish.append(new_line)
                except Exception:
                    logger.error("Error: Reading file: {}, position: {} "
                                 "failed.".format(
                        file_info.full_path,
                        file_info.file_info.file_handle.tell()))
                    raise

            # Record the current position in the file.
            file_info.file_position = file_info.file_handle.tell()

            if len(lines_to_publish) > 0:
                message = "\n".join(lines_to_publish)
                self.socket.send_string(message)
                res = self.socket.recv().decode("utf-8")
                if res == "received":
                    anything_published = True

        return anything_published

    def run(self):
        """Run the log monitor.

        This will query Redis once every second to check if there are new log
        files to monitor. It will also store those log files in Redis.
        """
        try:
            self.update_log_filenames()
            self.open_closed_files()
            while True:
                # Exit if we received a signal that we should stop.
                if self.threads_stopped.is_set():
                    return
                # self.update_log_filenames()
                # self.open_closed_files()
                anything_published = self.check_log_files_and_publish_updates()
                # If nothing was published, then wait a little bit before checking
                # for logs to avoid using too much CPU.
                if not anything_published:
                    time.sleep(1)
        except Exception as e:
            self.socket.send_string(str(e))
            raise e
        finally:
            self.close_all_files()

    def find_log_files(self):
        file_list = []
        for root, dirs, files in os.walk(self.logs_dir):
            for file in files:
                if file == "stdout" or file == "stderr":
                    file_list.append(os.path.join(root, file))

        print("file list is: ", file_list)

        log_files = []
        pattern_str = '(.*){}/(.*)/[stderr/stdout]'.format(self.application_id)

        pattern_re = re.compile(pattern_str)

        for file_path in file_list:
            matched = pattern_re.match(file_path)
            if matched is not None:
                log_files.append(file_path)
        print("log files is: ", log_files)

        return log_files

