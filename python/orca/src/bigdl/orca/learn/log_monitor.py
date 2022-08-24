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

import logging
import os
import time
from bigdl.dllib.utils.log4Error import *


logger = logging.getLogger(__name__)


class LogMonitor:
    """
    A monitor process for monitoring worker log files.

    """

    def __init__(self, driver_ip, driver_port, log_path, thread_stopped, partition_id):
        """Initialize the log monitor object."""
        from bigdl.dllib.utils.utils import get_node_ip
        self.ip = get_node_ip()
        self.log_path = log_path
        self.partition_id = partition_id
        import zmq
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(driver_ip, driver_port))
        logger.info("Connected log server on {}:{}".format(driver_ip, driver_port))
        self.thread_stopped = thread_stopped
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
        """
        Get any changes to the log file and push updates to zmq.

        Returns:
            True if anything was published and false otherwise.
        """
        anything_published = False
        if self.log_fd:
            invalidInputError(not self.log_fd.closed, "expect log is not closed here")

            lines_to_publish = []
            max_num_lines_to_read = 50
            for _ in range(max_num_lines_to_read):
                try:
                    next_line = self.log_fd.readline()
                    next_line = next_line.decode("utf-8", "replace")
                    if next_line == "":
                        break
                    if next_line[-1] == "\n":
                        next_line = next_line[:-1]
                    new_line = "[partition = {}, ip = {}] ".format(self.partition_id, self.ip) \
                               + next_line
                    lines_to_publish.append(new_line)
                except Exception:
                    msg = "Error: Reading file: {} at position: {} failed."\
                        .format(self.log_path, self.log_fd.tell())
                    invalidInputError(False, msg)

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
                if self.thread_stopped.is_set():
                    return
                self.check_and_open_log_file()
                anything_published = self.check_log_file_and_publish_updates()
                # If nothing was published, then wait a little bit to avoid using too much CPU.
                if not anything_published:
                    time.sleep(0.1)
        except Exception as e:
            self.socket.send_string(str(e))
            invalidInputError(False, str(e))
        finally:
            self.close_log_file()

    @staticmethod
    def start_log_monitor(driver_ip, driver_port, log_path, partition_id):
        def _start_log_monitor(driver_ip, driver_port, log_path, thread_stopped, partition_id):
            """
            Start a log monitor thread.

            """
            log_monitor = LogMonitor(driver_ip=driver_ip,
                                     driver_port=driver_port,
                                     log_path=log_path,
                                     thread_stopped=thread_stopped,
                                     partition_id=partition_id)
            log_monitor.run()
        import threading
        thread_stopped = threading.Event()
        logger_thread = threading.Thread(
            target=_start_log_monitor,
            args=(driver_ip, driver_port, log_path, thread_stopped, partition_id),
            name="monitor_logs")
        logger_thread.daemon = True
        logger_thread.start()
        return logger_thread, thread_stopped

    @staticmethod
    def stop_log_monitor(log_path, logger_thread, thread_stopped):
        thread_stopped.set()
        logger_thread.join()
        thread_stopped.clear()
        if os.path.exists(log_path):
            os.remove(log_path)


def start_log_server(ip, port):
    def _print_logs():
        """
        Prints log messages from workers on all of the nodes.

        """
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://{}:{}".format(ip, port))
        logger.info("started log server on {}:{}".format(ip, port))

        while True:
            message = socket.recv()
            print(message.decode("utf-8"))
            socket.send(b"received")

    import threading
    logger_thread = threading.Thread(
        target=_print_logs,
        name="print_logs")
    logger_thread.daemon = True
    logger_thread.start()
    return logger_thread


def stop_log_server(thread, ip, port):
    if thread.is_alive():
        import inspect
        import ctypes
        import zmq

        def _async_raise(tid, exctype):
            tid = ctypes.c_long(tid)
            if not inspect.isclass(exctype):
                exctype = type(exctype)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
            if res == 0:
                invalidInputError(False, "invalid thread id")
            elif res != 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                invalidOperationError(False, "PyThreadState_SetAsyncExc failed")

        def stop_thread(thread):
            _async_raise(thread.ident, SystemExit)

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://{}:{}".format(ip, port))
        socket.send_string("shutdown log server")
        stop_thread(thread)
