import argparse
import errno
import glob
import json
import logging
import os
import shutil
import time
import traceback
import subprocess

import redis

from bigdl.orca.learn.utils import format_error_message

logger = logging.getLogger(__name__)

LOG_FILE_CHANNEL = "SPARK_LOG_CHANNEL"


def create_redis_client(redis_address, password=None):
    """Create a Redis client.

    Args:
        The IP address, port, and password of the Redis server.

    Returns:
        A Redis client.
    """
    redis_ip_address, redis_port = redis_address.split(":")
    # For this command to work, some other client (on the same machine
    # as Redis) must have run "CONFIG SET protected-mode no".
    return redis.StrictRedis(
        host=redis_ip_address, port=int(redis_port), password=password)


class LogFileInfo:
    def __init__(self,
                 filename=None,
                 size_when_last_opened=None,
                 file_position=None,
                 file_handle=None):
        assert (filename is not None and size_when_last_opened is not None
                and file_position is not None)
        self.filename = filename
        self.size_when_last_opened = size_when_last_opened
        self.file_position = file_position
        self.file_handle = file_handle
        self.worker_pid = None


def open_file(self, data, path):
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        classpath = subprocess.Popen(["hadoop", "classpath", "--glob"],
                                     stdout=subprocess.PIPE).communicate()[0]
        os.environ["CLASSPATH"] = classpath.decode("utf-8")
        fs = pa.hdfs.connect()
        fd = fs.open(path, 'ab')
        return fd
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        fd = open(path, 'ab')
        return fd


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

    def __init__(self, logs_dir, sharing_log_dir):
        """Initialize the log monitor object."""
        from bigdl.dllib.utils.utils import get_node_ip
        self.ip = get_node_ip()
        self.logs_dir = logs_dir
        self.log_filenames = set()
        self.open_file_infos = []
        self.closed_file_infos = []
        self.can_open_more_files = True
        self.sharing_log_dir = sharing_log_dir
        self.fd = open_file()

    def close_all_files(self):
        """Close all open files (so that we can open more)."""
        while len(self.open_file_infos) > 0:
            file_info = self.open_file_infos.pop(0)
            file_info.file_handle.close()
            file_info.file_handle = None
            try:
                # Test if the worker process that generated the log file
                # is still alive. Only applies to worker processes.
                if file_info.worker_pid != "raylet":
                    os.kill(file_info.worker_pid, 0)
            except OSError:
                # The process is not alive any more, so move the log file
                # out of the log directory so glob.glob will not be slowed
                # by it.
                target = os.path.join(self.logs_dir, "old",
                                      os.path.basename(file_info.filename))
                try:
                    shutil.move(file_info.filename, target)
                except (IOError, OSError) as e:
                    if e.errno == errno.ENOENT:
                        logger.warning("Warning: The file {} was not "
                                       "found.".format(file_info.filename))
                    else:
                        raise e
            else:
                self.closed_file_infos.append(file_info)
        self.can_open_more_files = True

    def update_log_filenames(self):
        """Update the list of log files to monitor."""
        # output of user code is written here
        log_file_paths = glob.glob("{}/**/[stdout|stderr]".format(
            self.logs_dir))
        for file_path in log_file_paths:
            if os.path.isfile(
                    file_path) and file_path not in self.log_filenames:
                self.log_filenames.add(file_path)
                self.closed_file_infos.append(
                    LogFileInfo(
                        filename=file_path,
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
        # if not self.can_open_more_files:
        #     # If we can't open any more files. Close all of the files.
        #     self.close_all_files()

        files_with_no_updates = []
        while len(self.closed_file_infos) > 0:
            file_info = self.closed_file_infos.pop(0)
            assert file_info.file_handle is None
            # Get the file size to see if it has gotten bigger since we last
            # opened it.
            try:
                file_size = os.path.getsize(file_info.filename)
            except (IOError, OSError) as e:
                # Catch "file not found" errors.
                if e.errno == errno.ENOENT:
                    logger.warning("Warning: The file {} was not "
                                   "found.".format(file_info.filename))
                    self.log_filenames.remove(file_info.filename)
                    continue
                raise e

            # If some new lines have been added to this file, try to reopen the
            # file.
            if file_size > file_info.size_when_last_opened:
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
                file_info.filesize_when_last_opened = file_size
                file_info.file_handle = f
                self.open_file_infos.append(file_info)
            else:
                files_with_no_updates.append(file_info)

        # Add the files with no changes back to the list of closed files.
        self.closed_file_infos += files_with_no_updates

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
                    new_line = "[executor_id = %s, ip = %s]".format(file_info.worker_id, self.ip) \
                               + next_line + "\n"
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
                self.fd.write(lines_to_publish.encode("utf-8"))
                anything_published = True

        return anything_published

    def run(self):
        """Run the log monitor.

        This will query Redis once every second to check if there are new log
        files to monitor. It will also store those log files in Redis.
        """
        while True:
            self.update_log_filenames()
            self.open_closed_files()
            anything_published = self.check_log_files_and_publish_updates()
            # If nothing was published, then wait a little bit before checking
            # for logs to avoid using too much CPU.
            if not anything_published:
                time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("log monitor to connect "
                     "to."))
    parser.add_argument(
        "--logs_dir",
        required=True,
        type=str,
        help="Specify the path of the temporary directory used by Ray "
             "processes.")
    parser.add_argument(
        "--executor_id",
        required=True,
        type=int,
        help="Specify the executor id")
    parser.add_argument(
        "--target_file",
        required=True,
        type=str,
        help="Specify the sharing path of the file to be written.")

    args = parser.parse_args()
    log_monitor = LogMonitor(
        args.logs_dir, args.executor_id, args.target_file)

    try:
        log_monitor.run()
    except Exception as e:
        traceback_str = format_error_message(traceback.format_exc())
        message = ("The log monitor on executor {} failed with the following "
                   "error:\n{}".format(log_monitor.executor_id, traceback_str))
        raise Exception(message)
