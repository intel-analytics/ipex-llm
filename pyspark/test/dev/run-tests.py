#!/usr/bin/env python

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

# Adopt from Spark and it might be refactored in the future


from __future__ import print_function

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from optparse import OptionParser
from os import path
from threading import Thread, Lock
from modules import all_modules  # noqa
if sys.version < '3':
    import Queue
else:
    import queue as Queue

if sys.version_info >= (2, 7):
    subprocess_check_output = subprocess.check_output
    subprocess_check_call = subprocess.check_call
else:
    raise Exception("only support version >= 2,7")


def is_exe(path):
    """
    Check if a given path is an executable file.
    From: http://stackoverflow.com/a/377028
    """
    return os.path.isfile(path) and os.access(path, os.X_OK)


def which(program):
    """
    Find and return the given program by its absolute path or 'None' if the program cannot be found.
    From: http://stackoverflow.com/a/377028
    """

    fpath = os.path.split(program)[0]

    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ.get("PATH").split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


current_dir = path.dirname(path.realpath(__file__))
bigdl_python_home = path.abspath(path.join(current_dir, '..'))
sys.path.append(bigdl_python_home)

SPARK_HOME = os.getenv('SPARK_HOME')

python_modules = dict((m.name, m) for m in all_modules if m.python_test_goals if m.name != 'root')


def print_red(text):
    print('\033[31m' + text + '\033[0m')


LOG_FILE = os.path.join(current_dir, "./unit-tests.log")
FAILURE_REPORTING_LOCK = Lock()
LOGGER = logging.getLogger()


def run_individual_python_test(test_name, python_exec):
    env = dict(os.environ)
    env.update({
        'DL_CORE_NUMBER': '4',
        'PYSPARK_PYTHON': python_exec
    })
    LOGGER.debug("Starting test(%s): %s", python_exec, test_name)
    start_time = time.time()
    try:
        per_test_output = tempfile.TemporaryFile()
        retcode = subprocess.Popen(
            [python_exec, "-m", test_name],
            stderr=per_test_output, stdout=per_test_output, env=env).wait()
    except:
        LOGGER.exception("Got exception while running %s with %s", test_name, python_exec)
        # Here, we use os._exit() instead of sys.exit() in order to force Python to exit even if
        # this code is invoked from a thread other than the main thread.
        os._exit(1)
    duration = time.time() - start_time
    # Exit on the first failure.
    if retcode != 0:
        try:
            with FAILURE_REPORTING_LOCK:
                with open(LOG_FILE, 'ab') as log_file:
                    per_test_output.seek(0)
                    log_file.writelines(per_test_output)
                per_test_output.seek(0)
                for line in per_test_output:
                    decoded_line = line.decode()
                    if not re.match('[0-9]+', decoded_line):
                        print(decoded_line, end='')
                per_test_output.close()
        except:
            LOGGER.exception("Got an exception while trying to print failed test output")
        finally:
            print_red("\nHad test failures in %s with %s; see logs." % (test_name, python_exec))
            # Here, we use os._exit() instead of sys.exit() in order to force Python to exit even if
            # this code is invoked from a thread other than the main thread.
            os._exit(-1)
    else:
        per_test_output.close()
        LOGGER.info("Finished test(%s): %s (%is)", python_exec, test_name, duration)


def get_default_python_executables():
    python_execs = ["python2.7", "python3.5"]
    return python_execs


def parse_opts():
    parser = OptionParser(
        prog="run-tests"
    )
    parser.add_option(
        "--python-executables", type="string", default=','.join(get_default_python_executables()),
        help="A comma-separated list of Python executables to test against (default: %default)"
    )
    parser.add_option(
        "--modules", type="string",
        default=",".join(sorted(python_modules.keys())),
        help="A comma-separated list of Python modules to test (default: %default)"
    )
    parser.add_option(
        "-p", "--parallelism", type="int", default=4,
        help="The number of suites to test in parallel (default %default)"
    )
    parser.add_option(
        "--verbose", action="store_true",
        help="Enable additional debug logging"
    )

    (opts, args) = parser.parse_args()
    if args:
        parser.error("Unsupported arguments: %s" % ' '.join(args))
    if opts.parallelism < 1:
        parser.error("Parallelism cannot be less than 1")
    return opts


def main():
    opts = parse_opts()
    if (opts.verbose):
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(stream=sys.stdout, level=log_level, format="%(message)s")
    LOGGER.info("Running BigDL python tests. Output is in %s", LOG_FILE)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    python_execs = opts.python_executables.split(',')
    modules_to_test = []
    for module_name in opts.modules.split(','):
        if module_name in python_modules:
            modules_to_test.append(python_modules[module_name])
        else:
            print("Error: unrecognized module '%s'. Supported modules: %s" %
                  (module_name, ", ".join(python_modules)))
            sys.exit(-1)
    LOGGER.info("Will test against the following Python executables: %s", python_execs)
    LOGGER.info("Will test the following Python modules: %s", [x.name for x in modules_to_test])

    task_queue = Queue.PriorityQueue()
    for python_exec in python_execs:
        LOGGER.debug("%s version is: %s", python_exec, subprocess_check_output(
            [python_exec, "--version"], stderr=subprocess.STDOUT, universal_newlines=True).strip())
        for module in modules_to_test:
            for test_goal in module.python_test_goals:
                if test_goal in ('nn.layer'):
                    priority = 0
                else:
                    priority = 100
                task_queue.put((priority, (python_exec, test_goal)))

    def process_queue(task_queue):
        while True:
            try:
                (priority, (python_exec, test_goal)) = task_queue.get_nowait()
            except Queue.Empty:
                break
            try:
                run_individual_python_test(test_goal, python_exec)
            finally:
                task_queue.task_done()

    start_time = time.time()
    for _ in range(opts.parallelism):
        worker = Thread(target=process_queue, args=(task_queue,))
        worker.daemon = True
        worker.start()
    try:
        task_queue.join()
    except (KeyboardInterrupt, SystemExit):
        print_red("Exiting due to interrupt")
        sys.exit(-1)
    total_duration = time.time() - start_time
    LOGGER.info("Tests passed in %i seconds", total_duration)


if __name__ == "__main__":
    main()
