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
import subprocess
import signal
import atexit
import sys
import psutil

from zoo.ray.util import gen_shutdown_per_node, is_local


class ProcessInfo(object):
    def __init__(self, out, err, errorcode, pgid, tag="default", pids=None, node_ip=None):
        self.out = str(out.strip())
        self.err = str(err.strip())
        self.pgid = pgid
        self.pids = pids
        self.errorcode = errorcode
        self.tag = tag
        self.master_addr = None
        self.node_ip = node_ip

    def __str__(self):
        return "node_ip: {} tag: {}, pgid: {}, pids: {}, returncode: {}, \
                master_addr: {},  \n {} {}".format(self.node_ip, self.tag, self.pgid,
                                                   self.pids,
                                                   self.errorcode,
                                                   self.master_addr,
                                                   self.out,
                                                   self.err)


def pids_from_gpid(gpid):
    processes = psutil.process_iter()
    result = []
    for proc in processes:
        try:
            if os.getpgid(proc.pid) == gpid:
                result.append(proc.pid)
        except Exception:
            pass
    return result


def session_execute(command, env=None, tag=None, fail_fast=False, timeout=120):
    pro = subprocess.Popen(
        command,
        shell=True,
        env=env,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    pgid = os.getpgid(pro.pid)
    out, err = pro.communicate(timeout=timeout)
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    print(out)
    print(err)
    errorcode = pro.returncode
    if errorcode != 0:
        if fail_fast:
            raise Exception(err)
        print(err)
    else:
        print(out)
    return ProcessInfo(out=out,
                       err=err,
                       errorcode=pro.returncode,
                       pgid=pgid,
                       pids=pids_from_gpid(pgid),
                       tag=tag)


class ProcessMonitor:
    def __init__(self, process_infos, sc, ray_rdd, raycontext, verbose=False):
        self.sc = sc
        self.raycontext = raycontext
        self.verbose = verbose
        self.ray_rdd = ray_rdd
        self.master = []
        self.slaves = []
        self.pgids = []
        self.node_ips = []
        self.process_infos = process_infos
        for process_info in process_infos:
            self.pgids.append(process_info.pgid)
            self.node_ips.append(process_info.node_ip)
            if process_info.master_addr:
                self.master.append(process_info)
            else:
                self.slaves.append(process_info)
        ProcessMonitor.register_shutdown_hook(extra_close_fn=self.clean_fn)
        assert len(self.master) == 1, \
            "We should got 1 master only, but we got {}".format(len(self.master))
        self.master = self.master[0]
        if not is_local(self.sc):
            self.print_ray_remote_err_out()

    def print_ray_remote_err_out(self):
        if self.master.errorcode != 0:
            raise Exception(str(self.master))
        for slave in self.slaves:
            if slave.errorcode != 0:
                raise Exception(str(slave))
        if self.verbose:
            print(self.master)
            for slave in self.slaves:
                print(slave)

    def clean_fn(self):
        if self.raycontext.stopped:
            return
        import ray
        ray.shutdown()
        if not self.sc:
            print("WARNING: SparkContext has been stopped before cleaning the Ray resources")
        if self.sc and (not is_local(self.sc)):
            self.ray_rdd.map(gen_shutdown_per_node(self.pgids, self.node_ips)).collect()
        else:
            gen_shutdown_per_node(self.pgids, self.node_ips)([])

    @staticmethod
    def register_shutdown_hook(pgid=None, extra_close_fn=None):
        def _shutdown():
            if pgid:
                gen_shutdown_per_node(pgid)(0)
            if extra_close_fn:
                extra_close_fn()

        def _signal_shutdown(_signo, _stack_frame):
            _shutdown()
            sys.exit(0)

        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)
