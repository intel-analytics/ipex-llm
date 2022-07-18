import contextlib
import logging
import os
import pathlib
import signal
import subprocess
import sys
import unittest

import graminelibos

fspath = getattr(os, 'fspath', str) # pylint: disable=invalid-name

# pylint: disable=subprocess-popen-preexec-fn,subprocess-run-check

HAS_SGX = os.environ.get('SGX') == '1'
ON_X86 = os.uname().machine in ['x86_64']
USES_MUSL = os.environ.get('GRAMINE_MUSL') == '1'

def expectedFailureIf(predicate):
    if predicate:
        return unittest.expectedFailure
    return lambda func: func


def run_command(cmd, *, timeout, can_fail=False, **kwds):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          preexec_fn=os.setsid, **kwds) as proc:
        timed_out = False
        try:
            raw_stdout, raw_stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True

        proc.poll()
        main_returncode = proc.returncode

        # Kill the whole process group: even if we did not time out, there might be some processes
        # remaining

        try:
            # after `setsid`, pgid should be the same as pid
            if proc.pid != os.getpgid(proc.pid):
                logging.warning(
                    'run_command: main process changed pgid, this might indicate an error and '
                    'prevent all processes from being cleaned up'
                )
        except ProcessLookupError:
            pass

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        if timed_out:
            raw_stdout, raw_stderr = proc.communicate()

        stdout = raw_stdout.decode(errors='surrogateescape')
        stderr = raw_stderr.decode(errors='surrogateescape')

        # Print stdout/stderr so that pytest can capture it
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)

        if timed_out:
            if main_returncode is not None:
                # XXX: Don't fail the test as long the main process exited (i.e. if it left dangling
                # child processes). This can happen due to a known issue with Gramine failing to
                # deliver a signal for an arbitrary amount of time. See the comment in
                # `shim_internal.h:handle_signal` for details.
                #
                # This happens occasionally when running LTP tests (e.g. `sendfile04`,
                # `fdatasync01`, `recvfrom01`, `sendto01`) that send SIGKILL to child processes.
                logging.warning(
                    'run_command: Command %s timed out, but the main process exited. This might be '
                    'due to a known issue with Gramine failing to deliver a signal. Continuing.',
                    cmd)
            else:
                raise AssertionError('Command {} timed out after {} s'.format(cmd, timeout))

        assert main_returncode is not None

        if main_returncode != 0 and not can_fail:
            raise subprocess.CalledProcessError(proc.returncode, cmd, raw_stdout, raw_stderr)

        return main_returncode, stdout, stderr


class RegressionTestCase(unittest.TestCase):
    DEFAULT_TIMEOUT = (20 if HAS_SGX else 10)

    def get_env(self, name):
        try:
            return os.environ[name]
        except KeyError:
            self.fail('environment variable {} not set'.format(name))

    @property
    def pal_path(self):
        # pylint: disable=protected-access
        return pathlib.Path(graminelibos._CONFIG_PKGLIBDIR) / ('sgx' if HAS_SGX else 'direct')

    @property
    def libpal_path(self):
        return self.pal_path / 'libpal.so'

    @property
    def loader_path(self):
        return self.pal_path / 'loader'

    def has_debug(self):
        p = subprocess.run(['objdump', '-x', fspath(self.libpal_path)],
            check=True, stdout=subprocess.PIPE)
        dump = p.stdout.decode()
        return '.debug_info' in dump

    def run_gdb(self, args, gdb_script, **kwds):
        prefix = ['gdb', '-q']
        env = os.environ.copy()
        if HAS_SGX:
            prefix += ['-x', fspath(self.pal_path / 'gdb_integration/gramine_sgx_gdb.py')]
            sgx_gdb = fspath(self.pal_path / 'gdb_integration/sgx_gdb.so')
            env['LD_PRELOAD'] = sgx_gdb + ':' + env.get('LD_PRELOAD', '')
        else:
            prefix += ['-x', fspath(self.pal_path / 'gdb_integration/gramine_linux_gdb.py')]

        # Override TTY, as apparently os.setpgrp() confuses GDB and causes it to hang.
        prefix += ['-x', gdb_script, '-batch', '-tty=/dev/null']
        prefix += ['--args']

        return self.run_binary(args, prefix=prefix, env=env, **kwds)

    def run_binary(self, args, *, timeout=None, prefix=None, **kwds):
        timeout = (max(self.DEFAULT_TIMEOUT, timeout) if timeout is not None
            else self.DEFAULT_TIMEOUT)

        if not self.loader_path.exists():
            self.fail('loader ({}) not found'.format(self.loader_path))
        if not self.libpal_path.exists():
            self.fail('libpal ({}) not found'.format(self.libpal_path))

        if prefix is None:
            prefix = []

        cmd = [*prefix, fspath(self.loader_path), fspath(self.libpal_path), 'init', *args]
        _returncode, stdout, stderr = run_command(cmd, timeout=timeout, **kwds)
        return stdout, stderr

    @classmethod
    def run_native_binary(cls, args, timeout=None, libpath=None, **kwds):
        timeout = (max(cls.DEFAULT_TIMEOUT, timeout) if timeout is not None
            else cls.DEFAULT_TIMEOUT)

        my_env = os.environ.copy()
        if not libpath is None:
            my_env["LD_LIBRARY_PATH"] = libpath

        _returncode, stdout, stderr = run_command(args, timeout=timeout, env=my_env, **kwds)
        return stdout, stderr

    @contextlib.contextmanager
    def expect_returncode(self, returncode):
        if returncode == 0:
            raise ValueError('expected returncode should be nonzero')
        try:
            yield
            self.fail('did not fail (expected {})'.format(returncode))
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, returncode,
                'failed with returncode {} (expected {})'.format(
                    e.returncode, returncode))
