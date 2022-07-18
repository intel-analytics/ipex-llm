import os
import re
import shutil
import signal
import subprocess
import unittest

from graminelibos.regression import (
    HAS_SGX,
    ON_X86,
    USES_MUSL,
    RegressionTestCase,
)

class TC_00_Unittests(RegressionTestCase):
    def test_000_spinlock(self):
        stdout, _ = self.run_binary(['spinlock'], timeout=20)

        self.assertIn('Test successful!', stdout)

    def test_010_gramine_run_test(self):
        stdout, _ = self.run_binary(['run_test', 'pass'])
        self.assertIn('gramine_run_test("pass") = 0', stdout)

    @unittest.skipUnless(os.environ.get('UBSAN') == '1', 'test only enabled with UBSAN=1')
    def test_020_ubsan(self):
        self._test_abort('ubsan_int_overflow', ['ubsan: overflow'])

    def test_021_asan_heap(self):
        self._test_asan('heap', 'heap-buffer-overflow')

    def test_022_asan_stack(self):
        self._test_asan('stack', 'stack-buffer-overflow')

    def test_023_asan_stack(self):
        self._test_asan('global', 'global-buffer-overflow')

    @unittest.skipUnless(os.environ.get('ASAN') == '1', 'test only enabled with ASAN=1')
    def _test_asan(self, case, desc):
        expected_list = [f'asan: {desc}']
        if self.has_debug():
            expected_list.append(f'asan: location: run_test_asan_{case} at shim_call.c')
        self._test_abort(f'asan_{case}', expected_list)

    def _test_abort(self, test_name, expected_list):
        try:
            self.run_binary(['run_test', test_name])
            self.fail('run_test unexpectedly succeeded')
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()
            self.assertIn('run_test("{}") ...'.format(test_name), stderr,
                          'Gramine should not abort before attempting to run test')
            for expected in expected_list:
                self.assertIn(expected, stderr)
            self.assertNotIn('run_test("{}") ='.format(test_name), stderr,
                             'Gramine should abort before returning to application')

class TC_01_Bootstrap(RegressionTestCase):
    def test_001_helloworld(self):
        stdout, _ = self.run_binary(['helloworld'])
        self.assertIn('Hello world!', stdout)

    def test_100_basic_bootstrapping(self):
        stdout, _ = self.run_binary(['bootstrap'])

        # Basic Bootstrapping
        self.assertIn('User Program Started', stdout)

        # One Argument Given
        self.assertIn('# of arguments: 1', stdout)
        self.assertIn('argv[0] = bootstrap', stdout)

    def test_101_basic_bootstrapping_five_arguments(self):
        # Five Arguments Given
        stdout, _ = self.run_binary(['bootstrap', 'a', 'b', 'c', 'd'])
        self.assertIn('# of arguments: 5', stdout)
        self.assertIn('argv[0] = bootstrap', stdout)
        self.assertIn('argv[1] = a', stdout)
        self.assertIn('argv[2] = b', stdout)
        self.assertIn('argv[3] = c', stdout)
        self.assertIn('argv[4] = d', stdout)

    def test_102_argv_from_file(self):
        args = ['bootstrap', 'THIS', 'SHOULD GO', 'TO', '\nTHE\n', 'APP']
        result = subprocess.run(['gramine-argv-serializer'] + args,
                                stdout=subprocess.PIPE, check=True)
        with open('argv_test_input', 'wb') as f:
            f.write(result.stdout)
        try:
            stdout, _ = self.run_binary(['argv_from_file', 'WRONG', 'ARGUMENTS'])
            self.assertIn('# of arguments: %d\n' % len(args), stdout)
            for i, arg in enumerate(args):
                self.assertIn('argv[%d] = %s\n' % (i, arg), stdout)
        finally:
            os.remove('argv_test_input')

    def test_103_env_from_host(self):
        host_envs = {
            'A': '123',
            'PWD': '/some_dir',
            'some weir:d\nvar_name': ' even we\nirder\tvalue',
        }
        manifest_envs = {'LD_LIBRARY_PATH': '/lib'}
        stdout, _ = self.run_binary(['env_from_host'], env=host_envs)
        self.assertIn('# of envs: %d\n' % (len(host_envs) + len(manifest_envs)), stdout)
        for _, (key, val) in enumerate({**host_envs, **manifest_envs}.items()):
            # We don't enforce any specific order of envs, so we skip checking the index.
            self.assertIn('] = %s\n' % (key + '=' + val), stdout)

    def test_104_env_from_file(self):
        envs = ['A=123', 'PWD=/some_dir', 'some weir:d\nvar_name= even we\nirder\tvalue']
        manifest_envs = ['LD_LIBRARY_PATH=/lib']
        host_envs = {'THIS_SHOULDNT_BE_PASSED': '1234'}
        result = subprocess.run(['gramine-argv-serializer'] + envs,
                                stdout=subprocess.PIPE, check=True)
        with open('env_test_input', 'wb') as f:
            f.write(result.stdout)
        try:
            stdout, _ = self.run_binary(['env_from_file'], env=host_envs)
            self.assertIn('# of envs: %d\n' % (len(envs) + len(manifest_envs)), stdout)
            for _, arg in enumerate(envs + manifest_envs):
                # We don't enforce any specific order of envs, so we skip checking the index.
                self.assertIn('] = %s\n' % arg, stdout)
        finally:
            os.remove('env_test_input')

    def test_105_env_passthrough(self):
        host_envs = {
            'A': 'THIS_WILL_BE_PASSED',
            'B': 'THIS_WILL_BE_OVERWRITTEN',
            'C': 'THIS_SHOULDNT_BE_PASSED',
            'D': 'THIS_SHOULDNT_BE_PASSED_TOO',
        }
        manifest_envs = {'LD_LIBRARY_PATH': '/lib'}
        stdout, _ = self.run_binary(['env_passthrough'], env=host_envs)
        self.assertIn('# of envs: %d\n' % (len(host_envs) - 2 + len(manifest_envs)), stdout)

        # We don't enforce any specific order of envs, so we skip checking the index.
        self.assertIn('] = LD_LIBRARY_PATH=/lib\n', stdout)
        self.assertIn('] = A=THIS_WILL_BE_PASSED\n', stdout)
        self.assertIn('] = B=OVERWRITTEN_VALUE\n', stdout)
        self.assertNotIn('] = C=THIS_SHOULDNT_BE_PASSED\n', stdout)
        self.assertNotIn('] = D=THIS_SHOULDNT_BE_PASSED_TOO\n', stdout)

    def test_106_basic_bootstrapping_static(self):
        stdout, _ = self.run_binary(['bootstrap_static'])
        self.assertIn('Hello world (bootstrap_static)!', stdout)

    def test_107_basic_bootstrapping_pie(self):
        stdout, _ = self.run_binary(['bootstrap_pie'])
        self.assertIn('User program started', stdout)
        self.assertIn('Local Address in Executable: 0x', stdout)
        self.assertIn('argv[0] = bootstrap_pie', stdout)

    def test_108_uid_and_gid(self):
        stdout, _ = self.run_binary(['uid_gid'])
        self.assertIn('TEST OK', stdout)

    @unittest.skipUnless(ON_X86, 'x86-specific')
    @unittest.skipIf(USES_MUSL, 'C++ is not supported with musl')
    def test_110_basic_bootstrapping_cpp(self):
        stdout, _ = self.run_binary(['bootstrap_cpp'])
        self.assertIn('User Program Started', stdout)
        self.assertIn('Exception \'test runtime error\' caught', stdout)

    def test_200_exec(self):
        stdout, _ = self.run_binary(['exec'])

        # 2 page child binary
        self.assertIn(
            '0' * 89 + ' ' +
            ('0' * 93 + ' ') * 15,
            stdout)

    def test_201_exec_same(self):
        args = ['arg_#%d' % i for i in range(50)]
        stdout, _ = self.run_binary(['exec_same'] + args, timeout=40)
        for arg in args:
            self.assertIn(arg + '\n', stdout)

    def test_202_fork_and_exec(self):
        stdout, _ = self.run_binary(['fork_and_exec'], timeout=60)

        # fork and exec 2 page child binary
        self.assertIn('child exited with status: 0', stdout)
        self.assertIn('test completed successfully', stdout)

    def test_203_vfork_and_exec(self):
        stdout, _ = self.run_binary(['vfork_and_exec'], timeout=60)

        # vfork and exec 2 page child binary
        self.assertIn('child exited with status: 0', stdout)
        self.assertIn('test completed successfully', stdout)

    def test_204_exec_fork(self):
        stdout, _ = self.run_binary(['exec_fork'], timeout=60)
        self.assertNotIn('Handled SIGCHLD', stdout)
        self.assertIn('Set up handler for SIGCHLD', stdout)
        self.assertIn('child exited with status: 0', stdout)
        self.assertIn('test completed successfully', stdout)

    def test_205_double_fork(self):
        stdout, stderr = self.run_binary(['double_fork'])
        self.assertIn('TEST OK', stdout)
        self.assertNotIn('grandchild', stderr)

    def test_210_exec_invalid_args(self):
        stdout, _ = self.run_binary(['exec_invalid_args'])

        # Execve with invalid pointers in arguments
        self.assertIn('execve(invalid-path) correctly returned error', stdout)
        self.assertIn('execve(invalid-argv-ptr) correctly returned error', stdout)
        self.assertIn('execve(invalid-envp-ptr) correctly returned error', stdout)
        self.assertIn('execve(invalid-argv) correctly returned error', stdout)
        self.assertIn('execve(invalid-envp) correctly returned error', stdout)

    @unittest.skipIf(USES_MUSL, 'Test uses /bin/sh from the host which is built against Glibc')
    def test_211_exec_script(self):
        stdout, _ = self.run_binary(['exec_script'])
        self.assertIn('Printing Args: '
            'scripts/baz.sh ECHO FOXTROT GOLF scripts/bar.sh '
            'ALPHA BRAVO CHARLIE DELTA '
            'scripts/foo.sh STRING FROM EXECVE', stdout)

    def test_220_send_handle(self):
        path = 'tmp/send_handle_test'
        try:
            self._test_send_handle(path)
            self._test_send_handle(path, delete=True)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @unittest.skipUnless(HAS_SGX,
        'Protected files are only available with SGX')
    def test_221_send_handle_pf(self):
        path = 'tmp/pf/send_handle_test'
        os.makedirs('tmp/pf', exist_ok=True)
        # Delete the file: the test truncates the file anyway, but it may fail to open a malformed
        # protected file.
        if os.path.exists(path):
            os.unlink(path)
        try:
            self._test_send_handle(path)
            # TODO: Migrating a protected files handle is not supported when the file is deleted
            # self._test_send_handle(path, delete=True)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_222_send_handle_enc(self):
        path = 'tmp_enc/send_handle_test'
        os.makedirs('tmp_enc', exist_ok=True)
        # Delete the file: the test truncates the file anyway, but it may fail to open a malformed
        # encrypted file.
        if os.path.exists(path):
            os.unlink(path)
        try:
            self._test_send_handle(path)
            self._test_send_handle(path, delete=True)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_223_send_handle_tmpfs(self):
        path = '/mnt/tmpfs/send_handle_test'
        self._test_send_handle(path)
        self._test_send_handle(path, delete=True)

    def _test_send_handle(self, path, delete=False):
        if delete:
            cmd = ['send_handle', '-d', path]
        else:
            cmd = ['send_handle', path]

        stdout, _ = self.run_binary(cmd)
        self.assertIn('TEST OK', stdout, 'test failed: {}'.format(cmd))

    def test_230_keys(self):
        stdout, _ = self.run_binary(['keys'])
        self.assertIn('TEST OK', stdout)

    def test_300_shared_object(self):
        stdout, _ = self.run_binary(['shared_object'])

        # Shared Object
        self.assertIn('Hello world', stdout)

    def test_400_exit(self):
        with self.expect_returncode(113):
            self.run_binary(['exit'])

    def test_401_exit_group(self):
        for thread_idx in range(4):
            exit_code = 100 + thread_idx
            try:
                self.run_binary(['exit_group', str(thread_idx), str(exit_code)])
                self.fail('exit_group returned 0 instead of {}'.format(exit_code))
            except subprocess.CalledProcessError as e:
                self.assertEqual(e.returncode, exit_code)

    def test_402_signalexit(self):
        with self.expect_returncode(134):
            self.run_binary(['abort'])

    def test_403_signalexit_multithread(self):
        with self.expect_returncode(134):
            self.run_binary(['abort_multithread'])

    def test_404_sigprocmask_pending(self):
        stdout, _ = self.run_binary(['sigprocmask_pending'], timeout=60)
        self.assertIn('Child OK', stdout)
        self.assertIn('All tests OK', stdout)

    def test_500_init_fail(self):
        try:
            self.run_binary(['init_fail'])
            self.fail('expected to return nonzero (and != 42)')
        except subprocess.CalledProcessError as e:
            self.assertNotEqual(e.returncode, 42, 'expected returncode != 42')

    @unittest.skipUnless(HAS_SGX, 'This test relies on SGX-specific manifest options.')
    def test_501_init_fail2(self):
        try:
            self.run_binary(['init_fail2'], timeout=60)
            self.fail('expected to return nonzero (and != 42)')
        except subprocess.CalledProcessError as e:
            self.assertNotEqual(e.returncode, 42, 'expected returncode != 42')

    def test_600_multi_pthread(self):
        stdout, _ = self.run_binary(['multi_pthread'])

        # Multiple thread creation
        self.assertIn('256 Threads Created', stdout)

    @unittest.skipUnless(HAS_SGX, 'This test is only meaningful on SGX PAL')
    def test_601_multi_pthread_exitless(self):
        stdout, _ = self.run_binary(['multi_pthread_exitless'], timeout=60)

        # Multiple thread creation
        self.assertIn('256 Threads Created', stdout)

    def test_602_fp_multithread(self):
        stdout, _ = self.run_binary(['fp_multithread'])
        self.assertIn('FE_TONEAREST   child: 42.5 = 42.0, -42.5 = -42.0', stdout)
        self.assertIn('FE_TONEAREST  parent: 42.5 = 42.0, -42.5 = -42.0', stdout)
        self.assertIn('FE_UPWARD      child: 42.5 = 43.0, -42.5 = -42.0', stdout)
        self.assertIn('FE_UPWARD     parent: 42.5 = 43.0, -42.5 = -42.0', stdout)
        self.assertIn('FE_DOWNWARD    child: 42.5 = 42.0, -42.5 = -43.0', stdout)
        self.assertIn('FE_DOWNWARD   parent: 42.5 = 42.0, -42.5 = -43.0', stdout)
        self.assertIn('FE_TOWARDZERO  child: 42.5 = 42.0, -42.5 = -42.0', stdout)
        self.assertIn('FE_TOWARDZERO parent: 42.5 = 42.0, -42.5 = -42.0', stdout)

    def test_700_debug_log_inline(self):
        _, stderr = self.run_binary(['debug_log_inline'])
        self._verify_debug_log(stderr)

    def test_701_debug_log_file(self):
        log_path = 'tmp/debug_log_file.log'
        if os.path.exists(log_path):
            os.remove(log_path)

        self.run_binary(['debug_log_file'])

        with open(log_path) as log_file:
            log = log_file.read()

        self._verify_debug_log(log)

    def _verify_debug_log(self, log: str):
        self.assertIn('Host:', log)
        self.assertIn('Shim process initialized', log)
        self.assertIn('--- shim_exit_group', log)


class TC_02_OpenMP(RegressionTestCase):
    @unittest.skipIf(USES_MUSL, 'OpenMP is not supported with musl')
    def test_000_simple_for_loop(self):
        stdout, _ = self.run_binary(['openmp'])

        # OpenMP simple for loop
        self.assertIn('first: 0, last: 9', stdout)

@unittest.skipUnless(HAS_SGX,
    'This test is only meaningful on SGX PAL because file-check-policy is '
    'only relevant to SGX.')
class TC_03_FileCheckPolicy(RegressionTestCase):
    @classmethod
    def setUpClass(cls):
        with open('trusted_testfile', 'w') as f:
            f.write('trusted_testfile')

    @classmethod
    def tearDownClass(cls):
        os.remove('trusted_testfile')

    def test_000_strict_success(self):
        stdout, _ = self.run_binary(['file_check_policy_strict', 'read', 'trusted_testfile'])
        self.assertIn('file_check_policy succeeded', stdout)

    def test_001_strict_fail(self):
        try:
            self.run_binary(['file_check_policy_strict', 'read', 'unknown_testfile'])
            self.fail('expected to return nonzero')
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 2)
            stderr = e.stderr.decode()
            self.assertIn('Disallowing access to file \'unknown_testfile\'', stderr)

    def test_002_strict_fail_create(self):
        if os.path.exists('nonexisting_testfile'):
            os.remove('nonexisting_testfile')
        try:
            # this tests a previous bug in Gramine that allowed creating unknown files
            self.run_binary(['file_check_policy_strict', 'append', 'nonexisting_testfile'])
            self.fail('expected to return nonzero')
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 2)
            stderr = e.stderr.decode()
            self.assertIn('Disallowing access to file \'nonexisting_testfile\'', stderr)
            if os.path.exists('nonexisting_testfile'):
                self.fail('test created a file unexpectedly')

    def test_003_strict_fail_write(self):
        try:
            # writing to trusted files should not be possible
            self.run_binary(['file_check_policy_strict', 'write', 'trusted_testfile'])
            self.fail('expected to return nonzero')
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 2)
            stderr = e.stderr.decode()
            self.assertIn('Disallowing create/write/append to a trusted file \'trusted_testfile\'',
                          stderr)

    def test_004_allow_all_but_log_unknown(self):
        stdout, stderr = self.run_binary(['file_check_policy_allow_all_but_log', 'read',
                                          'unknown_testfile'])
        self.assertIn('Allowing access to unknown file \'unknown_testfile\' due to '
                      'file_check_policy settings.', stderr)
        self.assertIn('file_check_policy succeeded', stdout)

    def test_005_allow_all_but_log_trusted(self):
        stdout, stderr = self.run_binary(['file_check_policy_allow_all_but_log', 'read',
                                          'trusted_testfile'])
        self.assertNotIn('Allowing access to unknown file \'trusted_testfile\' due to '
                         'file_check_policy settings.', stderr)
        self.assertIn('file_check_policy succeeded', stdout)

    def test_006_allow_all_but_log_trusted_create_fail(self):
        try:
            # this fails because modifying trusted files is prohibited
            self.run_binary(['file_check_policy_allow_all_but_log', 'append', 'trusted_testfile'])
            self.fail('expected to return nonzero')
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 2)
            stderr = e.stderr.decode()
            self.assertIn('Disallowing create/write/append to a trusted file \'trusted_testfile\'',
                          stderr)

    def test_007_allow_all_but_log_unknown_create(self):
        if os.path.exists('nonexisting_testfile'):
            os.remove('nonexisting_testfile')
        try:
            stdout, stderr = self.run_binary(['file_check_policy_allow_all_but_log', 'append',
                                              'nonexisting_testfile'])
            self.assertIn('Allowing access to unknown file \'nonexisting_testfile\' due to '
                          'file_check_policy settings.', stderr)
            self.assertIn('file_check_policy succeeded', stdout)
            if not os.path.exists('nonexisting_testfile'):
                self.fail('test did not create a file')
        finally:
            os.remove('nonexisting_testfile')


@unittest.skipUnless(HAS_SGX,
    'These tests are only meaningful on SGX PAL because only SGX supports attestation.')
@unittest.skipIf(USES_MUSL,
    'These tests require custom build of mbedtls, which is cumbersome to do twice (musl and glibc)')
class TC_04_Attestation(RegressionTestCase):
    def test_000_attestation(self):
        stdout, _ = self.run_binary(['attestation'], timeout=60)
        self.assertIn("Test resource leaks in attestation filesystem... SUCCESS", stdout)
        self.assertIn("Test local attestation... SUCCESS", stdout)
        self.assertIn("Test quote interface... SUCCESS", stdout)

    def test_001_attestation_stdio(self):
        stdout, _ = self.run_binary(['attestation', 'test_stdio'], timeout=60)
        self.assertIn("Test resource leaks in attestation filesystem... SUCCESS", stdout)
        self.assertIn("Test local attestation... SUCCESS", stdout)
        self.assertIn("Test quote interface... SUCCESS", stdout)

class TC_30_Syscall(RegressionTestCase):
    def test_000_getcwd(self):
        stdout, _ = self.run_binary(['getcwd'])

        # Getcwd syscall
        self.assertIn('[bss_cwd_buf] getcwd succeeded: /', stdout)
        self.assertIn('[mmapped_cwd_buf] getcwd succeeded: /', stdout)

    def test_010_stat_invalid_args(self):
        stdout, _ = self.run_binary(['stat_invalid_args'])

        # Stat with invalid arguments
        self.assertIn('stat(invalid-path-ptr) correctly returned error', stdout)
        self.assertIn('stat(invalid-buf-ptr) correctly returned error', stdout)
        self.assertIn('lstat(invalid-path-ptr) correctly returned error', stdout)
        self.assertIn('lstat(invalid-buf-ptr) correctly returned error', stdout)

    def test_011_fstat_cwd(self):
        stdout, _ = self.run_binary(['fstat_cwd'])

        # fstat on a directory
        self.assertIn('fstat returned the fd type as S_IFDIR', stdout)

    def test_020_getdents(self):
        if os.path.exists("root"):
            shutil.rmtree("root")

        # This doesn't catch extraneous entries, but should be fine
        # until the LTP test can be run (need symlink support)

        stdout, _ = self.run_binary(['getdents'])
        self.assertIn('getdents: setup ok', stdout)

        # Directory listing (32-bit)
        self.assertIn('getdents32: . [0x4]', stdout)
        self.assertIn('getdents32: .. [0x4]', stdout)
        self.assertIn('getdents32: file1 [0x8]', stdout)
        self.assertIn('getdents32: file2 [0x8]', stdout)
        self.assertIn('getdents32: dir3 [0x4]', stdout)

        # Directory listing (64-bit)
        self.assertIn('getdents64: . [0x4]', stdout)
        self.assertIn('getdents64: .. [0x4]', stdout)
        self.assertIn('getdents64: file1 [0x8]', stdout)
        self.assertIn('getdents64: file2 [0x8]', stdout)
        self.assertIn('getdents64: dir3 [0x4]', stdout)

        # Directory listing across fork (we don't guarantee the exact names, just that there be at
        # least one of each)
        self.assertIn('getdents64 before fork:', stdout)
        self.assertIn('parent getdents64:', stdout)
        self.assertIn('child getdents64:', stdout)

    def test_021_getdents_large_dir(self):
        if os.path.exists("tmp/large_dir"):
            shutil.rmtree("tmp/large_dir")
        stdout, _ = self.run_binary(['large_dir_read', 'tmp/large_dir', '3000'])

        self.assertIn('Success!', stdout)

    def test_022_getdents_lseek(self):
        if os.path.exists("root"):
            shutil.rmtree("root")

        stdout, _ = self.run_binary(['getdents_lseek'])

        # First listing
        self.assertIn('getdents64 1: .', stdout)
        self.assertIn('getdents64 1: ..', stdout)
        self.assertIn('getdents64 1: file0', stdout)
        self.assertIn('getdents64 1: file1', stdout)
        self.assertIn('getdents64 1: file2', stdout)
        self.assertIn('getdents64 1: file3', stdout)
        self.assertIn('getdents64 1: file4', stdout)
        self.assertNotIn('getdents64 1: file5', stdout)

        # Second listing, after modifying directory and seeking back to 0
        self.assertIn('getdents64 2: .', stdout)
        self.assertIn('getdents64 2: ..', stdout)
        self.assertNotIn('getdents64 2: file0', stdout)
        self.assertIn('getdents64 2: file1', stdout)
        self.assertIn('getdents64 2: file2', stdout)
        self.assertIn('getdents64 2: file3', stdout)
        self.assertIn('getdents64 2: file4', stdout)
        self.assertIn('getdents64 2: file5', stdout)

    def test_023_readdir(self):
        stdout, _ = self.run_binary(['readdir'])
        self.assertIn('test completed successfully', stdout)

    def test_024_host_root_fs(self):
        stdout, _ = self.run_binary(['host_root_fs'])
        self.assertIn('Test was successful', stdout)

    def test_030_fopen(self):
        if os.path.exists("tmp/filecreatedbygramine"):
            os.remove("tmp/filecreatedbygramine")
        stdout, _ = self.run_binary(['fopen_cornercases'])

        # fopen corner cases
        self.assertIn('Successfully read from file: Hello World', stdout)

    def test_031_file_size(self):
        stdout, _ = self.run_binary(['file_size'])
        self.assertIn('test completed successfully', stdout)

    def test_032_large_file(self):
        try:
            stdout, _ = self.run_binary(['large_file'])
        finally:
            # This test generates a 4 GB file, don't leave it in FS.
            os.remove('tmp/large_file')

        self.assertIn('TEST OK', stdout)

    def test_033_rename_unlink_chroot(self):
        file1 = 'tmp/file1'
        file2 = 'tmp/file2'
        try:
            stdout, _ = self.run_binary(['rename_unlink', file1, file2])
        finally:
            for path in [file1, file2]:
                if os.path.exists(path):
                    os.unlink(path)
        self.assertIn('TEST OK', stdout)

    @unittest.skip('Protected files (as implemented in Linux-SGX PAL) do not support renaming yet')
    @unittest.skipUnless(HAS_SGX,
        'Protected files are only available with SGX')
    def test_034_rename_unlink_pf(self):
        os.makedirs('tmp/pf', exist_ok=True)
        file1 = 'tmp/pf/file1'
        file2 = 'tmp/pf/file2'
        try:
            stdout, _ = self.run_binary(['rename_unlink', file1, file2])
        finally:
            for path in [file1, file2]:
                if os.path.exists(path):
                    os.unlink(path)
        self.assertIn('TEST OK', stdout)

    def test_035_rename_unlink_enc(self):
        os.makedirs('tmp_enc', exist_ok=True)
        file1 = 'tmp_enc/file1'
        file2 = 'tmp_enc/file2'
        # Delete the files: the test overwrites them anyway, but it may fail if they are malformed.
        for path in [file1, file2]:
            if os.path.exists(path):
                os.unlink(path)
        try:
            stdout, _ = self.run_binary(['rename_unlink', file1, file2])
        finally:
            for path in [file1, file2]:
                if os.path.exists(path):
                    os.unlink(path)
        self.assertIn('TEST OK', stdout)

    def test_036_rename_unlink_tmpfs(self):
        file1 = '/mnt/tmpfs/file1'
        file2 = '/mnt/tmpfs/file2'
        stdout, _ = self.run_binary(['rename_unlink', file1, file2])
        self.assertIn('TEST OK', stdout)

    def test_040_futex_bitset(self):
        stdout, _ = self.run_binary(['futex_bitset'])

        # Futex Wake Test
        self.assertIn('Woke all kiddos', stdout)

    def test_041_futex_timeout(self):
        stdout, _ = self.run_binary(['futex_timeout'])

        # Futex Timeout Test
        self.assertIn('futex correctly timed out', stdout)

    def test_042_futex_requeue(self):
        stdout, _ = self.run_binary(['futex_requeue'])

        self.assertIn('Test successful!', stdout)

    def test_043_futex_wake_op(self):
        stdout, _ = self.run_binary(['futex_wake_op'])

        self.assertIn('Test successful!', stdout)

    def test_050_mmap(self):
        stdout, _ = self.run_binary(['mmap_file'], timeout=60)

        # Private mmap beyond file range
        self.assertIn('mmap test 6 passed', stdout)
        self.assertIn('mmap test 7 passed', stdout)

        # Private mmap beyond file range (after fork)
        self.assertIn('mmap test 1 passed', stdout)
        self.assertIn('mmap test 2 passed', stdout)
        self.assertIn('mmap test 3 passed', stdout)
        self.assertIn('mmap test 4 passed', stdout)

        # "test 5" and "test 8" are checked below, in test_051_mmap_sgx

    @unittest.skipIf(HAS_SGX,
        'On SGX, SIGBUS isn\'t always implemented correctly, for lack '
        'of memory protection. For now, some of these cases won\'t work.')
    def test_051_mmap_sgx(self):
        stdout, _ = self.run_binary(['mmap_file'], timeout=60)

        # SIGBUS test
        self.assertIn('mmap test 5 passed', stdout)
        self.assertIn('mmap test 8 passed', stdout)

    @unittest.skipUnless(HAS_SGX,
        'Trusted files are only available with SGX')
    def test_052_mmap_file_backed_trusted(self):
        stdout, _ = self.run_binary(['mmap_file_backed', 'mmap_file_backed'], timeout=60)
        self.assertIn('Child process done', stdout)
        self.assertIn('Parent process done', stdout)

    @unittest.skipUnless(HAS_SGX,
        'Protected files are only available with SGX')
    def test_053_mmap_file_backed_protected(self):
        # create the protected file
        pf_path = 'sealed_file_mrsigner.dat'
        if os.path.exists(pf_path):
            os.remove(pf_path)
        stdout, _ = self.run_binary(['sealed_file', pf_path])
        self.assertIn('CREATION OK', stdout)

        try:
            stdout, _ = self.run_binary(['mmap_file_backed', pf_path], timeout=60)
            self.assertIn('Child process done', stdout)
            self.assertIn('Parent process done', stdout)
        finally:
            os.remove(pf_path)

    def test_054_large_mmap(self):
        try:
            stdout, _ = self.run_binary(['large_mmap'], timeout=480)

            # Ftruncate
            self.assertIn('large_mmap: ftruncate OK', stdout)

            # Large mmap
            self.assertIn('large_mmap: mmap 1 completed OK', stdout)
            self.assertIn('large_mmap: mmap 2 completed OK', stdout)
        finally:
            # This test generates a 4 GB file, don't leave it in FS.
            os.remove('testfile')

    def test_055_mmap_emulated_tmpfs(self):
        path = '/mnt/tmpfs/test_mmap'
        stdout, _ = self.run_binary(['mmap_file_emulated', path])
        self.assertIn('TEST OK', stdout)

    def test_056_mmap_emulated_enc(self):
        path = 'tmp_enc/test_mmap'
        os.makedirs('tmp_enc', exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        try:
            stdout, _ = self.run_binary(['mmap_file_emulated', path])
            self.assertIn('TEST OK', stdout)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_057_mprotect_file_fork(self):
        stdout, _ = self.run_binary(['mprotect_file_fork'])

        self.assertIn('Test successful!', stdout)

    def test_058_mprotect_prot_growsdown(self):
        stdout, _ = self.run_binary(['mprotect_prot_growsdown'])

        self.assertIn('TEST OK', stdout)

    def test_059_madvise(self):
        stdout, _ = self.run_binary(['madvise'])
        self.assertIn('TEST OK', stdout)

    @unittest.skip('sigaltstack isn\'t correctly implemented')
    def test_060_sigaltstack(self):
        stdout, _ = self.run_binary(['sigaltstack'])

        # Sigaltstack Test
        self.assertIn('OK on sigaltstack in main thread before alarm', stdout)
        self.assertIn('&act == 0x', stdout)
        self.assertIn('sig %d count 1 goes off with sp=0x' % signal.SIGALRM, stdout)
        self.assertIn('OK on signal stack', stdout)
        self.assertIn('OK on sigaltstack in handler', stdout)
        self.assertIn('sig %d count 2 goes off with sp=0x' % signal.SIGALRM, stdout)
        self.assertIn('OK on signal stack', stdout)
        self.assertIn('OK on sigaltstack in handler', stdout)
        self.assertIn('sig %d count 3 goes off with sp=0x' % signal.SIGALRM, stdout)
        self.assertIn('OK on signal stack', stdout)
        self.assertIn('OK on sigaltstack in handler', stdout)
        self.assertIn('OK on sigaltstack in main thread', stdout)
        self.assertIn('done exiting', stdout)

    def test_070_eventfd(self):
        stdout, _ = self.run_binary(['eventfd'])

        # Eventfd Test
        self.assertIn('eventfd_using_poll completed successfully', stdout)
        self.assertIn('eventfd_using_various_flags completed successfully', stdout)
        self.assertIn('eventfd_using_fork completed successfully', stdout)

    @unittest.skipIf(USES_MUSL, 'sched_setscheduler is not supported in musl')
    def test_080_sched(self):
        stdout, _ = self.run_binary(['sched'])

        # Scheduling Syscalls Test
        self.assertIn('Test completed successfully', stdout)

    def test_090_sighandler_reset(self):
        stdout, _ = self.run_binary(['sighandler_reset'])
        self.assertIn('Got signal %d' % signal.SIGCHLD, stdout)
        self.assertIn('Handler was invoked 1 time(s).', stdout)

    def test_091_sigaction_per_process(self):
        stdout, _ = self.run_binary(['sigaction_per_process'])
        self.assertIn('TEST OK', stdout)

    def test_092_sighandler_sigpipe(self):
        try:
            self.run_binary(['sighandler_sigpipe'])
            self.fail('expected to return nonzero')
        except subprocess.CalledProcessError as e:
            # FIXME: It's unclear what Gramine process should return when the app
            # inside dies due to a signal.
            self.assertTrue(e.returncode in [signal.SIGPIPE, 128 + signal.SIGPIPE])
            stdout = e.stdout.decode()
            self.assertIn('Got signal %d' % signal.SIGPIPE, stdout)
            self.assertIn('Got 1 SIGPIPE signal(s)', stdout)
            self.assertIn('Could not write to pipe: Broken pipe', stdout)

    @unittest.skipUnless(ON_X86, "x86-specific")
    def test_093_sighandler_divbyzero(self):
        stdout, _ = self.run_binary(['sighandler_divbyzero'])
        self.assertIn('Got signal %d' % signal.SIGFPE, stdout)
        self.assertIn('Got 1 SIGFPE signal(s)', stdout)
        self.assertIn('TEST OK', stdout)

    def test_094_signal_multithread(self):
        stdout, _ = self.run_binary(['signal_multithread'])
        self.assertIn('TEST OK', stdout)

    def test_095_kill_all(self):
        stdout, _ = self.run_binary(['kill_all'])
        self.assertIn('TEST OK', stdout)

    def test_100_get_set_groups(self):
        stdout, _ = self.run_binary(['groups'])
        self.assertIn('child OK', stdout)
        self.assertIn('parent OK', stdout)

    def test_101_sched_set_get_cpuaffinity(self):
        stdout, _ = self.run_binary(['sched_set_get_affinity'])
        self.assertIn('TEST OK', stdout)

    def test_102_pthread_set_get_affinity(self):
        stdout, _ = self.run_binary(['pthread_set_get_affinity', '1000'])
        self.assertIn('TEST OK', stdout)

    def test_103_gettimeofday(self):
        stdout, _ = self.run_binary(['gettimeofday'])
        self.assertIn('TEST OK', stdout)

    def test_110_fcntl_lock(self):
        try:
            stdout, _ = self.run_binary(['fcntl_lock'])
        finally:
            if os.path.exists('tmp/lock_file'):
                os.remove('tmp/lock_file')
        self.assertIn('TEST OK', stdout)

class TC_31_Syscall(RegressionTestCase):
    def test_000_syscall_redirect(self):
        stdout, _ = self.run_binary(['syscall'])
        self.assertIn('TEST OK', stdout)

    def test_010_syscall_restart(self):
        stdout, _ = self.run_binary(['syscall_restart'])
        self.assertIn('Got: R', stdout)
        self.assertIn('TEST 1 OK', stdout)
        self.assertIn('Handling signal 15', stdout)
        self.assertIn('Got: P', stdout)
        self.assertIn('TEST 2 OK', stdout)

class TC_40_FileSystem(RegressionTestCase):
    def test_000_proc(self):
        stdout, _ = self.run_binary(['proc_common'])
        lines = stdout.splitlines()

        self.assertIn('/proc/meminfo: file', lines)
        self.assertIn('/proc/cpuinfo: file', lines)
        self.assertIn('/proc/stat: file', lines)

        # /proc/self, /proc/[pid]
        self.assertIn('/proc/self: link: 2', lines)
        self.assertIn('/proc/2: directory', lines)
        self.assertIn('/proc/2/cwd: link: /', lines)
        self.assertIn('/proc/2/exe: link: /proc_common', lines)
        self.assertIn('/proc/2/root: link: /', lines)
        self.assertIn('/proc/2/maps: file', lines)

        # /proc/[pid]/fd
        self.assertIn('/proc/2/fd/0: link: /dev/tty', lines)
        self.assertIn('/proc/2/fd/1: link: /dev/tty', lines)
        self.assertIn('/proc/2/fd/2: link: /dev/tty', lines)
        self.assertIn('/proc/2/fd/3: link: pipe:[?]', lines)
        self.assertIn('/proc/2/fd/4: link: pipe:[?]', lines)

        # /proc/[pid]/task/[tid]
        self.assertIn('/proc/2/task/2: directory', lines)
        self.assertIn('/proc/2/task/33: directory', lines)
        self.assertIn('/proc/2/task/33/cwd: link: /', lines)
        self.assertIn('/proc/2/task/33/exe: link: /proc_common', lines)
        self.assertIn('/proc/2/task/33/root: link: /', lines)
        self.assertIn('/proc/2/task/33/fd/0: link: /dev/tty', lines)
        self.assertIn('/proc/2/task/33/fd/1: link: /dev/tty', lines)
        self.assertIn('/proc/2/task/33/fd/2: link: /dev/tty', lines)

        # /proc/[ipc-pid]/*
        self.assertIn('/proc/1/cwd: link: /', lines)
        self.assertIn('/proc/1/exe: link: /proc_common', lines)
        self.assertIn('/proc/1/root: link: /', lines)

    def test_001_devfs(self):
        stdout, _ = self.run_binary(['devfs'])
        self.assertIn('/dev/.', stdout)
        self.assertIn('/dev/null', stdout)
        self.assertIn('/dev/zero', stdout)
        self.assertIn('/dev/random', stdout)
        self.assertIn('/dev/urandom', stdout)
        self.assertIn('/dev/stdin', stdout)
        self.assertIn('/dev/stdout', stdout)
        self.assertIn('/dev/stderr', stdout)
        self.assertIn('Four bytes from /dev/urandom', stdout)
        self.assertIn('TEST OK', stdout)

    def test_002_device_passthrough(self):
        stdout, _ = self.run_binary(['device_passthrough'])
        self.assertIn('TEST OK', stdout)

    def test_010_path(self):
        stdout, _ = self.run_binary(['proc_path'])
        self.assertIn('proc path test success', stdout)

    def test_020_cpuinfo(self):
        stdout, _ = self.run_binary(['proc_cpuinfo'])

        # proc/cpuinfo Linux-based formatting
        self.assertIn('cpuinfo test passed', stdout)

    def test_021_procstat(self):
        stdout, _ = self.run_binary(['proc_stat'])

        # proc/stat Linux-based formatting
        self.assertIn('/proc/stat test passed', stdout)

    def test_030_fdleak(self):
        stdout, _ = self.run_binary(['fdleak'], timeout=10)
        self.assertIn("Test succeeded.", stdout)

    def get_cache_levels_cnt(self):
        cpu0 = '/sys/devices/system/cpu/cpu0/'
        self.assertTrue(os.path.exists(f'{cpu0}/cache/'))

        n = 0
        while os.path.exists(f'{cpu0}/cache/index{n}'):
            n += 1

        self.assertGreater(n, 0, "no information about CPU cache found")
        return n

    def test_040_sysfs(self):
        cpus_cnt = os.cpu_count()
        cache_levels_cnt = self.get_cache_levels_cnt()

        stdout, _ = self.run_binary(['sysfs_common'])
        lines = stdout.splitlines()

        self.assertIn('/sys/devices/system/cpu: directory', lines)
        for i in range(cpus_cnt):
            cpu = f'/sys/devices/system/cpu/cpu{i}'
            self.assertIn(f'{cpu}: directory', lines)
            if i == 0:
                self.assertNotIn(f'{cpu}/online: file', lines)
            else:
                self.assertIn(f'{cpu}/online: file', lines)

            self.assertIn(f'{cpu}/topology/core_id: file', lines)
            self.assertIn(f'{cpu}/topology/physical_package_id: file', lines)
            self.assertIn(f'{cpu}/topology/core_siblings: file', lines)
            self.assertIn(f'{cpu}/topology/thread_siblings: file', lines)

            for j in range(cache_levels_cnt):
                cache = f'{cpu}/cache/index{j}'
                self.assertIn(f'{cache}: directory', lines)
                self.assertIn(f'{cache}/shared_cpu_map: file', lines)
                self.assertIn(f'{cache}/level: file', lines)
                self.assertIn(f'{cache}/type: file', lines)
                self.assertIn(f'{cache}/size: file', lines)
                self.assertIn(f'{cache}/coherency_line_size: file', lines)
                self.assertIn(f'{cache}/number_of_sets: file', lines)
                self.assertIn(f'{cache}/physical_line_partition: file', lines)

        self.assertIn('/sys/devices/system/node: directory', lines)
        nodes_cnt = len([line for line in lines
                         if re.match(r'/sys/devices/system/node/node[0-9]+:', line)])
        self.assertGreater(nodes_cnt, 0)
        for i in range(nodes_cnt):
            node = f'/sys/devices/system/node/node{i}'
            self.assertIn(f'{node}: directory', lines)
            self.assertIn(f'{node}/cpumap: file', lines)
            self.assertIn(f'{node}/distance: file', lines)
            self.assertIn(f'{node}/hugepages/hugepages-2048kB/nr_hugepages: file', lines)
            self.assertIn(f'{node}/hugepages/hugepages-1048576kB/nr_hugepages: file', lines)

    @unittest.skipUnless(HAS_SGX, 'Sealed (protected) files are only available with SGX')
    def test_050_sealed_file_mrenclave(self):
        # Test both old and new implementation
        for pf_path in ['sealed_file_mrenclave.dat', 'encrypted_file_mrenclave.dat']:
            if os.path.exists(pf_path):
                os.remove(pf_path)

            stdout, _ = self.run_binary(['sealed_file', pf_path])
            self.assertIn('CREATION OK', stdout)
            stdout, _ = self.run_binary(['sealed_file', pf_path])
            self.assertIn('READING OK', stdout)

    @unittest.skipUnless(HAS_SGX, 'Sealed (protected) files are only available with SGX')
    def test_051_sealed_file_mrsigner(self):
        # Test both old and new implementation
        for pf_path in ['sealed_file_mrsigner.dat', 'encrypted_file_mrsigner.dat']:
            if os.path.exists(pf_path):
                os.remove(pf_path)

            stdout, _ = self.run_binary(['sealed_file', pf_path])
            self.assertIn('CREATION OK', stdout)
            stdout, _ = self.run_binary(['sealed_file_mod', pf_path])
            self.assertIn('READING FROM MODIFIED ENCLAVE OK', stdout)

    @unittest.skipUnless(HAS_SGX, 'Sealed (protected) files are only available with SGX')
    def test_052_sealed_file_mrenclave_bad(self):
        # Test both old and new implementation
        for pf_path in ['sealed_file_mrenclave.dat', 'encrypted_file_mrenclave.dat']:
            # Negative test: Seal MRENCLAVE-bound file in one enclave -> opening this file in
            # another enclave (with different MRENCLAVE) should fail
            if os.path.exists(pf_path):
                os.remove(pf_path)

            stdout, _ = self.run_binary(['sealed_file', pf_path])
            self.assertIn('CREATION OK', stdout)

            try:
                self.run_binary(['sealed_file_mod', pf_path])
                self.fail('expected to return nonzero')
            except subprocess.CalledProcessError as e:
                self.assertEqual(e.returncode, 1)
                stdout = e.stdout.decode()
                self.assertNotIn('READING FROM MODIFIED ENCLAVE OK', stdout)
                self.assertIn('Permission denied', stdout)

    def test_060_synthetic(self):
        stdout, _ = self.run_binary(['synthetic'])
        self.assertIn("TEST OK", stdout)


class TC_50_GDB(RegressionTestCase):
    def setUp(self):
        if not self.has_debug():
            self.skipTest('test runs only when Gramine is compiled with DEBUG=1')

    def find(self, name, stdout):
        match = re.search('<{0} start>(.*)<{0} end>'.format(name), stdout, re.DOTALL)
        self.assertTrue(match, '{} not found in GDB output'.format(name))
        return match.group(1).strip()

    def test_000_gdb_backtrace(self):
        # pylint: disable=fixme
        #
        # To run this test manually, use:
        # GDB=1 GDB_SCRIPT=debug.gdb gramine-{direct|sgx} debug
        #
        # TODO: strengthen this test after SGX includes enclave entry.
        #
        # While the stack trace in SGX is unbroken, it currently starts at _start inside
        # enclave, instead of including eclave entry.

        stdout, _ = self.run_gdb(['debug'], 'debug.gdb')

        backtrace_1 = self.find('backtrace 1', stdout)
        self.assertIn(f' main ()', backtrace_1)
        self.assertIn(' _start ()', backtrace_1)
        self.assertIn('debug.c', backtrace_1)
        if not USES_MUSL:
            self.assertNotIn('??', backtrace_1)

        backtrace_2 = self.find('backtrace 2', stdout)
        self.assertIn(' dev_write (', backtrace_2)
        self.assertIn(' func ()', backtrace_2)
        self.assertIn(' main ()', backtrace_2)
        self.assertIn(' _start ()', backtrace_2)
        self.assertIn('debug.c', backtrace_2)
        if not USES_MUSL:
            self.assertNotIn('??', backtrace_2)

        if HAS_SGX:
            backtrace_3 = self.find('backtrace 3', stdout)
            self.assertIn(' sgx_ocall_write (', backtrace_3)
            self.assertIn(' dev_write (', backtrace_3)
            self.assertIn(' func ()', backtrace_3)
            self.assertIn(' main ()', backtrace_3)
            self.assertIn(' _start ()', backtrace_3)
            self.assertIn('debug.c', backtrace_3)
            if not USES_MUSL:
                self.assertNotIn('??', backtrace_3)

    @unittest.skipUnless(ON_X86, 'x86-specific')
    def test_010_regs_x86_64(self):
        # To run this test manually, use:
        # GDB=1 GDB_SCRIPT=debug_regs-x86_64.gdb gramine-{direct|sgx} debug_regs-x86_64

        stdout, _ = self.run_gdb(['debug_regs-x86_64'], 'debug_regs-x86_64.gdb')

        rdx = self.find('RDX', stdout)
        self.assertEqual(rdx, '$1 = 0x1000100010001000')

        rdx_result = self.find('RDX result', stdout)
        self.assertEqual(rdx_result, '$2 = 0x2000200020002000')

        xmm0 = self.find('XMM0', stdout)
        self.assertEqual(xmm0, '$3 = 0x30003000300030003000300030003000')

        xmm0_result = self.find('XMM0 result', stdout)
        self.assertEqual(xmm0_result, '$4 = 0x4000400040004000')


class TC_80_Socket(RegressionTestCase):
    def test_000_getsockopt(self):
        stdout, _ = self.run_binary(['getsockopt'])
        self.assertIn('getsockopt: Got socket type OK', stdout)
        self.assertIn('getsockopt: Got TCP_NODELAY flag OK', stdout)

    def test_010_epoll(self):
        stdout, _ = self.run_binary(['epoll_test'])
        self.assertIn('TEST OK', stdout)

    def test_011_epoll_epollet(self):
        stdout, _ = self.run_binary(['epoll_epollet'])
        self.assertIn('TEST OK', stdout)

    def test_020_poll(self):
        try:
            stdout, _ = self.run_binary(['poll'])
        finally:
            if os.path.exists("tmp/host_file"):
                os.remove("tmp/host_file")
        self.assertIn('TEST OK', stdout)

    def test_021_poll_many_types(self):
        stdout, _ = self.run_binary(['poll_many_types'])
        self.assertIn('poll(POLLIN) returned 3 file descriptors', stdout)

    def test_022_poll_closed_fd(self):
        stdout, _ = self.run_binary(['poll_closed_fd'], timeout=60)
        self.assertNotIn('poll with POLLIN failed', stdout)
        self.assertIn('read on pipe: Hello from write end of pipe!', stdout)
        self.assertIn('the peer closed its end of the pipe', stdout)

    def test_030_ppoll(self):
        stdout, _ = self.run_binary(['ppoll'])
        self.assertIn('ppoll(POLLOUT) returned 1 file descriptors', stdout)
        self.assertIn('ppoll(POLLIN) returned 1 file descriptors', stdout)

    def test_040_select(self):
        stdout, _ = self.run_binary(['select'])
        self.assertIn('select() on write event returned 1 file descriptors', stdout)
        self.assertIn('select() on read event returned 1 file descriptors', stdout)

    def test_050_pselect(self):
        stdout, _ = self.run_binary(['pselect'])
        self.assertIn('pselect() on write event returned 1 file descriptors', stdout)
        self.assertIn('pselect() on read event returned 1 file descriptors', stdout)

    def test_060_getsockname(self):
        stdout, _ = self.run_binary(['getsockname'])
        self.assertIn('getsockname: Got socket name with static port OK', stdout)
        self.assertIn('getsockname: Got socket name with arbitrary port OK', stdout)

    def test_090_pipe(self):
        stdout, _ = self.run_binary(['pipe'], timeout=60)
        self.assertIn('read on pipe: Hello from write end of pipe!', stdout)

    def test_091_pipe_nonblocking(self):
        stdout, _ = self.run_binary(['pipe_nonblocking'])
        self.assertIn('TEST OK', stdout)

    def test_092_pipe_ocloexec(self):
        stdout, _ = self.run_binary(['pipe_ocloexec'])
        self.assertIn('TEST OK', stdout)

    def test_095_mkfifo(self):
        try:
            stdout, _ = self.run_binary(['mkfifo'], timeout=60)
        finally:
            if os.path.exists('tmp/fifo'):
                os.remove('tmp/fifo')
        self.assertIn('read on FIFO: Hello from write end of FIFO!', stdout)
        self.assertIn('[parent] TEST OK', stdout)

    def test_100_socket_unix(self):
        if os.path.exists("dummy"):
            os.remove("dummy")
        if os.path.exists("u"):
            os.remove("u")

        stdout, _ = self.run_binary(['unix'])
        self.assertIn('Data: This is packet 0', stdout)
        self.assertIn('Data: This is packet 1', stdout)
        self.assertIn('Data: This is packet 2', stdout)
        self.assertIn('Data: This is packet 3', stdout)
        self.assertIn('Data: This is packet 4', stdout)
        self.assertIn('Data: This is packet 5', stdout)
        self.assertIn('Data: This is packet 6', stdout)
        self.assertIn('Data: This is packet 7', stdout)
        self.assertIn('Data: This is packet 8', stdout)
        self.assertIn('Data: This is packet 9', stdout)

    def test_200_socket_udp(self):
        stdout, _ = self.run_binary(['udp'], timeout=50)
        self.assertIn('This is packet 0', stdout)
        self.assertIn('This is packet 1', stdout)
        self.assertIn('This is packet 2', stdout)
        self.assertIn('This is packet 3', stdout)
        self.assertIn('This is packet 4', stdout)
        self.assertIn('This is packet 5', stdout)
        self.assertIn('This is packet 6', stdout)
        self.assertIn('This is packet 7', stdout)
        self.assertIn('This is packet 8', stdout)
        self.assertIn('This is packet 9', stdout)

    def test_300_socket_tcp_msg_peek(self):
        stdout, _ = self.run_binary(['tcp_msg_peek'], timeout=50)
        self.assertIn('[client] receiving with MSG_PEEK: Hello from server!', stdout)
        self.assertIn('[client] receiving with MSG_PEEK again: Hello from server!', stdout)
        self.assertIn('[client] receiving without MSG_PEEK: Hello from server!', stdout)
        self.assertIn('[client] checking how many bytes are left unread: 0', stdout)
        self.assertIn('[client] done', stdout)
        self.assertIn('[server] done', stdout)

    def test_310_socket_tcp_ipv6_v6only(self):
        stdout, _ = self.run_binary(['tcp_ipv6_v6only'], timeout=50)
        self.assertIn('test completed successfully', stdout)

    @unittest.skipIf(USES_MUSL, 'C++ is not supported with musl')
    def test_320_ioctl(self):
        stdout, _ = self.run_binary(['ioctl'])
        self.assertIn('Ip Address 127.0.0.1', stdout)
        self.assertIn('Device lo -> Ethernet 00:00:00:00:00:00', stdout)


@unittest.skipUnless(HAS_SGX,
    'This test is only meaningful on SGX PAL because only SGX emulates CPUID.')
class TC_90_CpuidSGX(RegressionTestCase):
    def test_000_cpuid(self):
        stdout, _ = self.run_binary(['cpuid'])
        self.assertIn('CPUID test passed.', stdout)

# note that `rdtsc` also correctly runs on non-SGX PAL, but non-SGX CPU may not have rdtscp
@unittest.skipUnless(HAS_SGX,
    'This test is only meaningful on SGX PAL because only SGX emulates RDTSC/RDTSCP.')
class TC_91_RdtscSGX(RegressionTestCase):
    def test_000_rdtsc(self):
        stdout, _ = self.run_binary(['rdtsc'])
        self.assertIn('TEST OK', stdout)
