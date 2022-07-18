#!/usr/bin/python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2019 Wojtek Porczyk <woju@invisiblethingslab.com>
# Copyright (C) 2021 Intel Corporation
#                    Paweł Marczewski <pawel@invisiblethingslab.com>

import configparser
import fnmatch
import logging
import os
import pathlib
import shlex
import subprocess
import sys

import pytest

from graminelibos.regression import HAS_SGX, run_command

DEFAULT_LTP_SCENARIO = 'install/runtest/syscalls'
DEFAULT_LTP_CONFIG = 'ltp.cfg'
if HAS_SGX:
    DEFAULT_LTP_CONFIG = 'ltp.cfg ltp-sgx.cfg ltp-bug-1075.cfg'

LTP_SCENARIO = os.environ.get('LTP_SCENARIO', DEFAULT_LTP_SCENARIO)
LTP_CONFIG = os.environ.get('LTP_CONFIG', DEFAULT_LTP_CONFIG).split(' ')
LTP_TIMEOUT_FACTOR = float(os.environ.get('LTP_TIMEOUT_FACTOR', '1'))


def read_scenario(scenario):
    """Read an LTP scenario file (list of tests).

    Each line specifies a name (tag) and a command.
    """

    with open(scenario, 'r') as f:
        for line in f:
            if line[0] in '\n#':
                continue
            tag, *cmd = shlex.split(line)
            yield tag, cmd


def is_wildcard_pattern(name):
    return bool(set(name) & set('*?[]!'))


def get_int_set(value):
    return set(int(i) for i in value.strip().split())


class Config:
    """Parser for LTP configuration files.

    A section name can be a test tag, or a wildcard matching many tags (e.g. `access*`). A wildcard
    section can only contain `skip = yes`.

    TODO: Instead of Python's INI flavor (`configparser`), use TOML, for consistency with the rest
    of the project.
    """

    def __init__(self, config_paths):
        self.cfg = configparser.ConfigParser(
            converters={
                'path': pathlib.Path,
                'intset': get_int_set,
            },
            defaults={
                'timeout': '30',
            },
        )

        for path in config_paths:
            with open(path, 'r') as f:
                self.cfg.read_file(f)

        self.skip_patterns = []
        for name, section in self.cfg.items():
            if is_wildcard_pattern(name):
                for key in section:
                    if key != 'skip' and section[key] != self.cfg.defaults().get(key):
                        raise ValueError(
                            'wildcard sections like {!r} can only contain "skip", not {!r}'.format(
                                name, key))
                if section.get('skip'):
                    self.skip_patterns.append(name)

    def get(self, tag):
        """Find a section for given tag.

        Returns the default section if there's no specific one, and None if the test should be
        skipped.
        """
        if self.cfg.has_section(tag):
            section = self.cfg[tag]
            if section.get('skip'):
                return None
            return section

        for pattern in self.skip_patterns:
            if fnmatch.fnmatch(tag, pattern):
                return None

        return self.cfg[self.cfg.default_section]


def list_tests(ltp_config=LTP_CONFIG, ltp_scenario=LTP_SCENARIO):
    """List all tests along with their configuration."""

    config = Config(ltp_config)

    for tag, cmd in read_scenario(ltp_scenario):
        section = config.get(tag)
        yield tag, cmd, section


def parse_test_output(stdout):
    """Parse LTP stdout to determine passed/failed subtests.

    Returns two sets: passed and failed subtest numbers.
    """

    passed = set()
    failed = set()

    subtest = 0
    for line in stdout.splitlines():
        if line == 'Summary':
            break

        # Drop this line so that we get consistent offsets
        if line == 'WARNING: no physical memory support, process creation may be slow.':
            continue

        tokens = line.split()
        if len(tokens) < 2:
            continue

        if 'INFO' in line:
            continue

        if tokens[1].isdigit():
            subtest = int(tokens[1])
        else:
            subtest += 1

        if 'TPASS' in line or 'PASS:' in line:
            passed.add(subtest)
        elif any(t in line for t in ['TFAIL', 'FAIL:', 'TCONF', 'CONF:', 'TBROK', 'BROK:']):
            failed.add(subtest)

    return passed, failed


def check_must_pass(passed, failed, must_pass):
    """Verify the test results based on `must-pass` specified in configuration file."""

    # No `must-pass` means all tests must pass
    if not must_pass:
        if failed:
            pytest.fail('Failed subtests: {}'.format(failed))
        return

    must_pass_passed = set()
    must_pass_failed = set()
    must_pass_unknown = set()
    for subtest in must_pass:
        if subtest in passed:
            must_pass_passed.add(subtest)
        elif subtest in failed:
            must_pass_failed.add(subtest)
        else:
            must_pass_unknown.add(subtest)

    if must_pass_failed or must_pass_unknown:
        pytest.fail('Failed or unknown subtests specified in must-pass: {}'.format(
            must_pass_failed | must_pass_unknown))

    if not failed and passed == must_pass_passed:
        pytest.fail('The must-pass list specifies all tests, remove it from config')

    if not passed:
        pytest.fail('All subtests skipped, replace must-pass with skip')


def test_ltp(cmd, section):
    must_pass = section.getintset('must-pass')

    loader = 'gramine-sgx' if HAS_SGX else 'gramine-direct'
    timeout = int(section.getfloat('timeout') * LTP_TIMEOUT_FACTOR)
    full_cmd = [loader, *cmd]

    logging.info('command: %s', full_cmd)
    logging.info('must_pass: %s', list(must_pass) if must_pass else 'all')

    returncode, stdout, _stderr = run_command(full_cmd, timeout=timeout, can_fail=True)

    # Parse output regardless of whether `must_pass` is specified: unfortunately some tests
    # do not exit with non-zero code when failing, because they rely on `MAP_SHARED` (which
    # we do not support correctly) for collecting test results.
    passed, failed = parse_test_output(stdout)

    logging.info('returncode: %s', returncode)
    logging.info('passed: %s', list(passed))
    logging.info('failed: %s', list(failed))

    if not must_pass and returncode:
        pytest.fail('{} exited with status {}'.format(full_cmd, returncode))

    check_must_pass(passed, failed, must_pass)


def test_lint():
    cmd = ['./contrib/conf_lint.py', '--scenario', LTP_SCENARIO, *LTP_CONFIG]
    p = subprocess.run(cmd)
    if p.returncode:
        pytest.fail('conf_lint.py failed, see stdout for details')


def pytest_generate_tests(metafunc):
    """Generate all tests.

    This function is called by Pytest, and it's responsible for generating parameters for
    `test_ltp`.
    """

    if metafunc.function is test_ltp:
        params = []
        for tag, cmd, section in list_tests():
            # If a test should be skipped, mark it as such, but add it for Pytest anyway: we want
            # skipped tests to be visible in the report.
            marks = [] if section else [pytest.mark.skip]
            params.append(pytest.param(cmd, section, id=tag, marks=marks))

        metafunc.parametrize('cmd,section', params)


def main():
    if sys.argv[1:] == ['--list']:
        seen = set()
        for _tag, cmd, section in list_tests():
            executable = cmd[0]
            if section and executable not in seen:
                seen.add(executable)
                print(executable)
    else:
        usage = '''\
Usage:

    {} --list   (to list test executables)

Invoke Pytest directly (python3 -m pytest) to run tests.

Supports the following environment variables:

    SGX: set to 1 to enable SGX mode (default: disabled)
    LTP_SCENARIO: LTP scenario file (default: {})
    LTP_CONFIG: space-separated list of LTP config files (default: {})
    LTP_TIMEOUT_FACTOR: multiply all timeouts by given value
'''.format(sys.argv[0], DEFAULT_LTP_SCENARIO, DEFAULT_LTP_CONFIG)
        print(usage, file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
