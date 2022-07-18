# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2021 Intel Corporation
#                    Paweł Marczewski <pawel@invisiblethingslab.com>

import io
import os
import platform
import subprocess
import sys

import toml

from . import ninja_syntax, _CONFIG_SYSLIBDIR, _CONFIG_PKGLIBDIR

try:
    from .sgx_sign import SGX_RSA_KEY_PATH as _SGX_RSA_KEY_PATH
except ImportError:
    # if we don't have sgx built, this won't work anyway
    _SGX_RSA_KEY_PATH = '/dev/null'


class TestConfig:
    '''
    Class responsible for loading the test configuration (`tests.toml`) and generating a Ninja build
    file (`build.ninja`).

    `tests.toml` can have the following keys:

    - `manifests`, `sgx.manifests`, `arch.[ARCH].manifests`: list of manifests to build
      (for all hosts, SGX-only, [ARCH]-only)

    - `manifests_cmd` (and same with `sgx.` and `arch.[ARCH].`): a shell command that prints out
      manifests to build, in separate lines (used by LTP, where the list depends on enabled tests)

    - `binary_dir`: path to test binaries, passed as `binary_dir` to manifest templates; expands
      @GRAMINE_PKGLIBDIR@ to library directory of Gramine's installation

    - `libc`: name of the libc to build against, currently supported: 'glibc' (default), 'musl'

    Ninja handles the following targets:

    - `NAME.manifest`, `NAME.manifest.sgx`, `NAME.sig`, `NAME.token`
    - `direct`, `sgx`: all files
    - `direct-NAME`, `sgx-NAME`: files related to a single manifest
    '''

    def __init__(self, path):
        self.config_path = path

        data = toml.load(path)

        self.manifests = self.get_manifests(data)
        arch = platform.machine()
        arch_data = data.get('arch', {}).get(arch, {})
        self.manifests += self.get_manifests(arch_data)

        self.sgx_manifests = self.get_manifests(data.get('sgx', {}))

        self.binary_dir = data.get('binary_dir', '.').replace(
            '@GRAMINE_PKGLIBDIR@', _CONFIG_PKGLIBDIR)

        self.libc = data.get('libc', 'glibc')

        self.arch_libdir = _CONFIG_SYSLIBDIR

        # Used by LTP, for `libstdbuf.so`
        for coreutils_libdir in [
                os.path.join('/usr', self.arch_libdir[1:], 'coreutils'),
                '/usr/libexec/coreutils',
        ]:
            if os.path.isdir(coreutils_libdir):
                self.coreutils_libdir = coreutils_libdir
                break
        else:
            raise Exception('Cannot determine coreutils libdir')

        self.key = os.environ.get('SGX_SIGNER_KEY', None)
        if not self.key:
            self.key = os.fspath(_SGX_RSA_KEY_PATH)

        self.all_manifests = self.manifests + self.sgx_manifests

    @staticmethod
    def get_manifests(data):
        manifests = data.get('manifests', [])
        cmd = data.get('manifests_cmd')
        if cmd:
            output = subprocess.check_output(cmd, shell=True).decode()
            manifests += output.splitlines()
        return manifests

    def gen_build_file(self, ninja_path):
        output = io.StringIO()
        ninja = ninja_syntax.Writer(output)

        self._gen_header(ninja)
        self._gen_rules(ninja)
        self._gen_targets(ninja, ninja_path)

        with open(ninja_path, 'w') as f:
            f.write(output.getvalue())

    def _gen_header(self, ninja):
        ninja.comment('Auto-generated, do not edit!')
        ninja.newline()

    def _gen_rules(self, ninja):
        ninja.variable('BINARY_DIR', self.binary_dir)
        ninja.variable('ARCH_LIBDIR', self.arch_libdir)
        ninja.variable('COREUTILS_LIBDIR', self.coreutils_libdir)
        ninja.variable('KEY', self.key)
        ninja.variable('GRAMINE_LIBC', self.libc)
        ninja.newline()

        ninja.rule(
            name='manifest',
            command=('gramine-manifest '
                     '-Darch_libdir=$ARCH_LIBDIR '
                     '-Dcoreutils_libdir=$COREUTILS_LIBDIR '
                     '-Dentrypoint=$ENTRYPOINT '
                     '-Dbinary_dir=$BINARY_DIR '
                     '-Dlibc=$GRAMINE_LIBC '
                     '$in $out'),
            description='manifest: $out'
        )
        ninja.newline()

        ninja.rule(
            name='sgx-sign',
            command=('gramine-sgx-sign --quiet --manifest $in --key $KEY --depfile $out.d '
                     '--output $out'),
            depfile='$out.d',
            description='SGX sign: $out',
        )
        ninja.newline()

        ninja.rule(
            name='sgx-get-token',
            command='gramine-sgx-get-token --quiet --sig $in --output $out',
            description='SGX token: $out',
        )
        ninja.newline()

        ninja.rule(
            name='regenerate',
            command='gramine-test regenerate',
            description='Regenerating build file',
            generator=True,
        )

        ninja.newline()

    def _gen_targets(self, ninja, ninja_path):
        # regenerate `build.ninja` (in case the user invokes Ninja directly)
        ninja.build(
            outputs=[ninja_path],
            rule='regenerate',
            inputs=[self.config_path],
        )

        ninja.build(
            outputs=['direct'],
            rule='phony',
            inputs=([f'{name}.manifest' for name in self.manifests]),
        )
        ninja.default('direct')
        ninja.newline()

        ninja.build(
            outputs=['sgx'],
            rule='phony',
            inputs=([f'{name}.manifest' for name in self.all_manifests] +
                    [f'{name}.manifest.sgx' for name in self.all_manifests] +
                    [f'{name}.sig' for name in self.all_manifests] +
                    [f'{name}.token' for name in self.all_manifests]),
        )
        ninja.newline()

        for name in self.all_manifests:
            template = f'{name}.manifest.template'
            if not os.path.exists(template):
                template = 'manifest.template'

            ninja.build(
                outputs=[f'{name}.manifest'],
                rule='manifest',
                inputs=[template],
                variables={'ENTRYPOINT': name},
            )

            ninja.build(
                outputs=[f'{name}.manifest.sgx'],
                implicit_outputs=[f'{name}.sig'],
                rule='sgx-sign',
                inputs=[f'{name}.manifest'],
                implicit=([self.key]),
            )

            ninja.build(
                outputs=[f'{name}.token'],
                rule='sgx-get-token',
                inputs=[f'{name}.sig'],
            )

            ninja.build(
                outputs=[f'direct-{name}'],
                rule='phony',
                inputs=[f'{name}.manifest'],
            )

            ninja.build(
                outputs=[f'sgx-{name}'],
                rule='phony',
                inputs=[f'{name}.manifest', f'{name}.manifest.sgx', f'{name}.sig', f'{name}.token'],
            )

            ninja.newline()


def gen_build_file(conf_file_name='tests.toml'):
    config = TestConfig(conf_file_name)
    config.gen_build_file('build.ninja')


def exec_pytest(sgx, args):
    env = os.environ.copy()
    env['SGX'] = '1' if sgx else ''

    argv = [os.path.basename(sys.executable), '-m', 'pytest'] + list(args)
    print(' '.join(argv))
    os.execve(sys.executable, argv, env)


def run_ninja(args):
    argv = ['ninja'] + list(args)
    print(' '.join(argv))
    subprocess.check_call(argv)


def exec_gramine(sgx, name, args):
    prog = 'gramine-sgx' if sgx else 'gramine-direct'
    argv = [prog, name] + list(args)
    print(' '.join(argv))
    os.execvp(prog, argv)
