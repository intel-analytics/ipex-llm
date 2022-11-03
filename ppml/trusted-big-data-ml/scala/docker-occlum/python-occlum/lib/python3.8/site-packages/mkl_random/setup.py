#!/usr/bin/env python
# Copyright (c) 2017-2019, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division, print_function

import os
import sys
from os.path import join, split, dirname, abspath
from distutils.msvccompiler import get_build_version as get_msvc_build_version
from numpy import get_include as get_numpy_include
from distutils.sysconfig import get_python_inc as get_python_include

def needs_mingw_ftime_workaround():
    # We need the mingw workaround for _ftime if the msvc runtime version is
    # 7.1 or above and we build with mingw ...
    # ... but we can't easily detect compiler version outside distutils command
    # context, so we will need to detect in randomkit whether we build with gcc
    msver = get_msvc_build_version()
    if msver and msver >= 8:
        return True

    return False

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('mkl_random', parent_package, top_path)
    mkl_root = os.getenv('MKLROOT', None)
    if mkl_root:
        mkl_info = {
            'include_dirs': [join(mkl_root, 'include')],
            'library_dirs': [join(mkl_root, 'lib'), join(mkl_root, 'lib', 'intel64')],
            'libraries': ['mkl_rt']
        }
    else:
        mkl_info = get_info('mkl')

    mkl_include_dirs = mkl_info.get('include_dirs', [])
    mkl_library_dirs = mkl_info.get('library_dirs', [])
    libs = mkl_info.get('libraries', ['mkl_rt'])
    if sys.platform == 'win32':
        libs.append('Advapi32')

    Q = '/Q' if sys.platform.startswith('win') or sys.platform == 'cygwin' else '-'

    pdir = 'mkl_random'
    wdir = join(pdir, 'src')

    eca = [Q + 'std=c++11']
    if sys.platform == "linux":
        eca.extend(["-Wno-unused-but-set-variable", "-Wno-unused-function"])

    config.add_library(
        'mkl_dists',
        sources=join(wdir, 'mkl_distributions.cpp'),
        libraries=libs,
        include_dirs=[wdir,pdir,get_numpy_include(),get_python_include()],
        extra_compiler_args=eca,
        depends=[join(wdir, '*.h'),],
        language='c++',
    )

    try:
        from Cython.Build import cythonize
        sources = [join(pdir, 'mklrand.pyx')]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = [join(pdir, 'mklrand.c')]
        if not exists(sources[0]):
            raise ValueError(str(e) + '. ' + 
                             'Cython is required to build the initial .c file.')


    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    defs = [('_FILE_OFFSET_BITS', '64'),
            ('_LARGEFILE_SOURCE', '1'),
            ('_LARGEFILE64_SOURCE', '1')]
    if needs_mingw_ftime_workaround():
        defs.append(("NEED_MINGW_TIME_WORKAROUND", None))


    sources = sources + [join(wdir, x) for x in ['randomkit.c']] 
    libs = libs + ['mkl_dists']
    config.add_extension(
        name='mklrand',
        sources=sources,
        libraries=libs,
        include_dirs=[wdir,pdir] + mkl_include_dirs,
        library_dirs=mkl_library_dirs,
        define_macros=defs,
    )

    config.add_data_files(('.', join('src', 'randomkit.h')))
    config.add_data_files(('.', join('src', 'mkl_distributions.h')))
    config.add_data_dir('tests')

    if have_cython:
        config.ext_modules = cythonize(config.ext_modules, include_path=[pdir, wdir])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
