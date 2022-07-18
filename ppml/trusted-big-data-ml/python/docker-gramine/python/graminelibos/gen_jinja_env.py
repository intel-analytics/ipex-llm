import os
import pathlib
import subprocess
import sys
import sysconfig

import jinja2

from . import _CONFIG_PKGLIBDIR

def ldd(*args):
    '''
    Args:
        binaries for which to generate manifest trusted files list.
    '''
    # Be careful: We have to skip vdso, which doesn't have a corresponding file on the disk (we
    # assume that such files have paths starting with '/', seems ldd always prints absolute paths).
    # Also, old ldd (from Ubuntu 16.04) prints vdso differently than newer ones:
    # old:
    #     linux-vdso.so.1 =>  (0x00007ffd31fee000)
    # new:
    #     linux-vdso.so.1 (0x00007ffd31fee000)
    ret = set()
    for line in subprocess.check_output(['ldd', *(os.fspath(i) for i in args)]).decode('ascii'):
        line = line.strip().split()
        if line[1] == '=>' and line[2].startswith('/'):
            ret.add(line[2])
        elif line[0].startswith('/') and line[1].startswith('/'):
            ret.add(line[0])
    return sorted(ret)

def add_globals_from_python(env):
    paths = sysconfig.get_paths()
    env.globals['python'] = {
        'stdlib': pathlib.Path(paths['stdlib']),
        'platstdlib': pathlib.Path(paths['platstdlib']),
        'purelib': pathlib.Path(paths['purelib']),

        # TODO rpm-based distros
        'distlib': pathlib.Path(sysconfig.get_path('stdlib',
                vars={'py_version_short': sys.version_info[0]})
            ) / 'dist-packages',

        'get_config_var': sysconfig.get_config_var,
        'ext_suffix': sysconfig.get_config_var('EXT_SUFFIX'),

        'get_path': sysconfig.get_path,
        'get_paths': sysconfig.get_paths,

        'implementation': sys.implementation,
    }

class Runtimedir:
    @staticmethod
    def __call__(libc='glibc'):
        return (pathlib.Path(_CONFIG_PKGLIBDIR) / 'runtime' / libc).resolve()
    def __str__(self):
        return str(self())
    def __truediv__(self, other):
        return self() / other

def add_globals_from_gramine(env):
    env.globals['gramine'] = {
        'libos': pathlib.Path(_CONFIG_PKGLIBDIR) / 'libsysdb.so',
        'pkglibdir': pathlib.Path(_CONFIG_PKGLIBDIR),
        'runtimedir': Runtimedir(),
    }

    try:
        from . import _offsets as offsets # pylint: disable=import-outside-toplevel
    except ImportError: # no SGX gramine installed, skipping
        pass
    else:
        env.globals['gramine'].update(
            (k, v) for k, v in offsets.__dict__.items()
            if not k.startswith('_'))

def add_globals_misc(env):
    env.globals['env'] = os.environ
    env.globals['ldd'] = ldd

def make_env():
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    add_globals_from_gramine(env)
    add_globals_from_python(env)
    add_globals_misc(env)
    return env
