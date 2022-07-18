'''Python support for Gramine'''
import os as _os

__version__ = '@VERSION@'

_CONFIG_PKGLIBDIR = '@PKGLIBDIR@'
_CONFIG_LIBDIR = '@LIBDIR@'
_CONFIG_SYSLIBDIR = '@SYSLIBDIR@'
_CONFIG_SGX_ENABLED = '@SGX_ENABLED@' == '1'

if __version__.startswith('@') and not _os.getenv('GRAMINE_IMPORT_FOR_SPHINX_ANYWAY') == '1':
    raise RuntimeError(
        'You are attempting to run the tools from repo, without installing. '
        'Please install Gramine before running Python tools. See '
        'https://gramine.readthedocs.io/en/latest/devel/building.html.')

# pylint: disable=wrong-import-position
from .gen_jinja_env import make_env

_env = make_env()

from .manifest import Manifest, ManifestError
if _CONFIG_SGX_ENABLED:
    from .sgx_get_token import get_token
    from .sgx_sign import get_tbssigstruct, sign_with_local_key, SGX_LIBPAL, SGX_RSA_KEY_PATH
    from .sigstruct import Sigstruct
