# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2021 Wojtek Porczyk <woju@invisiblethingslab.com>
# Copyright (C) 2022 Intel Corporation
#                    Michał Kowalczyk <mkow@invisiblethingslab.com>
#                    Borys Popławski <borysp@invisiblethingslab.com>

"""
Gramine manifest management and rendering
"""

import hashlib
import os
import pathlib

import toml

from . import _env

DEFAULT_ENCLAVE_SIZE = '256M'
DEFAULT_THREAD_NUM = 4

class ManifestError(Exception):
    """Thrown at errors in manifest parsing and handling.

    Contains a string with error description.
    """

def hash_file_contents(path):
    with open(path, 'rb') as f:
        sha = hashlib.sha256()
        for chunk in iter(lambda: f.read(128 * sha.block_size), b''):
            sha.update(chunk)
        return sha.hexdigest()

def uri2path(uri):
    if not uri.startswith('file:'):
        raise ManifestError(f'Unsupported URI type: {uri}')
    return pathlib.Path(uri[len('file:'):])

def append_tf(trusted_files, uri, hash_):
    trusted_files.append({'uri': uri, 'sha256': hash_})

def append_trusted_dir_or_file(trusted_files, val, expanded):
    if isinstance(val, dict):
        uri = val['uri']
        if val.get('sha256'):
            append_tf(trusted_files, uri, val['sha256'])
            return
    elif isinstance(val, str):
        uri = val
    else:
        raise ManifestError(f'Unknown trusted file format: {val!r}')

    path = uri2path(uri)
    if not path.exists():
        raise ManifestError(f'Cannot resolve {path}')
    if path.is_dir():
        if not uri.endswith('/'):
            raise ManifestError(f'Directory URI ({uri}) does not end with "/"')

        expanded.append(path)
        for sub_path in sorted(path.rglob('*')):
            expanded.append(sub_path)
            if sub_path.is_file():
                # Skip inaccessible files
                if os.access(sub_path, os.R_OK):
                    append_tf(trusted_files, f'file:{sub_path}', hash_file_contents(sub_path))
    else:
        assert path.is_file()
        append_tf(trusted_files, uri, hash_file_contents(path))
        expanded.append(path)

class Manifest:
    """Just a representation of a manifest.

    You can access or change specific manifest entries via ``[]`` operator (just like a normal
    python ``dict``).

    Args:
        manifest_str (str): the manifest in the TOML format.
    """

    def __init__(self, manifest_str):
        manifest = toml.loads(manifest_str)

        sgx = manifest.setdefault('sgx', {})
        sgx.setdefault('trusted_files', [])
        sgx.setdefault('enclave_size', DEFAULT_ENCLAVE_SIZE)
        sgx.setdefault('thread_num', DEFAULT_THREAD_NUM)
        sgx.setdefault('isvprodid', 0)
        sgx.setdefault('isvsvn', 0)
        sgx.setdefault('remote_attestation', False)
        sgx.setdefault('debug', False)
        sgx.setdefault('require_avx', False)
        sgx.setdefault('require_avx512', False)
        sgx.setdefault('require_mpx', False)
        sgx.setdefault('require_pkru', False)
        sgx.setdefault('require_amx', False)
        sgx.setdefault('support_exinfo', False)
        sgx.setdefault('nonpie_binary', False)
        sgx.setdefault('enable_stats', False)

        if not isinstance(sgx['trusted_files'], list):
            raise ValueError("Unsupported trusted files syntax, more info: " +
                  "https://gramine.readthedocs.io/en/latest/manifest-syntax.html#trusted-files")

        # Current toml versions (< 1.0) do not support non-homogeneous arrays
        trusted_files = []
        for tf in sgx['trusted_files']:
            if isinstance(tf, dict) and 'uri' in tf:
                trusted_files.append(tf)
            elif isinstance(tf, str):
                append_tf(trusted_files, tf, None)
            else:
                raise ManifestError(f'Unknown trusted file format: {tf!r}')

        sgx['trusted_files'] = trusted_files

        self._manifest = manifest

    def __getitem__(self, key):
        return self._manifest[key]

    def __setitem__(self, key, val):
        self._manifest[key] = val

    @classmethod
    def from_template(cls, template, variables=None):
        """Render template into Manifest.

        Creates a manifest from the jinja template given as string. Optional variables may be given
        as mapping.

        Args:
            template (str): jinja2 template of the manifest
            variables (:obj:`dict`, optional): Dictionary of variables that are used in
                the template.

        Returns:
            Manifest: instance created from rendered template.
        """
        return cls(_env.from_string(template).render(**(variables or {})))

    @classmethod
    def loads(cls, s):
        return cls(s)

    @classmethod
    def load(cls, f):
        return cls.loads(f.read())

    def dumps(self):
        return toml.dumps(self._manifest)

    def dump(self, f):
        toml.dump(self._manifest, f)

    def expand_all_trusted_files(self):
        """Expand all trusted files entries.

        Collects all trusted files entries, hashes each of them (skipping these which already had a
        hash present) and updates ``sgx.trusted_files`` manifest entry with the result.

        Returns a list of all expanded files, i.e. files that we need to hash, and directories that
        we needed to list.

        Raises:
            ManifestError: There was an error with the format of some trusted files in the manifest
                or some of them could not be loaded from the filesystem.

        """
        trusted_files = []
        expanded = []
        for tf in self['sgx']['trusted_files']:
            append_trusted_dir_or_file(trusted_files, tf, expanded)

        self['sgx']['trusted_files'] = trusted_files
        return expanded

    def get_dependencies(self):
        """Generate list of files which this manifest depends on.

        Collects all trusted files that are not yet expanded (do not have a hash in the entry) and
        returns them.

        Returns:
            list(pathlib.Path): List of paths to the files this manifest depends on.

        Raises:
            ManifestError: One of the found URIs is in an unsupported format.
        """
        deps = set()

        for tf in self['sgx']['trusted_files']:
            if not tf.get('sha256'):
                deps.add(uri2path(tf['uri']))

        return deps
