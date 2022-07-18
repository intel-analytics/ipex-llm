# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2014 Stony Brook University
# Copyright (C) 2022 Intel Corporation
#                    Michał Kowalczyk <mkow@invisiblethingslab.com>
#                    Borys Popławski <borysp@invisiblethingslab.com>
#                    Wojtek Porczyk <woju@invisiblethingslab.com>
#

import hashlib
import os
import pathlib
import struct
import subprocess

from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from . import _CONFIG_PKGLIBDIR
from . import _offsets as offs # pylint: disable=import-error,no-name-in-module
from .manifest import Manifest
from .sigstruct import Sigstruct


_cryptography_backend = backends.default_backend()

# Default / Architectural Options

ARCHITECTURE = 'amd64'

SGX_LIBPAL = os.path.join(_CONFIG_PKGLIBDIR, 'sgx/libpal.so')

SGX_RSA_PUBLIC_EXPONENT = 3
SGX_RSA_KEY_SIZE = 3072
_xdg_config_home = pathlib.Path(os.getenv('XDG_CONFIG_HOME',
    pathlib.Path.home() / '.config'))
SGX_RSA_KEY_PATH = _xdg_config_home / 'gramine' / 'enclave-key.pem'

# Utilities

ZERO_PAGE = bytes(offs.PAGESIZE)


def roundup(addr):
    remaining = addr % offs.PAGESIZE
    if remaining:
        return addr + (offs.PAGESIZE - remaining)
    return addr


def rounddown(addr):
    return addr - addr % offs.PAGESIZE


def parse_size(value):
    scale = 1
    if value.endswith('K'):
        scale = 1024
    elif value.endswith('M'):
        scale = 1024 * 1024
    elif value.endswith('G'):
        scale = 1024 * 1024 * 1024
    if scale != 1:
        value = value[:-1]
    return int(value, 0) * scale


# Loading Enclave Attributes


def collect_bits(manifest_sgx, options_dict):
    val = 0
    for opt, bits in options_dict.items():
        if manifest_sgx[opt] == 1:
            val |= bits
    return val


def get_enclave_attributes(manifest_sgx):
    flags_dict = {
        'debug': offs.SGX_FLAGS_DEBUG,
    }

    xfrms_dict = {
        'require_avx': offs.SGX_XFRM_AVX,
        'require_avx512': offs.SGX_XFRM_AVX512,
        'require_mpx': offs.SGX_XFRM_MPX,
        'require_pkru': offs.SGX_XFRM_PKRU,
        'require_amx': offs.SGX_XFRM_AMX,
    }

    miscs_dict = {
        'support_exinfo': offs.SGX_MISCSELECT_EXINFO,
    }

    flags = collect_bits(manifest_sgx, flags_dict)
    if ARCHITECTURE == 'amd64':
        flags |= offs.SGX_FLAGS_MODE64BIT

    xfrms = offs.SGX_XFRM_LEGACY | collect_bits(manifest_sgx, xfrms_dict)
    miscs = collect_bits(manifest_sgx, miscs_dict)

    return flags, xfrms, miscs


# Populate Enclave Memory

PAGEINFO_R = 0x1
PAGEINFO_W = 0x2
PAGEINFO_X = 0x4
PAGEINFO_TCS = 0x100
PAGEINFO_REG = 0x200


def get_loadcmds(elf_filename):
    loadcmds = []
    proc = subprocess.Popen(['readelf', '-l', '-W', elf_filename],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.decode()
        stripped = line.strip()
        if not stripped.startswith('LOAD'):
            continue
        tokens = stripped.split()
        if len(tokens) < 6:
            continue
        if len(tokens) >= 7 and tokens[7] == 'E':
            tokens[6] += tokens[7]
        prot = 0
        for token in tokens[6]:
            if token == 'R':
                prot = prot | 4
            if token == 'W':
                prot = prot | 2
            if token == 'E':
                prot = prot | 1

        loadcmds.append((int(tokens[1][2:], 16),  # offset
                         int(tokens[2][2:], 16),  # addr
                         int(tokens[4][2:], 16),  # filesize
                         int(tokens[5][2:], 16),  # memsize
                         prot))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'Parsing {elf_filename} as ELF failed')
    return loadcmds


class MemoryArea:
    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(self, desc, elf_filename=None, content=None, addr=None, size=None,
                 flags=None, measure=True):
        # pylint: disable=too-many-arguments
        self.desc = desc
        self.elf_filename = elf_filename
        self.content = content
        self.addr = addr
        self.size = size
        self.flags = flags
        self.measure = measure

        if elf_filename:
            loadcmds = get_loadcmds(elf_filename)
            mapaddr = 0xffffffffffffffff
            mapaddr_end = 0
            for (_, addr_, _, memsize, _) in loadcmds:
                if rounddown(addr_) < mapaddr:
                    mapaddr = rounddown(addr_)
                if roundup(addr_ + memsize) > mapaddr_end:
                    mapaddr_end = roundup(addr_ + memsize)

            self.size = mapaddr_end - mapaddr
            if mapaddr > 0:
                self.addr = mapaddr

        if self.addr is not None:
            self.addr = rounddown(self.addr)
        if self.size is not None:
            self.size = roundup(self.size)


def get_memory_areas(attr, libpal):
    areas = []
    areas.append(
        MemoryArea('ssa',
                   size=attr['thread_num'] * offs.SSA_FRAME_SIZE * offs.SSA_FRAME_NUM,
                   flags=PAGEINFO_R | PAGEINFO_W | PAGEINFO_REG))
    areas.append(MemoryArea('tcs', size=attr['thread_num'] * offs.TCS_SIZE,
                            flags=PAGEINFO_TCS))
    areas.append(MemoryArea('tls', size=attr['thread_num'] * offs.PAGESIZE,
                            flags=PAGEINFO_R | PAGEINFO_W | PAGEINFO_REG))

    for _ in range(attr['thread_num']):
        areas.append(MemoryArea('stack', size=offs.ENCLAVE_STACK_SIZE,
                                flags=PAGEINFO_R | PAGEINFO_W | PAGEINFO_REG))
    for _ in range(attr['thread_num']):
        areas.append(MemoryArea('sig_stack', size=offs.ENCLAVE_SIG_STACK_SIZE,
                                flags=PAGEINFO_R | PAGEINFO_W | PAGEINFO_REG))

    areas.append(MemoryArea('pal', elf_filename=libpal, flags=PAGEINFO_REG))
    return areas


def find_areas(areas, desc):
    return [area for area in areas if area.desc == desc]


def find_area(areas, desc, allow_none=False):
    matching = find_areas(areas, desc)

    if not matching and allow_none:
        return None

    if len(matching) != 1:
        raise KeyError(f'Could not find exactly one MemoryArea {desc!r}')

    return matching[0]


def entry_point(elf_path):
    env = os.environ
    env['LC_ALL'] = 'C'
    out = subprocess.check_output(
        ['readelf', '-l', '--', elf_path], env=env)
    for line in out.splitlines():
        line = line.decode()
        if line.startswith('Entry point '):
            return int(line[12:], 0)
    raise ValueError('Could not find entry point of elf file')


def gen_area_content(attr, areas, enclave_base, enclave_heap_min):
    # pylint: disable=too-many-locals
    manifest_area = find_area(areas, 'manifest')
    pal_area = find_area(areas, 'pal')
    ssa_area = find_area(areas, 'ssa')
    tcs_area = find_area(areas, 'tcs')
    tls_area = find_area(areas, 'tls')
    stacks = find_areas(areas, 'stack')
    sig_stacks = find_areas(areas, 'sig_stack')

    tcs_data = bytearray(tcs_area.size)

    def set_tcs_field(t, offset, pack_fmt, value):
        struct.pack_into(pack_fmt, tcs_data, t * offs.TCS_SIZE + offset, value)

    tls_data = bytearray(tls_area.size)

    def set_tls_field(t, offset, value):
        struct.pack_into('<Q', tls_data, t * offs.PAGESIZE + offset, value)

    enclave_heap_max = pal_area.addr

    # Sanity check that we measure everything except the heap which is zeroed
    # on enclave startup.
    for area in areas:
        if (area.addr + area.size <= enclave_heap_min or
                area.addr >= enclave_heap_max):
            if not area.measure:
                raise ValueError('Memory area, which is not the heap, is not measured')
        elif area.desc != 'free':
            raise ValueError('Unexpected memory area is in heap range')

    for t in range(0, attr['thread_num']):
        ssa = ssa_area.addr + offs.SSA_FRAME_SIZE * offs.SSA_FRAME_NUM * t
        ssa_offset = ssa - enclave_base
        set_tcs_field(t, offs.TCS_OSSA, '<Q', ssa_offset)
        set_tcs_field(t, offs.TCS_NSSA, '<L', offs.SSA_FRAME_NUM)
        set_tcs_field(t, offs.TCS_OENTRY, '<Q',
                      pal_area.addr + entry_point(pal_area.elf_filename) - enclave_base)
        set_tcs_field(t, offs.TCS_OGS_BASE, '<Q', tls_area.addr - enclave_base + offs.PAGESIZE * t)
        set_tcs_field(t, offs.TCS_OFS_LIMIT, '<L', 0xfff)
        set_tcs_field(t, offs.TCS_OGS_LIMIT, '<L', 0xfff)

        set_tls_field(t, offs.SGX_COMMON_SELF, tls_area.addr + offs.PAGESIZE * t)
        set_tls_field(t, offs.SGX_COMMON_STACK_PROTECTOR_CANARY,
                      offs.STACK_PROTECTOR_CANARY_DEFAULT)
        set_tls_field(t, offs.SGX_ENCLAVE_SIZE, attr['enclave_size'])
        set_tls_field(t, offs.SGX_TCS_OFFSET, tcs_area.addr - enclave_base + offs.TCS_SIZE * t)
        set_tls_field(t, offs.SGX_INITIAL_STACK_ADDR, stacks[t].addr + stacks[t].size)
        set_tls_field(t, offs.SGX_SIG_STACK_LOW, sig_stacks[t].addr)
        set_tls_field(t, offs.SGX_SIG_STACK_HIGH, sig_stacks[t].addr + sig_stacks[t].size)
        set_tls_field(t, offs.SGX_SSA, ssa)
        set_tls_field(t, offs.SGX_GPR, ssa + offs.SSA_FRAME_SIZE - offs.SGX_GPR_SIZE)
        set_tls_field(t, offs.SGX_MANIFEST_SIZE, len(manifest_area.content))
        set_tls_field(t, offs.SGX_HEAP_MIN, enclave_heap_min)
        set_tls_field(t, offs.SGX_HEAP_MAX, enclave_heap_max)

    tcs_area.content = tcs_data
    tls_area.content = tls_data


def populate_memory_areas(attr, areas, enclave_base, enclave_heap_min):
    last_populated_addr = enclave_base + attr['enclave_size']

    for area in areas:
        if area.addr is not None:
            continue

        area.addr = last_populated_addr - area.size
        if area.addr < enclave_heap_min:
            raise Exception('Enclave size is not large enough')
        last_populated_addr = area.addr

    free_areas = []
    for area in areas:
        addr = area.addr + area.size
        if addr < last_populated_addr:
            flags = PAGEINFO_R | PAGEINFO_W | PAGEINFO_X | PAGEINFO_REG
            free_areas.append(
                MemoryArea('free', addr=addr, size=last_populated_addr - addr,
                           flags=flags, measure=False))
            last_populated_addr = area.addr

    if last_populated_addr > enclave_heap_min:
        flags = PAGEINFO_R | PAGEINFO_W | PAGEINFO_X | PAGEINFO_REG
        free_areas.append(
            MemoryArea('free', addr=enclave_heap_min,
                       size=last_populated_addr - enclave_heap_min, flags=flags,
                       measure=False))

    gen_area_content(attr, areas, enclave_base, enclave_heap_min)

    return areas + free_areas


def generate_measurement(enclave_base, attr, areas, verbose=False):
    # pylint: disable=too-many-statements,too-many-branches,too-many-locals

    def do_ecreate(digest, size):
        data = struct.pack('<8sLQ44s', b'ECREATE', offs.SSA_FRAME_SIZE // offs.PAGESIZE, size, b'')
        digest.update(data)

    def do_eadd(digest, offset, flags):
        assert offset < attr['enclave_size']
        data = struct.pack('<8sQQ40s', b'EADD', offset, flags, b'')
        digest.update(data)

    def do_eextend(digest, offset, content):
        assert offset < attr['enclave_size']

        if len(content) != 256:
            raise ValueError('Exactly 256 bytes expected')

        data = struct.pack('<8sQ48s', b'EEXTEND', offset, b'')
        digest.update(data)
        digest.update(content)

    def include_page(digest, addr, flags, content, measure):
        if len(content) != offs.PAGESIZE:
            raise ValueError('Exactly one page expected')

        do_eadd(digest, addr - enclave_base, flags)
        if measure:
            for i in range(0, offs.PAGESIZE, 256):
                do_eextend(digest, addr - enclave_base + i, content[i:i + 256])

    mrenclave = hashlib.sha256()
    do_ecreate(mrenclave, attr['enclave_size'])

    def print_area(addr, size, flags, desc, measured):
        assert verbose

        if flags & PAGEINFO_REG:
            type_ = 'REG'
        if flags & PAGEINFO_TCS:
            type_ = 'TCS'
        prot = ['-', '-', '-']
        if flags & PAGEINFO_R:
            prot[0] = 'R'
        if flags & PAGEINFO_W:
            prot[1] = 'W'
        if flags & PAGEINFO_X:
            prot[2] = 'X'
        prot = ''.join(prot)

        desc = f'({desc})'
        if measured:
            desc += ' measured'

        print(f'    {addr:016x}-{addr+size:016x} [{type_}:{prot}] {desc}')

    def load_file(digest, file, offset, addr, filesize, memsize, desc, flags):
        # pylint: disable=too-many-arguments
        f_addr = rounddown(offset)
        m_addr = rounddown(addr)
        m_size = roundup(addr + memsize) - m_addr

        if verbose:
            print_area(m_addr, m_size, flags, desc, True)

        for page in range(m_addr, m_addr + m_size, offs.PAGESIZE):
            start = page - m_addr + f_addr
            end = start + offs.PAGESIZE
            start_zero = b''
            if start < offset:
                if offset - start >= offs.PAGESIZE:
                    start_zero = ZERO_PAGE
                else:
                    start_zero = bytes(offset - start)
            end_zero = b''
            if end > offset + filesize:
                if end - offset - filesize >= offs.PAGESIZE:
                    end_zero = ZERO_PAGE
                else:
                    end_zero = bytes(end - offset - filesize)
            start += len(start_zero)
            end -= len(end_zero)
            if start < end:
                file.seek(start)
                data = file.read(end - start)
            else:
                data = b''
            if len(start_zero + data + end_zero) != offs.PAGESIZE:
                raise Exception('wrong calculation')

            include_page(digest, page, flags, start_zero + data + end_zero, True)

    if verbose:
        print('Memory:')

    for area in areas:
        if area.elf_filename is not None:
            with open(area.elf_filename, 'rb') as file:
                loadcmds = get_loadcmds(area.elf_filename)
                if loadcmds:
                    mapaddr = 0xffffffffffffffff
                    for (offset, addr, filesize, memsize,
                         prot) in loadcmds:
                        if rounddown(addr) < mapaddr:
                            mapaddr = rounddown(addr)
                baseaddr_ = area.addr - mapaddr
                for (offset, addr, filesize, memsize, prot) in loadcmds:
                    flags = area.flags
                    if prot & 4:
                        flags = flags | PAGEINFO_R
                    if prot & 2:
                        flags = flags | PAGEINFO_W
                    if prot & 1:
                        flags = flags | PAGEINFO_X

                    if flags & PAGEINFO_X:
                        desc = 'code'
                    else:
                        desc = 'data'
                    load_file(mrenclave, file, offset, baseaddr_ + addr, filesize, memsize,
                              desc, flags)
        else:
            for addr in range(area.addr, area.addr + area.size, offs.PAGESIZE):
                data = ZERO_PAGE
                if area.content is not None:
                    start = addr - area.addr
                    end = start + offs.PAGESIZE
                    data = area.content[start:end]
                    data += b'\0' * (offs.PAGESIZE - len(data)) # pad last page
                include_page(mrenclave, addr, area.flags, data, area.measure)

            if verbose:
                print_area(area.addr, area.size, area.flags, area.desc, area.measure)

    return mrenclave.digest()


def get_mrenclave_and_manifest(manifest_path, libpal, verbose=False):
    with open(manifest_path, 'rb') as f: # pylint: disable=invalid-name
        manifest_data = f.read()
    manifest = Manifest.loads(manifest_data.decode('utf-8'))

    manifest_sgx = manifest['sgx']
    attr = {
        'enclave_size': parse_size(manifest_sgx['enclave_size']),
        'thread_num': manifest_sgx['thread_num'],
        'isv_prod_id': manifest_sgx['isvprodid'],
        'isv_svn': manifest_sgx['isvsvn'],
    }
    attr['flags'], attr['xfrms'], attr['misc_select'] = get_enclave_attributes(manifest_sgx)

    if verbose:
        print('Attributes:')
        print(f'    size:        {attr["enclave_size"]:#x}')
        print(f'    thread_num:  {attr["thread_num"]}')
        print(f'    isv_prod_id: {attr["isv_prod_id"]}')
        print(f'    isv_svn:     {attr["isv_svn"]}')
        print(f'    attr.flags:  {attr["flags"]:#x}')
        print(f'    attr.xfrm:   {attr["xfrms"]:#x}')
        print(f'    misc_select: {attr["misc_select"]:#x}')

        if manifest_sgx['remote_attestation']:
            spid = manifest_sgx.get('ra_client_spid', '')
            linkable = manifest_sgx.get('ra_client_linkable', False)
            print('SGX remote attestation:')
            if not spid:
                print('    DCAP/ECDSA')
            else:
                print(f'    EPID (spid = {spid}, linkable = {linkable})')

    # Populate memory areas
    memory_areas = get_memory_areas(attr, libpal)

    if manifest_sgx['nonpie_binary']:
        enclave_base = offs.DEFAULT_ENCLAVE_BASE
        enclave_heap_min = offs.MMAP_MIN_ADDR
    else:
        enclave_base = attr['enclave_size']
        enclave_heap_min = enclave_base

    manifest_data += b'\0' # in-memory manifest needs NULL-termination

    memory_areas = [
        MemoryArea('manifest', content=manifest_data, size=len(manifest_data),
                   flags=PAGEINFO_R | PAGEINFO_REG)
        ] + memory_areas

    memory_areas = populate_memory_areas(attr, memory_areas, enclave_base, enclave_heap_min)

    # Generate measurement
    mrenclave = generate_measurement(enclave_base, attr, memory_areas, verbose=verbose)

    if verbose:
        print('Measurement:')
        print(f'    {mrenclave.hex()}')

    return mrenclave, manifest


def get_tbssigstruct(manifest_path, date, libpal=SGX_LIBPAL, verbose=False):
    """Generate To Be Signed Sigstruct (TBSSIGSTRUCT).

    Generates a Sigstruct object using the provided data with all required fields initialized (i.e.
    all except those corresponding to the signature itself).

    Args:
        manifest_path (str): Path to the manifest file.
        date (date): Date to put into SIGSTRUCT.
        libpal (:obj:`str`, optional): Path to the libpal file.
        verbose (:obj:`bool`, optional): If true, print details to stdout.

    Returns:
        Sigstruct: SIGSTRUCT generated from provided data.
    """

    mrenclave, manifest = get_mrenclave_and_manifest(manifest_path, libpal, verbose=verbose)

    manifest_sgx = manifest['sgx']

    sig = Sigstruct()

    sig['date_year'] = date.year
    sig['date_month'] = date.month
    sig['date_day'] = date.day
    sig['enclave_hash'] = mrenclave
    sig['isv_prod_id'] = manifest_sgx['isvprodid']
    sig['isv_svn'] = manifest_sgx['isvsvn']

    attribute_flags, attribute_xfrms, misc_select = get_enclave_attributes(manifest_sgx)
    sig['attribute_flags'] = attribute_flags
    sig['attribute_xfrms'] = attribute_xfrms
    sig['misc_select'] = misc_select

    return sig


def sign_with_local_key(data, key):
    """Signs *data* using *key*.

    Function used to generate an RSA signature over provided data using a 3072-bit private key with
    the public exponent of 3 (hard Intel SGX requirement on the key size and the exponent).
    Suitable to be used as a callback to :py:func:`graminelibos.Sigstruct.sign()`.

    Args:
        data (bytes): Data to calculate the signature over.
        key (str): Path to a file with RSA private key.

    Returns:
        (int, int, int): Tuple of exponent, modulus and signature respectively.
    """
    proc = subprocess.Popen(
        ['openssl', 'rsa', '-modulus', '-in', key, '-noout'],
        stdout=subprocess.PIPE)
    modulus_out, _ = proc.communicate()
    modulus = bytes.fromhex(modulus_out[8:8+offs.SE_KEY_SIZE*2].decode())

    proc = subprocess.Popen(
        ['openssl', 'sha256', '-binary', '-sign', key],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    signature, _ = proc.communicate(data)

    exponent_int = 3
    modulus_int = int.from_bytes(modulus, byteorder='big')
    signature_int = int.from_bytes(signature, byteorder='big')

    return exponent_int, modulus_int, signature_int


def generate_private_key():
    """Generate RSA key suitable for use with SGX

    Returns:
        cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey: private key
    """
    return rsa.generate_private_key(
        public_exponent=SGX_RSA_PUBLIC_EXPONENT,
        key_size=SGX_RSA_KEY_SIZE,
        backend=_cryptography_backend)

def generate_private_key_pem():
    """Generate PEM-encoded RSA key suitable for use with SGX

    Returns:
        bytes: PEM-encoded private key
    """
    return generate_private_key().private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption())
