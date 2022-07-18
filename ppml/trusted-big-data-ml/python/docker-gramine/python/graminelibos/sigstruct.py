# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2021 Intel Corporation
#                    Borys Popławski <borysp@invisiblethingslab.com>

import struct

from . import _offsets as offs # pylint: disable=import-error,no-name-in-module


class Sigstruct:
    """Class for holding SGX SIGSTRUCT.

    Each field can be accessed and modified using ``[]`` operator. Accessing or setting an unknown
    key raises ``KeyError`` and setting a key to a value not matching required format raises
    ``ValueError``.
    """

    fields = {
        'header': (offs.SGX_ARCH_ENCLAVE_CSS_HEADER, '16s'),
        'module_vendor': (offs.SGX_ARCH_ENCLAVE_CSS_MODULE_VENDOR, '<L'),
        'date_year': (offs.SGX_ARCH_ENCLAVE_CSS_DATE, '<H'),
        'date_month': (offs.SGX_ARCH_ENCLAVE_CSS_DATE + 2, '<B'),
        'date_day': (offs.SGX_ARCH_ENCLAVE_CSS_DATE + 3, '<B'),
        'header2': (offs.SGX_ARCH_ENCLAVE_CSS_HEADER2, '16s'),
        'hw_version': (offs.SGX_ARCH_ENCLAVE_CSS_HW_VERSION, '<L'),
        'modulus': (offs.SGX_ARCH_ENCLAVE_CSS_MODULUS, '384s'),
        'exponent': (offs.SGX_ARCH_ENCLAVE_CSS_EXPONENT, '<L'),
        'signature': (offs.SGX_ARCH_ENCLAVE_CSS_SIGNATURE, '384s'),
        'misc_select': (offs.SGX_ARCH_ENCLAVE_CSS_MISC_SELECT, '<L'),
        'misc_mask': (offs.SGX_ARCH_ENCLAVE_CSS_MISC_MASK, '<L'),
        'attribute_flags': (offs.SGX_ARCH_ENCLAVE_CSS_ATTRIBUTES, '<Q'),
        'attribute_xfrms': (offs.SGX_ARCH_ENCLAVE_CSS_ATTRIBUTES + 8, '<Q'),
        'attribute_flags_mask': (offs.SGX_ARCH_ENCLAVE_CSS_ATTRIBUTE_MASK, '<Q'),
        'attribute_xfrm_mask': (offs.SGX_ARCH_ENCLAVE_CSS_ATTRIBUTE_MASK + 8, '<Q'),
        'enclave_hash': (offs.SGX_ARCH_ENCLAVE_CSS_ENCLAVE_HASH, '32s'),
        'isv_prod_id': (offs.SGX_ARCH_ENCLAVE_CSS_ISV_PROD_ID, '<H'),
        'isv_svn': (offs.SGX_ARCH_ENCLAVE_CSS_ISV_SVN, '<H'),
        'q1': (offs.SGX_ARCH_ENCLAVE_CSS_Q1, '384s'),
        'q2': (offs.SGX_ARCH_ENCLAVE_CSS_Q2, '384s'),
    }


    defaults = {
        'header': b'\x06\x00\x00\x00\xe1\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00',
        'module_vendor': 0,
        'header2': b'\x01\x01\x00\x00`\x00\x00\x00`\x00\x00\x00\x01\x00\x00\x00',
        'hw_version': 0,
        'misc_mask': offs.SGX_MISCSELECT_MASK_CONST,
        'attribute_flags_mask': offs.SGX_FLAGS_MASK_CONST,
        'attribute_xfrm_mask': offs.SGX_XFRM_MASK_CONST,
    }


    def __init__(self):
        self._data = {}
        for k, v in self.defaults.items():
            self._data[k] = v


    def __getitem__(self, key):
        return self._data[key]


    def __setitem__(self, key, val):
        try:
            struct.pack(self.fields[key][1], val)
        except KeyError:
            raise KeyError(f'unknown field name {key}')
        except struct.error:
            raise ValueError(f'{val} does not match required format {self.fields[key][1]}')

        self._data[key] = val


    def __contains__(self, key):
        return key in self._data


    def to_bytes(self, verify=True, verify_sig_fields=False):
        """Turn SIGSTRUCT into bytes.

        You probably want to set all required fields before calling this function. If you not wish
        to do so, you can pass :obj:`False` as *verify* argument - in such case non-set fields
        will be implicitly ``0``-initialized. Some fields have default values and therefore not need
        to be set (e.g. *header*).

        Args:
            verify (bool): if ``True``, check if all fields were initialized.
            verify_sig_fields (bool): if ``True``, don't skip signature fields in checks. Used only
                when `verify` is ``True``.

        Returns:
            bytearray: buffer containing byte representation of this SIGSTRUCT.

        Raises:
            KeyError: some SIGSTRUCT fields were not set.
        """
        buffer = bytearray(offs.SGX_ARCH_ENCLAVE_CSS_SIZE)
        for key, (offset, fmt) in self.fields.items():
            if key not in self:
                if verify and (key not in ['modulus', 'exponent', 'signature', 'q1', 'q2']
                               or verify_sig_fields):
                    raise KeyError(f'{key} is not set')
                continue
            struct.pack_into(fmt, buffer, offset, self[key])
        return buffer


    @classmethod
    def from_bytes(cls, buffer):
        """Load a SIGSTRUCT from bytes-like object.

        Input bytes must match the required SIGSTRUCT format.

        Args:
            buffer (bytes-like): buffer containing SIGSTRUCT.

        Returns:
            Sigstruct: parsed SIGSTRUCT object.

        Raises:
            TypeError: *buffer* has a wrong type (is not bytes or bytearray).
            ValueError: *buffer* does not have required length or one of the headers does not match.
        """
        if not isinstance(buffer, bytes) and not isinstance(buffer, bytearray):
            raise TypeError(f'a bytes-like object is required, not {type(buffer).__name__}')
        if len(buffer) != offs.SGX_ARCH_ENCLAVE_CSS_SIZE:
            raise ValueError(f'buffer len does not equal {offs.SGX_ARCH_ENCLAVE_CSS_SIZE}')

        sig = cls()

        for key, (offset, fmt) in cls.fields.items():
            sig[key] = struct.unpack_from(fmt, buffer, offset)[0]

        if sig['header'] != cls.defaults['header']:
            raise ValueError('header value does not mach')
        if sig['header2'] != cls.defaults['header2']:
            raise ValueError('header2 value does not mach')

        return sig


    def get_signing_data(self):
        data = self.to_bytes()
        after_sig_offset = offs.SGX_ARCH_ENCLAVE_CSS_MISC_SELECT
        return data[:128] + data[after_sig_offset:after_sig_offset+128]


    def sign(self, do_sign_callback, *args, **kwargs):
        """Sign the SIGSTRUCT.

        Signs this SIGSTRUCT in place (i.e. fills appropriate fields with the signing result) using
        *do_sign_callback* function.

        Args:
            do_sign_callback: Function doing actual signing. Gets to-be-signed data as the first
                argument (bytearray), then optional ``*args`` and ``**kwargs``. Must return a tuple
                ``(int, int, int)`` containing exponent, modulus and signature respectively.
            *args: Arbitrary arguments list passed to *do_sign_callback*.
            **kwargs:  Arbitrary keyword arguments passed to *do_sign_callback*.

        Raises:
            KeyError: some SIGSTRUCT fields were not set.
        """
        data = self.get_signing_data()

        exponent_int, modulus_int, signature_int = do_sign_callback(data, *args, **kwargs)

        tmp1 = signature_int * signature_int
        q1_int = tmp1 // modulus_int
        tmp2 = tmp1 % modulus_int
        q2_int = tmp2 * signature_int // modulus_int

        assert exponent_int == 3, "SGX requires RSA exponent to be 3"

        self['modulus'] = modulus_int.to_bytes(384, byteorder='little')
        self['exponent'] = exponent_int
        self['signature'] = signature_int.to_bytes(384, byteorder='little')
        self['q1'] = q1_int.to_bytes(384, byteorder='little')
        self['q2'] = q2_int.to_bytes(384, byteorder='little')
