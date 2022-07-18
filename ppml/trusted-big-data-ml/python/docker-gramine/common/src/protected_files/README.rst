===========================
Protected (Encrypted) Files
===========================

This directory contains the implementation of Protected Files (PF), a library
used for implementing *encrypted files* in Gramine. These files are encrypted on
disk and transparently decrypted when accessed by Gramine or by application
running inside Gramine.

Originally, the whole feature was called *protected files*, and was implemented
for SGX only. After moving it to LibOS, we updated the name:

* *encrypted files* is the name of the feature in Gramine,
* ``protected_files`` is the name of the platform-independent library used by
  that feature (i.e. this directory).

Features
========

- Data is encrypted (confidentiality) and integrity protected (tamper
  resistance).
- File swap protection (an encrypted file can only be accessed when in a
  specific path).
- Transparency (the application sees encrypted files as regular files, no need
  to modify the application).

Example
-------

::

   fs.mounts = [
     ...
     { type = "encrypted", path = "/some_file", uri = "file:tmp/some_file" },
     { type = "encrypted", path = "/some_dir", uri = "file:tmp/some_dir" },
     { type = "encrypted", path = "/another_file", uri = "file:another_dir/some_file" },
   ]

Gramine allows mounting files and directories as encrypted. If a directory is
mounted as encrypted, all existing files/directories within it are recursively
treated as encrypted.

See ``Documentation/manifest-syntax.rst`` for details.

Limitations
-----------

Metadata currently limits PF path size to 512 bytes and filename size to 260
bytes.

NOTE
----

The ``tools`` directory in Linux-SGX PAL contains the ``pf_crypt`` utility that
converts files to/from the protected format.

Internal protected file format in this version was ported from the `SGX SDK
<https://github.com/intel/linux-sgx/tree/1eaa4551d4b02677eec505684412dc288e6d6361/sdk/protected_fs>`_.

Tests
=====

Tests in ``LibOS/shim/test/fs`` contain encrypted file tests (``test_enc.py``).
Some tests in ``LibOS/shim/test/regression`` also work with encrypted files.

TODO
====

- Truncating protected files is not yet implemented.
- The recovery file feature is disabled, this needs to be discussed if it's
  needed in Gramine.
- Tests for invalid/malformed/corrupted files need to be ported to the new
  format.
