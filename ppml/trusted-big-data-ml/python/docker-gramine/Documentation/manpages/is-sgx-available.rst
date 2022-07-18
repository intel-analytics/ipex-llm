.. program:: is-sgx-available

======================================================================
:program:`is-sgx-available` -- Check environment for SGX compatibility
======================================================================

Synopsis
========

:command:`is-sgx-available` [*OPTIONS*]

Description
===========

`is-sgx-available` checks the CPU and operating system for :term:`SGX`
compatibility. A detailed report is printed unless suppressed by the
:option:`--quiet` option.

Command line arguments
======================

.. option:: --quiet

   Suppress displaying detailed report. See exit status for the result.

Exit status
===========

0
   SGX version 1 is available and ready

1
   No CPU cupport

2
   No BIOS support

3
   SGX Platform Software (:term:`PSW`) is not installed

4
   The ``aesmd`` :term:`PSW` daemon is not installed

Example
=======

.. code-block:: sh

    $ is-sgx-available
    SGX supported by CPU: true
    SGX1 (ECREATE, EENTER, ...): true
    SGX2 (EAUG, EACCEPT, EMODPR, ...): false
    Flexible Launch Control (IA32_SGXPUBKEYHASH{0..3} MSRs): false
    SGX extensions for virtualizers (EINCVIRTCHILD, EDECVIRTCHILD, ESETCONTEXT): false
    Extensions for concurrent memory management (ETRACKC, ELDBC, ELDUC, ERDINFO): false
    CET enclave attributes support (See Table 37-5 in the SDM): false
    Key separation and sharing (KSS) support (CONFIGID, CONFIGSVN, ISVEXTPRODID, ISVFAMILYID report fields): false
    Max enclave size (32-bit): 0x80000000
    Max enclave size (64-bit): 0x1000000000
    EPC size: 0x5d80000
    SGX driver loaded: true
    AESMD installed: true
    SGX PSW/libsgx installed: true
