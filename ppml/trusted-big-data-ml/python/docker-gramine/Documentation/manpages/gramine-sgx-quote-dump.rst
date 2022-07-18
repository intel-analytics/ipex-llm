.. program:: gramine-sgx-quote-dump

================================================================
:program:`gramine-sgx-quote-dump` -- Display SGX quote structure
================================================================

Synopsis
========

:command:`gramine-sgx-quote-dump` [*OPTION*]... *FILE*

Description
===========

`gramine-sgx-quote-dump` displays internal structure of an :term:`SGX` quote
contained in *FILE*.

Command line arguments
======================

.. option:: -h, --help

   Display usage.

.. option:: -m, --msb

   Print hex strings in big-endian order.

Example
=======

.. code-block:: sh

    $ gramine-sgx-quote-dump -m gr.quote
    version           : 0002
    sign_type         : 0001
    epid_group_id     : 00000aef
    qe_svn            : 0007
    pce_svn           : 0006
    xeid              : 00000000
    basename          : 0000000000000000000000000000000094b929a21f249e5eccb9a5fa33fa5a65
    report_body       :
     cpu_svn          : 000000000000000000000201ffff0e08
     misc_select      : 00000000
     reserved1        : 000000000000000000000000
     isv_ext_prod_id  : 00000000000000000000000000000000
     attributes.flags : 0000000000000007
     attributes.xfrm  : 000000000000001f
     mr_enclave       : 4d69102c40401f40a54eb156601be73fb7605db0601845580f036fd284b7b303
     reserved2        : 0000000000000000000000000000000000000000000000000000000000000000
     mr_signer        : 14b284525c45c4f526bf1535d05bd88aa73b9e184464f2d97be3dabc0d187b57
     reserved3        : 0000000000000000000000000000000000000000000000000000000000000000
     config_id        : 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
     isv_prod_id      : 0000
     isv_svn          : 0000
     config_svn       : 0000
     reserved4        : 000000000000000000000000000000000000000000000000000000000000000000000000000000000000
     isv_family_id    : 00000000000000000000000000000000
     report_data      : 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004ba476e321e12c720000000000000001
    signature_len     : 680 (0x2a8)
    signature         : 2e13c6a3...
