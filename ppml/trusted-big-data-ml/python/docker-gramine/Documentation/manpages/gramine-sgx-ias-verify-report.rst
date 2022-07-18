.. program:: gramine-sgx-ias-verify-report

===================================================================================
:program:`gramine-sgx-ias-verify-report` -- Verify Intel Attestation Service report
===================================================================================

Synopsis
========

:command:`gramine-sgx-ias-verify-report` [*OPTION*]...

Description
===========

`gramine-sgx-ias-verify-report` verifies attestation report retrieved from the
Intel Attestation Service (using ``gramine-sgx-ias-request`` for example). It
also verifies that the quote contained in the IAS report contains expected
values.

Command line arguments
======================

.. option:: -h, --help

   Display usage.

.. option:: -v, --verbose

   Print more information.

.. option:: -m, --msb

   Print/parse hex strings in big-endian order.

.. option:: -r, --report-path

   IAS report to verify.

.. option:: -s, --sig-path

   Path to the IAS report's signature.

.. option:: -o, --allow-outdated-tcb

   Treat IAS status GROUP_OUT_OF_DATE as OK.

.. option:: -d, --allow-debug-enclave

   Allow debug enclave (SGXREPORT.ATTRIBUTES.DEBUG = 1).

.. option:: -n, --nonce

   Nonce that's expected in the report (optional).

.. option:: -S, --mr-signer

   Expected mr_signer field (hex string, optional).

.. option:: -E, --mr-enclave

   Expected mr_enclave field (hex string, optional).

.. option:: -R, --report-data

   Expected report_data field (hex string, optional).

.. option:: -P, --isv-prod-id

   Expected isv_prod_id field (hex string, optional).

.. option:: -V, --isv-svn

   Expected isv_svn field (hex string, optional).

.. option:: -i, --ias-pubkey

   Path to IAS public RSA key (PEM format, optional).

Example
=======

Report verification with all options enabled:

.. code-block:: sh

    $ gramine-sgx-ias-verify-report -v -m -r rp -s sp -i ias.pem -o -d -n thisisnonce -S 14b284525c45c4f526bf1535d05bd88aa73b9e184464f2d97be3dabc0d187b57 -E 4d69102c40401f40a54eb156601be73fb7605db0601845580f036fd284b7b303 -R 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004ba476e321e12c720000000000000001 -P 0 -V 0
    Verbose output enabled
    Endianness set to MSB
    Using IAS public key from file 'ias.pem'
    IAS key: RSA, 2048 bits
    Decoded IAS signature size: 256 bytes
    IAS report: signature verified correctly
    IAS report: allowing quote status GROUP_OUT_OF_DATE
    IAS report: nonce OK
    IAS report: quote decoded, size 432 bytes
    [...quote dump...]
    Quote: mr_signer OK
    Quote: mr_enclave OK
    Quote: isv_prod_id OK
    Quote: isv_svn OK
    Quote: report_data OK
    Quote: enclave attributes OK
