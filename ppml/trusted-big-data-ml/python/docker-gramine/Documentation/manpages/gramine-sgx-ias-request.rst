.. program:: gramine-sgx-ias-request

==============================================================================
:program:`gramine-sgx-ias-request` -- Submit Intel Attestation Service request
==============================================================================

Synopsis
========

:command:`gramine-sgx-ias-request` *COMMAND* [*OPTION*]...

Description
===========

`gramine-sgx-ias-request` submits requests to Intel Attestation Service (IAS).
Possible commands are retrieving EPID signature revocation list and verifying
attestation evidence for an SGX enclave quote.

Command line arguments
======================

General options
---------------

.. option:: -h, --help

   Display usage.

.. option:: -v, --verbose

   Print more information.

.. option:: -m, --msb

   Print/parse hex strings in big-endian order.

.. option:: -k, --api-key

   IAS API key.

Commands
--------

.. option:: sigrl

   Retrieve signature revocation list for a given EPID group.

   Possible ``sigrl`` options:

   .. option:: -g, --gid

      EPID group id (hex string).

   .. option:: -i, --sigrl-path

      Path to save retrieved SigRL to.

   .. option:: -S, --sigrl-url

      URL for the IAS SigRL endpoint (optional).

.. option:: report

   Verify attestation evidence (quote).

   Possible ``report`` options:

   .. option:: -q, --quote-path

      Path to quote to submit.

   .. option:: -r, --report-path

      Path to save IAS report to.

   .. option:: -s, --sig-path

      Path to save IAS report's signature to.

   .. option:: -n, --nonce

      Nonce to use (optional).

   .. option:: -c, --cert-path

      Path to save IAS certificate to (optional).

   .. option:: -R, --report-url

      URL for the IAS attestation report endpoint (optional).

Examples
========

*SigRL* retrieval:

.. code-block:: sh

    $ gramine-sgx-ias-request sigrl -k $IAS_API_KEY -g ef0a0000 -i sigrl
    No SigRL for given EPID group ID ef0a0000

*Quote* verification:

.. code-block:: sh

    $ gramine-sgx-ias-request report -k $IAS_API_KEY -q gr.quote -r ias.report -s ias.sig -c ias.cert -v
    Verbose output enabled
    IAS request:
    {"isvEnclaveQuote":"AgABAO8..."}
    [...snip curl output...]
    IAS response: 200
    IAS report saved to: ias.report
    IAS report signature saved to: ias.sig
    IAS certificate saved to: ias.cert
    IAS submission successful

    $ cat ias.report
    {"id":"205146415611480061439763344693868541328","timestamp":"2020-03-20T10:48:32.353294","version":3,"epidPseudonym":"Itmg0 [...]","isvEnclaveQuoteStatus":"GROUP_OUT_OF_DATE" [...]}
