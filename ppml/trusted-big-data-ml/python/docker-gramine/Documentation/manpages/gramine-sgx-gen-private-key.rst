.. program:: gramine-sgx-gen-private-key
.. _gramine-sgx-gen-private-key:

===================================================================
:program:`gramine-sgx-gen-private-key` -- Gramine SGX key generator
===================================================================

Synopsis
========

:command:`gramine-sgx-gen-private-key` [*OPTION*]... [*FILE*]

Description
===========

:program:`gramine-sgx-gen-private-key` is used to generate the private key
suitable for signing SGX enclaves. SGX requires RSA 3072 keys with public
exponent equal to 3.

The default path is ``$XDG_CONFIG_HOME/gramine/enclave-key.pem`` (which is
usually ``$HOME/.config/gramine/enclave-key.pem`` — see `XDG Base Directory
Specification
<https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html>`__
for details). You can specify an alternative path as an argument.

Command line arguments
======================

.. option:: --force, -f

   By default :program:`gramine-sgx-gen-private-key` will refuse to overwrite an
   existing file. Use this switch to force clobbering.

.. option:: --no-force

   Refuse to overwrite an existing file, which is the default. Use this switch
   to undo :option:`--force`.

.. option:: --verbose, -v

    Print details to standard output. This is the default.

.. option:: --quiet, -q

    Don't print details to standard output.
