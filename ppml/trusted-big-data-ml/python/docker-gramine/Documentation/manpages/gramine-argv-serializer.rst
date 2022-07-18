.. _gramine-argv-serializer:

.. program:: gramine-argv-serializer

======================================================================
:program:`gramine-argv-serializer` -- Serialize command line arguments
======================================================================

Synopsis
========

:command:`gramine-argv-serializer` [*ARGS*]

Description
===========

`gramine-argv-serializer` serializes the command line arguments and prints it
to stdout. The result is intended to be redirected to a file that can
be used as a trusted or a protected file and ``loader.argv_src_file`` manifest
option pointed to it.
For more information on the usage, please refer to :doc:`../manifest-syntax`.

For an example on how to use this utility from Python, please refer to
:file:`LibOS/shim/test/regression/test_libos.py`.

Example
=======

.. code-block:: sh

   gramine-argv-serializer "binary_name" "arg1" "arg2" > gramine_args.txt

