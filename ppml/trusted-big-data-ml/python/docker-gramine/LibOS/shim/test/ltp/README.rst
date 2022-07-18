LTP FOR GRAMINE
===============

Building
--------

* Install Gramine with tests (``-Dtests=enabled``).

* Download LTP sources: ``git submodule update --init src``.

* Run ``make``. This will compile LTP, then build all the manifests.

Running tests
-------------

The easiest is to run ``make regression``, or ``make SGX=1 regression``.

The tests use Pytest. To have more control over the test run, invoke Pytest
directly. For instance, use ``-k`` to run a single test by name::

    python3 -m pytest -v -k chmod01

Parallel execution
------------------

If you want to speed up the execution, you can use the ``pytest-xdist`` plugin::

    # Install the plugin
    apt install python3-pytest-xdist

    # Run with 2 processes
    python3 -m pytest -v -n 2

    # Run with one process per core
    python3 -m pytest -v -n auto

Note that running many tests in parallel is less stable: some of the tests are
resource-intensive, and might time out, and under SGX execution of concurrent
tests might fail due to limited EPC size.

Tips for debugging
------------------

To enable debugging output, edit a manifest for a test case (``*.manifest``) to
set ``loader.log_level = "trace"``. When running under SGX, you will need to
re-sign the manifest using ``gramine-test build``.

(NOTE: the manifests are all built from a single template,
``manifest.template``, but you can edit the generated manifests anyway. If
necessary, ``gramine-test build -f`` will rebuild the manifests from scratch).

You can also use GDB: ``GDB=1 gramine-{direct|sgx} <TEST_BINARY>``. You should
compile Gramine in debug mode so that you can see the symbols inside Gramine.

``ltp.cfg``
------------

Currently, we run only a subset of tests with Gramine. Many tests are skipped
because Gramine doesn't yet provide required functionality, or because they're
very Linux-specific. The exact configuration (i.e. which LTP tests are skipped)
is written in ``ltp.cfg``.

This is a file in ``.ini`` format
(https://docs.python.org/library/configparser). Lines starting with ``#`` and
``;`` are comments. Section names are names of the binaries. There is one
special section, ``[DEFAULT]``, which holds defaults for all binaries and some
global options.

Global options:
- ``sgx``: if true-ish, run under SGX (default: false)
- ``jobs``: run that many tests in parallel (default is 1 under SGX and number of
  CPUs otherwise); **WARNING:** Because EPC has limited size, test suite may
  become unstable if more than one test is running concurrently under SGX
- ``junit-classname``: classname to be shown in JUnit-XML report (``LTP``)
- ``ltproot``: path to LTP (default: ``./install``)

Per-binary options:
- ``skip``: if true-ish, do not attempt to run the binary (default: false).
- ``timeout`` in seconds (default: ``30``)
- ``must-pass``: if not specified (the default), treat the whole binary as
  a single test and report its return code; if specified, only those subtests
  (numbers separated by whitespace) are expected to pass, but they must be in
  the report, so if the binary TBROK'e earlier and did not run the test, report
  failure; empty ``must-pass`` is valid (means nothing is required); **NOTE**
  that this depends on stability of subtest numbering, which may be affected by
  various factors, among those the glibc version and/or what is ``#define``\ d
  in the headers (see ``signal03`` for example).

Another config file path can be specified by overriding the ``LTP_CONFIG``
environment variable. Run ``python3 test_ltp.py`` for more options.

A lot of LTP tests cause problems in Gramine. The ones we've already analyzed
should have an appropriate comment in the ``ltp.cfg`` file.

Running all the cases
---------------------

As explained above, we skip many LTP tests. In case you want to analyze all the
test results, including the tests that are currently skipped, you can use the
``ltp-all.cfg`` configuration.

* Build all the manifests: temporarily edit ``tests.toml`` to change ``ltp.cfg``
  to ``ltp-all.cfg``, and invoke ``gramine-test build``.

  (To save time, we build manifests only for those tests that are not skipped).

* Run Pytest with ``ltp-all.cfg`` as configuration::

    LTP_CONFIG=ltp-all.cfg python3 -m pytest -v --junit-xml=ltp-all.xml

The ``ltp-all.xml`` file should contain output for all tests.

SGX mode
--------

In SGX mode, we use additional files: ``ltp-sgx.cfg``, and (temporarily)
``ltp-bug-1075.cfg``. These function as an override for ``ltp.cfg``, so that
configuration is not duplicated.

Helper scripts (``contrib/``)
-----------------------------

The ``contrib/`` directory contains a few scripts for dealing with ``.cfg``
files. Except for ``conf_lint.py``, they are used for manual and one-off tasks.

* ``conf_lint.py``: Validate the configuration (check if it's sorted, look for
  outdated test names). Used in ``make regression``.

* ``conf_merge.py``: Merge two ``.cfg`` files. If there are duplicate section
  names, concatenate the sections.

* ``conf_missing.py``: Add missing sections to a ``.cfg`` file, so that it
  contains sections for all tests (based on an LTP scenario file with a list of
  tests).

* ``conf_remove_must_pass.py``: Remove all sections with ``must-pass``
  directive.

* ``conf_subtract.py``: Generate a difference between two files, i.e. output all
  sections that are in the second file but not in the first. This effectively
  converts a "full" configuration to an "override" one.
