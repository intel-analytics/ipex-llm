# SQLite example

This directory contains an example for running SQLite in Gramine, including the
Makefile and a template for generating the manifest.

# Generating the manifest

## Installing prerequisite:

Please run the following command to install SQLite (Ubuntu-specific):

    sudo apt-get install sqlite3

## Building for Linux

Run `make` (non-debug) or `make DEBUG=1` (debug) in the directory.

## Building for SGX

Run `make SGX=1` (non-debug) or `make SGX=1 DEBUG=1` (debug) in the directory.

# Running SQLite with Gramine

Here's an example of running SQLite under Gramine:

Without SGX:
```
gramine-direct sqlite3 scripts/testdir/test.db < scripts/create.sql
gramine-direct sqlite3 scripts/testdir/test.db < scripts/update.sql
gramine-direct sqlite3 scripts/testdir/test.db < scripts/select.sql
```

With SGX:
```
gramine-sgx sqlite3 scripts/testdir/test.db < scripts/create.sql
gramine-sgx sqlite3 scripts/testdir/test.db < scripts/update.sql
gramine-sgx sqlite3 scripts/testdir/test.db < scripts/select.sql
```

# Note about concurrency

SQLite uses POSIX record locks (`fcntl`) to guard concurrent accesses to the
database file. These locks are emulated within Gramine, and not translated to
host-level locks, even if you are mounting a file from the host.

That means it is safe to access the same database file from multiple processes,
but only within a **single Gramine instance**. In other words, a multi-process
Gramine application is OK, but multiple Gramine instances should not access
the same database file concurrently.

Note that in a production setup, the database should be either mounted as a
protected file, or from tmpfs, which would make it impossible to access from
multiple Gramine instances anyway.
