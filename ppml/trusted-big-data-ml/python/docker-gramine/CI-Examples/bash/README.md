# Bash example

This directory contains an example for running Bash in Gramine, including
the Makefile and a template for generating the manifest.

# Generating the manifest

## Building for Linux

Run `make` (non-debug) or `make DEBUG=1` (debug) in the directory.

## Building for SGX

Run `make SGX=1` (non-debug) or `make SGX=1 DEBUG=1` (debug) in the directory.

# Running Bash with Gramine

Here's an example of running Bash scripts under Gramine:

Without SGX:
```
gramine-direct ./bash -c "ls"
gramine-direct ./bash -c "cd scripts && bash bash_test.sh 2"
```

With SGX:
```
gramine-sgx ./bash -c "ls"
gramine-sgx ./bash -c "cd scripts && bash bash_test.sh 2"
```
