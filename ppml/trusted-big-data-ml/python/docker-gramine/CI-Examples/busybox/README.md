# Busybox

This directory contains the Makefile and the template manifest for the most
recent version of Busybox (as of this writing, commit ac78f2ac96).

# Quick Start

```sh
# build Busybox and the final manifest
make SGX=1

# run Busybox shell in non-SGX Gramine
gramine-direct busybox sh

# run Busybox shell in Gramine-SGX
gramine-sgx busybox sh

# now a shell session should be running e.g. typing:
ls
# should run program `ls` which lists current working directory
```
