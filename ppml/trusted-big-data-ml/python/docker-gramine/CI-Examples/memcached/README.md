# Memcached

This directory contains the Makefile and the template manifest for the most
recent version of Memcached as of this writing (v1.5.21).

# Prerequisites

Please install `libevent-dev` package. If you want to benchmark with memcslap,
also install `libmemcached-tools`.

# Quick Start

```sh
# build Memcached and the final manifest
make SGX=1

# run original Memcached against a benchmark (memtier_benchmark,
# install the benchmark on your host OS first)
./memcached &
memtier_benchmark --port=11211 --protocol=memcache_binary --hide-histogram
killall memcached

# run Memcached in non-SGX Gramine against a benchmark
gramine-direct memcached -u nobody &
memtier_benchmark --port=11211 --protocol=memcache_binary --hide-histogram
killall pal-Linux

# run Memcached in Gramine-SGX against a benchmark
gramine-sgx memcached -u nobody &
memtier_benchmark --port=11211 --protocol=memcache_binary --hide-histogram
killall pal-Linux-SGX
```

# Why this Memcached configuration?

Notice that we run Memcached with `-u nobody` (means "user is nobody"). User
argument is required because Gramine currently emulates real/effective user ID
as 0 (root). This leads Memcached to believe it is run under root. For security
reasons, Memcached drops privileges and assumes non-privileged user ID which
must be specified as command-line argument. The assumed user ID is irrelevant
for consequent Memcached execution, so we use an existing host-OS username
"nobody" (this username is forwarded from the host because we mount host's
`/etc/passwd` file, see the manifest).
