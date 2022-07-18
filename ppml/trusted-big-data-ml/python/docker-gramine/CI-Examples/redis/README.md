# Redis

This directory contains the Makefile and the template manifest for the most
recent version of Redis (as of this writing, version 6.0.5).

The Makefile and the template manifest contain extensive comments and are made
self-explanatory. Please review them to gain understanding of Gramine-SGX and
requirements for applications running under Gramine-SGX. If you want to
contribute a new example to Gramine and you take this Redis example as a
template, we recommend to remove the comments from your copies as they only add
noise (see e.g. Memcached for a "stripped-down" example).


# Quick Start

```sh
# build Redis and the final manifest
make SGX=1

# run original Redis against a benchmark (redis-benchmark supplied with Redis)
./redis-server --save '' --protected-mode no &
src/src/redis-benchmark
kill %%

# run Redis in non-SGX Gramine against a benchmark
gramine-direct redis-server --save '' --protected-mode no &
src/src/redis-benchmark
kill %%

# run Redis in Gramine-SGX against a benchmark
gramine-sgx redis-server --save '' --protected-mode no &
src/src/redis-benchmark
kill %%
```

# Why this Redis configuration?

Notice that we run Redis with two parameters: `save ''` and `protected-mode no`:

- `save ''` disables saving DB to disk (both RDB snapshots and AOF logs). We use
  this parameter to side-step some bugs in Gramine triggered during the
  graceful shutdown of Redis.

- `protected-mode no` allows clients to connect to Redis on any network
  interface. Even though we use the loopback interface (which is always allowed
  in Redis), Gramine hides this information. Therefore, we ask Redis to allow
  clients on all interfaces.

# Redis with Select

By default, Redis uses the epoll mechanism of Linux to monitor client
connections.  To test Redis with select, add `USE_SELECT=1`, e.g., `make SGX=1
USE_SELECT=1`.
