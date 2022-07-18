# RA-TLS Minimal Example

This directory contains the Makefile, the template server manifest, and the
minimal server and client written against the mbedTLS library.

The server and client are based on `ssl_server.c` and `ssl_client1.c` example
programs shipped with mbedTLS. We modified them to allow using RA-TLS flows if
the programs are given the command-line argument `epid`/`dcap`.  In particular,
the server uses a self-signed RA-TLS cert with the SGX quote embedded in it via
`ra_tls_create_key_and_crt()`. The client uses an RA-TLS verification callback
to verify the server RA-TLS certificate via `ra_tls_verify_callback()`.

This example uses the RA-TLS libraries `ra_tls_attest.so` for server and
`ra_tls_verify_epid.so`/ `ra_tls_verify_dcap.so` for client. These libraries are
installed together with Gramine (for DCAP version, you need `meson setup ...
-Ddcap=enabled`). Additionally, mbedTLS libraries are required to correctly run
RA-TLS, the client, and the server. For ECDSA/DCAP attestation, the DCAP
software infrastructure must be installed and work correctly on the host.

The current example works with both EPID (IAS) and ECDSA (DCAP) remote
attestation schemes. For more documentation, refer to
https://gramine.readthedocs.io/en/latest/attestation.html.

## RA-TLS server

The server is supposed to run in the SGX enclave with Gramine and RA-TLS
dlopen-loaded. If RA-TLS library `ra_tls_attest.so` is not requested by user via
`epid`/`dcap` command-line argument, the server falls back to using normal X.509
PKI flows (specified as `native` command-line argument).

If server is run with more command-line arguments (the only important thing is
to have at least one additional argument), then the server will maliciously
modify the SGX quote before sending to the client. This is useful for testing
purposes.

## RA-TLS client

The client is supposed to run on a trusted machine (*not* in an SGX enclave). If
RA-TLS library `ra_tls_verify_epid.so` or `ra_tls_verify_dcap.so` is not
requested by user via `epid` or `dcap` command-line arguments respectively, the
client falls back to using normal X.509 PKI flows (specified as `native`
command-line argument).

It is also possible to run the client in an SGX enclave. This will create a
secure channel between two Gramine SGX processes, possibly running on different
machines. It can be used as an example of in-enclave remote attestation and
verification.

If client is run without additional command-line arguments, it uses default
RA-TLS verification callback that compares `MRENCLAVE`, `MRSIGNER`,
`ISV_PROD_ID` and `ISV_SVN` against the corresonding `RA_TLS_*` environment
variables. To run the client with its own verification callback, execute it with
four additional command-line arguments (see the source code for details).

Also, because this example builds and uses debug SGX enclaves (`sgx.debug` is
set to `true`), we use environment variable `RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1`.
Note that in production environments, you must *not* use this option!

Moreover, we set `RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1`, to allow performing
the tests when some of Intel's security advisories haven't been addressed (for
example, when the microcode or architectural enclaves aren't fully up-to-date).
As the name of this setting suggests, this is not secure and likewise should not
be used in production.

# Quick Start

In most of the examples below, you need to tell the client what values it should
expect for `MRENCLAVE`, `MRSIGNER`, `ISV_PROD_ID` and `ISV_SVN`. One way to
obtain them is to note them down when they get printed by `gramine-sgx-get-token`
when generating `server.token`.

Moreover, for EPID-based (IAS) attestation, you will need to provide
an [SPID and the corresponding IAS API keys][spid].

[spid]: https://gramine.readthedocs.io/en/latest/sgx-intro.html#term-spid

- Normal non-RA-TLS flows; without SGX and without Gramine:

```sh
make app
./server native &
./client native
# client will successfully connect to the server via normal x.509 PKI flows
kill %%
```

- RA-TLS flows with SGX and with Gramine, EPID-based (IAS) attestation:

```sh
make clean
RA_CLIENT_SPID=<your SPID> RA_CLIENT_LINKABLE=<1 if SPID is linkable, else 0> make app epid

gramine-sgx ./server epid &

RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1 \
RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1 \
RA_TLS_EPID_API_KEY=<your EPID API key> \
RA_TLS_MRENCLAVE=<MRENCLAVE of the server enclave> \
RA_TLS_MRSIGNER=<MRSIGNER of the server enclave> \
RA_TLS_ISV_PROD_ID=<ISV_PROD_ID of the server enclave> \
RA_TLS_ISV_SVN=<ISV_SVN of the server enclave> \
./client epid

# client will successfully connect to the server via RA-TLS/EPID flows
kill %%
```

- RA-TLS flows with SGX and with Gramine, ECDSA-based (DCAP) attestation:

```sh
make clean
make app dcap

gramine-sgx ./server dcap &

RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1 \
RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1 \
RA_TLS_MRENCLAVE=<MRENCLAVE of the server enclave> \
RA_TLS_MRSIGNER=<MRSIGNER of the server enclave> \
RA_TLS_ISV_PROD_ID=<ISV_PROD_ID of the server enclave> \
RA_TLS_ISV_SVN=<ISV_SVN of the server enclave> \
./client dcap

# client will successfully connect to the server via RA-TLS/DCAP flows
kill %%
```

- RA-TLS flows with SGX and with Gramine, client with its own verification callback:

```sh
make clean
make app dcap

gramine-sgx ./server dcap &

RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1 RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1 ./client dcap \
    <MRENCLAVE of the server enclave> \
    <MRSIGNER of the server enclave> \
    <ISV_PROD_ID of the server enclave> \
    <ISV_SVN of the server enclave>

# client will successfully connect to the server via RA-TLS/DCAP flows
kill %%
```

- RA-TLS flows with SGX and with Gramine, server sends malicious SGX quote:

```sh
make clean
make app dcap

gramine-sgx ./server dcap dummy-option &
./client dcap

# client will fail to verify the malicious SGX quote and will *not* connect to the server
kill %%
```

- RA-TLS flows with SGX and with Gramine, running EPID client in SGX:

Note: you may also add environment variables to `client.manifest.template`, such
as `RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE`, `RA_TLS_ALLOW_OUTDATED_TCB_INSECURE`,
`RA_TLS_MRENCLAVE`, `RA_TLS_MRSIGNER`, `RA_TLS_ISV_PROD_ID` and
`RA_TLS_ISV_SVN`.

```sh
make clean
RA_CLIENT_SPID=<your SPID> RA_CLIENT_LINKABLE=<1 if SPID is linkable, else 0> make app client_epid.manifest.sgx

gramine-sgx ./server epid &

RA_TLS_EPID_API_KEY=<your EPID API key> gramine-sgx ./client_epid epid

# client will successfully connect to the server via RA-TLS/EPID flows
kill %%
```

- RA-TLS flows with SGX and with Gramine, running DCAP client in SGX:

```sh
make clean
make app dcap

gramine-sgx ./server dcap &

RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1 RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1 gramine-sgx \
    ./client_dcap dcap \
    <MRENCLAVE of the server enclave> \
    <MRSIGNER of the server enclave> \
    <ISV_PROD_ID of the server enclave> \
    <ISV_SVN of the server enclave>

# client will successfully connect to the server via RA-TLS/DCAP flows
kill %%
```
