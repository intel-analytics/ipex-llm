# Secret Provisioning Minimal Examples

This directory contains the Makefile, the template client manifests, and the
minimal server and clients written against the Secret Provisioning library.

This example uses the Secret Provisioning libraries `secret_prov_attest.so` for
clients and `secret_prov_verify_epid.so`/`secret_prov_verify_dcap.so` for
server. These libraries are installed together with Gramine (for DCAP version,
you need `meson setup ... -Ddcap=enabled`). Additionally, mbedTLS libraries are
required. For ECDSA/DCAP attestation, the DCAP software infrastructure must be
installed and work correctly on the host.

The current example works with both EPID (IAS) and ECDSA (DCAP) remote
attestation schemes. For more documentation, refer to
https://gramine.readthedocs.io/en/latest/attestation.html.

## Secret Provisioning server

The server is supposed to run on a trusted machine (not in the SGX enclave). The
server listens for client connections. For each connected client, the server
verifies the client's RA-TLS certificate and the embedded SGX quote and, if
verification succeeds, sends the first secret back to the client (the master key
for encrypted files, read from `files/wrap-key`). If the client requests a
second secret, the server sends the dummy string `42` as the second secret.

There are two versions of the server: the EPID one and the DCAP one. Each of
them links against the corresponding EPID/DCAP secret-provisioning library at
build time.

Because this example builds and uses debug SGX enclaves (`sgx.debug` is set
to `true`), we use environment variable `RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1`.
Note that in production environments,
you must *not* use this option!

Moreover, we set `RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1`, to allow performing
the tests when some of Intel's security advisories haven't been addressed (for
example, when the microcode or architectural enclaves aren't fully up-to-date).
As the name of this setting suggests, this is not secure and likewise should not
be used in production.

## Secret Provisioning clients

There are three clients in this example:

1. Minimal client. It relies on constructor-time secret provisioning and gets
   the first (and only) secret from the environment variable
   `SECRET_PROVISION_SECRET_STRING`.
2. Feature-rich client. It uses a programmatic C API to get two secrets from the
   server.
3. Encrypted-files client. Similarly to the minimal client, it relies on
   constructor-time secret provisioning and instructs Gramine to consider the
   provisioned secret as the encryption key for the Encrypted Files feature.
   After the master key is applied, the client reads an encrypted file
   `files/input.txt`.

As part of secret provisioning flow, all clients create a self-signed RA-TLS
certificate with the embedded SGX quote, send it to the server for verification,
and expect secrets in return.

The minimal and the encrypted-files clients rely on the `LD_PRELOAD` trick that
preloads `libsecret_prov_attest.so` and runs it before the clients' main logic.
The feature-rich client links against `libsecret_prov_attest.so` explicitly at
build time.

# Quick Start

- Secret Provisioning flows, EPID-based (IAS) attestation (you will need to
  provide an [SPID and the corresponding IAS API keys][spid]):

[spid]: https://gramine.readthedocs.io/en/latest/sgx-intro.html#term-spid

```sh
RA_CLIENT_SPID=<your SPID> RA_CLIENT_LINKABLE=<1 if SPID is linkable, else 0> make app epid files/input.txt

RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1 \
RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1 \
RA_TLS_EPID_API_KEY=<your EPID API key> \
./secret_prov_server_epid &

# test minimal client
gramine-sgx ./secret_prov_min_client

# test feature-rich client
gramine-sgx ./secret_prov_client

# test encrypted-files client
gramine-sgx ./secret_prov_pf_client

kill %%
```

- Secret Provisioning flows, ECDSA-based (DCAP) attestation:

```sh
make app dcap files/input.txt

RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1 \
RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1 \
./secret_prov_server_dcap &

# test minimal client
gramine-sgx ./secret_prov_min_client

# test feature-rich client
gramine-sgx ./secret_prov_client

# test encrypted-files client
gramine-sgx ./secret_prov_pf_client

kill %%
```
