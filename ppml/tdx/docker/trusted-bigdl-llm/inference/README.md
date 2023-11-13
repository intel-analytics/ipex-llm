## Build bigdl-trusted-bigdl-llm tdx image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-ppml-trusted-bigdl-llm-tdx:2.5.0-SNAPSHOT .
```

## Attestation on anolis-tdx

For workloads on kernel `5.10.134-anolis-tdx-ge3f6888f7855`, it's required to replace standard DCAP 1.16 version `libtdx_attest.so` because of differences in interfaces. You can modify the [code](https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/tdx_1.5_dcap_mvp_23q1/QuoteGeneration/quote_wrapper/tdx_attest/tdx_attest.c) according to the kernel's requirements (reference modification is provided at `./tdx_attest.diff`), and build your `libtdx_attest.so` refering to [DCAP documents](https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/tdx_1.5_dcap_mvp_23q1/QuoteGeneration#for-linux-os), or you can use the one we provided at `./libtdx_attest.so`

```bash
# In docker container
cp ./libtdx_attest.so /usr/lib/x86_64-linux-gnu/libtdx_attest.so.1.16.100.2
```