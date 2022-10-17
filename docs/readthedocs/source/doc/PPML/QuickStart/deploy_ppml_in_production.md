# Deploy PPML Applicaiton in Production Environment

PPML applications are a little different from normal machine learning applications, especially when these applications are buit on Intel SGX. Because, Intel SGX requires applications to be signed by user specified key, i.e., `enclave-key`. That requirement separetes PPML deployment into 3 stages:
1. Test & Development
2. Build & sign applicaitons with enclave-key
3. Deployment applications


## Prerequisite

* BigDL PPML image, e.g., `intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene` or `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum`.
* Secured environment for signing applications & Build image. This environment has access to `enclave-key` and can build image based on BigDL PPML image.

## Test with PPML image

BigDL PPML provides 

[]()


## Sign applications & Build your image


[]()

## Deploy applicaitons in production environment

## References

1. Intel SGX
2. 