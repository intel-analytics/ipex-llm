### Quick Check
You can run `check-service.sh` to check whether the ppml services have been correctly initialized.

Before running `check-service.sh`, some environment variables should be exported like this:
```bash
export ATTESTATION_ADDRESS=...   # for attestation 
export BIGDL_KMS_IP=...          # for bigdl-kms
export EHSM_IP=...               # for ehsm
export KMS_UTILS_IP=...          # for kms utils 
export PCCS_IP=...               # for pccs
export PCCS_PORT=...             # for pccs
```
To test a specific component, pass one or more argument to it among the following: "attestation", "bigdl-kms", "ehsm", "kms-utils", "pccs".

For example, you can run the following command to check ehsm and pccs service.
```
bash check-service.sh ehsm pccs
```
To test all the services, you can either pass no auguments or pass the "all" argument.
```
bash check-service.sh all
```
if all is well, the following results should be displayed:
```
bigdl-attestation-service initialization successful.
bigdl-kms initialization successful.
ehsm initialization successful.
kms-utils initialization successful.
pccs initalization successful.
```