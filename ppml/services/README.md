##### trouble shooting 
You can run `check-service.sh` to check whether the ppml services have been correctly initialized.

To test a specific component, pass one or more argument to it among the following: "attestation", "bigDLKMS", "ehsm", "kmsUtils", "pccs".

For example, you can run the following command to check ehsm and pccs service.
```
bash check-service.sh ehsm pccs
```
To test all the services, you can either pass no auguments or pass the "all" argument.
```
bash check-service.sh all
```
if all is well, the following results should be displayed;
```
