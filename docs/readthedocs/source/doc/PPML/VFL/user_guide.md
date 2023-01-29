# BigDL PPML VFL User Guide
## Deployment
### SGX
FL Server is protected by SGX, please see [PPML Prerequisite](https://github.com/intel-analytics/BigDL/blob/main/docs/readthedocs/source/doc/PPML/Overview/ppml.md#21-prerequisite) to get SGX environment ready.

### FL Server
You could set configurations of FL Server by editting `ppml-conf.yaml`
#### Configuration
##### clientNum
an integer, the total client number of this FL application
##### serverPort
an integer, the port used by FL Server
##### privateKeyFilePath
a string, the file path of TLS private key
##### certChainFilePath
a string, the file path of TLS certificate chain
#### Start
You can run FL Server in SGX with the following command:
```bash
docker exec -it YOUR_DOCKER bash /ppml/trusted-big-data-ml/work/start-scripts/start-python-fl-server-sgx.sh -p 8980 -c 2
```
You can also set port with `-p` and set client number with `-c` while the default settings are `port=8980` and `client-num=2`.

## Programming Guide
Once the FL Server deployment is ready, you can write the client code and start your FL application. 

You could see the [examples](overview.md#quick-start-examples) in overview for basic usages of the APIs.

You could check [API Doc]() for more details.
