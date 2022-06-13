# FL Server

FL Server is a gRPC server handling the training requests from the federated client applications. Usually FL Server is expected to [run in SGX]().

The easiest way to start a FL Server with default config is
```bash
./ppml/start-fl-server.sh
```
## Run in SGX
TODO: add

## Configuration
To config FL Server, a `ppml-conf.yaml` should be provided in the directory where FL Server starts. The configurable parameters include following
```yaml
# the client number of the federated application
clientNum:
# TODO: add others sgx related
```