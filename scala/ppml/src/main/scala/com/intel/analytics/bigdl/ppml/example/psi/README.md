# PSI(Private Set Intersection) Example

This example show how to create an end-to-end PSI application on BigDL PPML.

## Start FLServer
Start FLServer at server machine
```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.FLServer
```

## Start PSI Client
Change the config file to following. 
```
clientTarget: FLServer_URL
```
The port of server is provided in `ppml-conf.yaml` when server starts, default value `8980`. e.g. if you run the server and clients on same machine, `FLServer_URL` should be `localhost:8980`

```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.example.psi.PSIExample
```
