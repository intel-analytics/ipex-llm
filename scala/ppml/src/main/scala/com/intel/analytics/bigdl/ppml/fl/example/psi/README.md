# PSI(Private Set Intersection) Example

This example show how to create an end-to-end PSI application on BigDL PPML.

## Start FLServer
Start FLServer at server machine
```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.fl.FLServer
```

## Start PSI Client
On the same machine, start client by

```
java -cp bigdl-assembly-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.psi.PSIExample
```
