# HTTP Frontend Serving Guide

This guide introduce the usage of HTTP Serving

Currently, Both Inference Model and Cluster Serving are supported in the frontend serving. Multi-Model Serving is also supported.

1. Build a jar of HTTP serving, use command

```
mvn clean package
```

A jar package with name contains "http" will be generated.  

2. Start HTTP server

You can execute the jar to start the server.

```
java -jar analytics-zoo-bigdl_0.13.0-spark_3.1.2-0.12.0-SNAPSHOT-grpc.jar
```
Notice that the default path for servableManager Conf File is "./servables-conf.yaml" You can change it with 
"--servableManagerConfPath /home/test.yaml" options.
Attached is an example of the yaml file.

```
                ---
                 modelMetaDataList:
                 - !<ClusterServingMetaData>
                    modelName: "1"
                    modelVersion:"1.0"
                    redisHost: "localhost"
                    redisPort: "6381"
                    redisInputQueue: "serving_stream2"
                    redisOutputQueue: "cluster-serving_serving_stream2:"
                 - !<InflerenceModelMetaData>
                    modelName: "1"
                    modelVersion:"1.0"
                    modelPath:"/"
                    modelType:"OpenVINO"
                    features:
                      - "a"
                      - "b"
```


3. Test Performance

To test the performance of the http serving, you may use wrk tool to send requests and measure performance.
