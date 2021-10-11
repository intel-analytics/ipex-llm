# gRPC Frontend Serving Guide

This guide introduce the usage of gRPC Serving


1. Build a jar of gRPC serving, use command

```
mvn clean package -Pspark_3.x -Pscala_2.12
```

A jar package with name contains "grpc" will be generated.  
Notice that spark 2.x is not supported.

2. Start gRPC server

Similar to http serving, you can execute the jar to start the server.

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

3. Setup the Client

Firstly, you need to have a grpc client. Attached is an example of define the class

```
class FrontEndGRPCClient {
  private var blockingStub: FrontEndGRPCServiceGrpc.FrontEndGRPCServiceBlockingStub = null
  /** Construct client for accessing HelloWorld server using the existing channel. */
  def this(channel: Channel) {
    this()
    blockingStub = FrontEndGRPCServiceGrpc.newBlockingStub(channel)
  }

  def ping(): Unit = {
    val request = Empty.newBuilder().build()
    val response = try {
      blockingStub.ping(request)
    }
    catch {
      case e: StatusRuntimeException =>
        println(Level.WARN, "RPC failed: {0}", e.getStatus)
        e.printStackTrace()
        return
    }
    println("succeed:" + response)
  }

  def getAllModels(): Unit = {
    val request = Empty.newBuilder.build
    val response = try {
      blockingStub.getAllModels(request)
    }
    catch {
      case e: StatusRuntimeException =>
        println(Level.WARN, "RPC failed: {0}", e.getStatus)
        e.printStackTrace()
        return
    }
    println("succeed:" + response)
  }
  /** get model with name. */
  def getModelsWithName(modelName: String): Unit = {
    val request = GetModelsWithNameReq.newBuilder.setModelName(modelName).build
    val response = try {
      blockingStub.getModelsWithName(request)
    }
    catch {
      case e: StatusRuntimeException =>
        println(Level.WARN, "RPC failed: {0}", e.getStatus)
        e.printStackTrace()
        return
    }
    println("succeed:" + response)
  }

  def getModelWithNameAndVersion(modelName: String, modelVersion: String): Unit = {
    val request = GetModelsWithNameAndVersionReq.newBuilder.setModelName(modelName).setModelVersion(modelVersion).build
    val response = try {
      blockingStub.getModelsWithNameAndVersion(request)
    }
    catch {
      case e: StatusRuntimeException =>
        println(Level.WARN, "RPC failed: {0}", e.getStatus)
        e.printStackTrace()
        return
    }
    println("succeed:" + response)
  }

  def predict(modelName: String, modelVersion: String, input: String): Unit = {
    val request = PredictReq.newBuilder().setModelName(modelName).setModelVersion(modelVersion).setInput(input).build()
    val response = try {
      blockingStub.predict(request)
    }
    catch {
      case e: StatusRuntimeException =>
        println(Level.WARN, "RPC failed: {0}", e.getStatus)
        Console.flush()
        e.printStackTrace()
        return
    }
//    println("succeed:" + response)
//    Console.flush()
  }
}
```
And also define the corresponds object with main function to execute as follows:
```
object FrontEndGRPCClient {

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var target = "localhost:8980"
    if (args.length > 1) target = args(1)
    val channel = ManagedChannelBuilder.forTarget(target).usePlaintext().asInstanceOf[ManagedChannelBuilder[ManagedChannelImplBuilder]].build()
    try {
      val client = new FrontEndGRPCClient(channel)
      client.ping()
      client.getAllModels()
      client.getModelsWithName("first-model")
      client.getModelWithNameAndVersion("first-model", "1.0")
      client.predict("first-model", "1.0", openvinoInput)
    } finally {
      channel.shutdownNow.awaitTermination(200, TimeUnit.SECONDS)
    }
  }
```

4. Test Performance

To test the performance of the grpc serving, simply use for loop in the client program.
You may also write a multi-thread client to test multi-client performance.
