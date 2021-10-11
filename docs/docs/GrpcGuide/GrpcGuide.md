# gRPC Guide
This guide introduces how to use a wrapped gRPC Server in Analytics Zoo to ease the grpc service development.

Users can write proto and Service, and pass Service to `ZooGrpcServer`

A simple HelloWorld example proto could be
```
syntax = "proto3";

option java_multiple_files = true;
option java_package = "io.grpc.examples.helloworld";
option java_outer_classname = "HelloWorldProto";
option objc_class_prefix = "HLW";

package helloworld;

// The greeting service definition.
service Greeter {
  // Sends a greeting
  rpc SayHello (HelloRequest) returns (HelloReply) {}
}

// The request message containing the user's name.
message HelloRequest {
  string name = 1;
}

// The response message containing the greetings
message HelloReply {
  string message = 1;
}
```
After `mvn clean install` with source code generated, you could write a Service
```
import com.intel.analytics.zoo.grpc.ZooGrpcServer
class GreeterImpl extends GreeterGrpc.GreeterImplBase {
  override def sayHello(req: HelloRequest, responseObserver: StreamObserver[HelloReply]): Unit = {
    val reply = HelloReply.newBuilder.setMessage("Hello " + req.getName).build
    responseObserver.onNext(reply)
    responseObserver.onCompleted()
  }
}
object GreeterImpl {
  /**
   * Main launches the server from the command line.
   */
  @throws[IOException]
  @throws[InterruptedException]
  def main(args: Array[String]): Unit = {
    val server = new ZooGrpcServer(new GreeterImpl)
    server.start
    server.blockUntilShutdown
  }
}
```
And the client could be
```
class HelloWorldClient {

  private val logger = Logger.getLogger(classOf[HelloWorldClient].getName)

  private var blockingStub: GreeterGrpc.GreeterBlockingStub = null

  /** Construct client for accessing HelloWorld server using the existing channel. */
  def this(channel: Channel) {
    this()
    // 'channel' here is a Channel, not a ManagedChannel, so it is not this code's responsibility to
    // shut it down.
    // Passing Channels to code makes code easier to test and makes it easier to reuse Channels.
    blockingStub = GreeterGrpc.newBlockingStub(channel)
  }

  /** Say hello to server. */
  def greet(name: String): Unit = {
    logger.info("Will try to greet " + name + " ...")
    val request = HelloRequest.newBuilder.setName(name).build
    val response = try {
      blockingStub.sayHello(request)
    }
    catch {
      case e: StatusRuntimeException =>
        logger.warn(Level.WARN, "RPC failed: {0}", e.getStatus)
        return
    }
    logger.info("Greeting: " + response.getMessage)
  }

  /**
   * Greet server. If provided, the first element of {@code args} is the name to use in the
   * greeting. The second argument is the target server.
   */


}
object HelloWorldClient {
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var user = "world"
    // Access a service running on the local machine on port 50051
    var target = "localhost:8980"
    // Allow passing in the user and target strings as command line arguments
    if (args.length > 0) {
      if ("--help" == args(0)) {
        System.err.println("Usage: [name [target]]")
        System.err.println("")
        System.err.println("  name    The name you wish to be greeted by. Defaults to " + user)
        System.err.println("  target  The server to connect to. Defaults to " + target)
        System.exit(1)
      }
      user = args(0)
    }
    if (args.length > 1) target = args(1)
    // Create a communication channel to the server, known as a Channel. Channels are thread-safe
    // and reusable. It is common to create channels at the beginning of your application and reuse
    // them until the application shuts down.
    val channel = ManagedChannelBuilder.forTarget(target).usePlaintext().asInstanceOf[ManagedChannelBuilder[ManagedChannelImplBuilder]].build()
    // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
    // needing certificates.

    try {
      val client = new HelloWorldClient(channel)
      client.greet(user)
    } finally {
      // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
      // resources the channel should be shut down when it will no longer be used. If it may be used
      // again leave it running.
      channel.shutdownNow.awaitTermination(5, TimeUnit.SECONDS)
    }
  }
}
```
After `HelloWorldServer` starts, you could run `HelloWorldClient` to complete this gRPC example.

The HelloWorld example is from official grpc-java exmaple with following license
```
// Copyright 2015 The gRPC Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```