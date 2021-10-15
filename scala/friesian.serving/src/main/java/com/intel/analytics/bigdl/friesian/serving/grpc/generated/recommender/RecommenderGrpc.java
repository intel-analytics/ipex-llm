package com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: recommender.proto")
public final class RecommenderGrpc {

  private RecommenderGrpc() {}

  public static final String SERVICE_NAME = "recommender.Recommender";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs> getGetRecommendIDsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getRecommendIDs",
      requestType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs> getGetRecommendIDsMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs> getGetRecommendIDsMethod;
    if ((getGetRecommendIDsMethod = RecommenderGrpc.getGetRecommendIDsMethod) == null) {
      synchronized (RecommenderGrpc.class) {
        if ((getGetRecommendIDsMethod = RecommenderGrpc.getGetRecommendIDsMethod) == null) {
          RecommenderGrpc.getGetRecommendIDsMethod = getGetRecommendIDsMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getRecommendIDs"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs.getDefaultInstance()))
              .setSchemaDescriptor(new RecommenderMethodDescriptorSupplier("getRecommendIDs"))
              .build();
        }
      }
    }
    return getGetRecommendIDsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = RecommenderGrpc.getGetMetricsMethod) == null) {
      synchronized (RecommenderGrpc.class) {
        if ((getGetMetricsMethod = RecommenderGrpc.getGetMetricsMethod) == null) {
          RecommenderGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new RecommenderMethodDescriptorSupplier("getMetrics"))
              .build();
        }
      }
    }
    return getGetMetricsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.google.protobuf.Empty> getResetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "resetMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.google.protobuf.Empty> getResetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.google.protobuf.Empty> getResetMetricsMethod;
    if ((getResetMetricsMethod = RecommenderGrpc.getResetMetricsMethod) == null) {
      synchronized (RecommenderGrpc.class) {
        if ((getResetMetricsMethod = RecommenderGrpc.getResetMetricsMethod) == null) {
          RecommenderGrpc.getResetMetricsMethod = getResetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "resetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new RecommenderMethodDescriptorSupplier("resetMetrics"))
              .build();
        }
      }
    }
    return getResetMetricsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getGetClientMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getClientMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getGetClientMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getGetClientMetricsMethod;
    if ((getGetClientMetricsMethod = RecommenderGrpc.getGetClientMetricsMethod) == null) {
      synchronized (RecommenderGrpc.class) {
        if ((getGetClientMetricsMethod = RecommenderGrpc.getGetClientMetricsMethod) == null) {
          RecommenderGrpc.getGetClientMetricsMethod = getGetClientMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getClientMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new RecommenderMethodDescriptorSupplier("getClientMetrics"))
              .build();
        }
      }
    }
    return getGetClientMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static RecommenderStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecommenderStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecommenderStub>() {
        @java.lang.Override
        public RecommenderStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecommenderStub(channel, callOptions);
        }
      };
    return RecommenderStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static RecommenderBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecommenderBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecommenderBlockingStub>() {
        @java.lang.Override
        public RecommenderBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecommenderBlockingStub(channel, callOptions);
        }
      };
    return RecommenderBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static RecommenderFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecommenderFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecommenderFutureStub>() {
        @java.lang.Override
        public RecommenderFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecommenderFutureStub(channel, callOptions);
        }
      };
    return RecommenderFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class RecommenderImplBase implements io.grpc.BindableService {

    /**
     */
    public void getRecommendIDs(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetRecommendIDsMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMetricsMethod(), responseObserver);
    }

    /**
     */
    public void resetMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getResetMetricsMethod(), responseObserver);
    }

    /**
     */
    public void getClientMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetClientMetricsMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getGetRecommendIDsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs>(
                  this, METHODID_GET_RECOMMEND_IDS)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage>(
                  this, METHODID_GET_METRICS)))
          .addMethod(
            getResetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.google.protobuf.Empty>(
                  this, METHODID_RESET_METRICS)))
          .addMethod(
            getGetClientMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage>(
                  this, METHODID_GET_CLIENT_METRICS)))
          .build();
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RecommenderStub extends io.grpc.stub.AbstractAsyncStub<RecommenderStub> {
    private RecommenderStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected RecommenderStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecommenderStub(channel, callOptions);
    }

    /**
     */
    public void getRecommendIDs(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetRecommendIDsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void resetMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getResetMetricsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getClientMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetClientMetricsMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RecommenderBlockingStub extends io.grpc.stub.AbstractBlockingStub<RecommenderBlockingStub> {
    private RecommenderBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected RecommenderBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecommenderBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs getRecommendIDs(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetRecommendIDsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage getMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty resetMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getResetMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage getClientMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetClientMetricsMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RecommenderFutureStub extends io.grpc.stub.AbstractFutureStub<RecommenderFutureStub> {
    private RecommenderFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected RecommenderFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecommenderFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs> getRecommendIDs(
        com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetRecommendIDsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> resetMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getResetMetricsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage> getClientMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetClientMetricsMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_GET_RECOMMEND_IDS = 0;
  private static final int METHODID_GET_METRICS = 1;
  private static final int METHODID_RESET_METRICS = 2;
  private static final int METHODID_GET_CLIENT_METRICS = 3;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final RecommenderImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(RecommenderImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_GET_RECOMMEND_IDS:
          serviceImpl.getRecommendIDs((com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.RecommendIDProbs>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage>) responseObserver);
          break;
        case METHODID_RESET_METRICS:
          serviceImpl.resetMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_GET_CLIENT_METRICS:
          serviceImpl.getClientMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.ServerMessage>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class RecommenderBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    RecommenderBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Recommender");
    }
  }

  private static final class RecommenderFileDescriptorSupplier
      extends RecommenderBaseDescriptorSupplier {
    RecommenderFileDescriptorSupplier() {}
  }

  private static final class RecommenderMethodDescriptorSupplier
      extends RecommenderBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    RecommenderMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (RecommenderGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new RecommenderFileDescriptorSupplier())
              .addMethod(getGetRecommendIDsMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getResetMetricsMethod())
              .addMethod(getGetClientMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
