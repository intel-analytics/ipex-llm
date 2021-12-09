package com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: ranking.proto")
public final class RankingGrpc {

  private RankingGrpc() {}

  public static final String SERVICE_NAME = "ranking.Ranking";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<RankingProto.Content,
      RankingProto.Prediction> getDoPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "doPredict",
      requestType = RankingProto.Content.class,
      responseType = RankingProto.Prediction.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<RankingProto.Content,
      RankingProto.Prediction> getDoPredictMethod() {
    io.grpc.MethodDescriptor<RankingProto.Content, RankingProto.Prediction> getDoPredictMethod;
    if ((getDoPredictMethod = RankingGrpc.getDoPredictMethod) == null) {
      synchronized (RankingGrpc.class) {
        if ((getDoPredictMethod = RankingGrpc.getDoPredictMethod) == null) {
          RankingGrpc.getDoPredictMethod = getDoPredictMethod =
              io.grpc.MethodDescriptor.<RankingProto.Content, RankingProto.Prediction>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "doPredict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.Content.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.Prediction.getDefaultInstance()))
              .setSchemaDescriptor(new RankingMethodDescriptorSupplier("doPredict"))
              .build();
        }
      }
    }
    return getDoPredictMethod;
  }

  private static volatile io.grpc.MethodDescriptor<RankingProto.ModelPath,
      RankingProto.OperationStatus> getAddModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "addModel",
      requestType = RankingProto.ModelPath.class,
      responseType = RankingProto.OperationStatus.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<RankingProto.ModelPath,
      RankingProto.OperationStatus> getAddModelMethod() {
    io.grpc.MethodDescriptor<RankingProto.ModelPath, RankingProto.OperationStatus> getAddModelMethod;
    if ((getAddModelMethod = RankingGrpc.getAddModelMethod) == null) {
      synchronized (RankingGrpc.class) {
        if ((getAddModelMethod = RankingGrpc.getAddModelMethod) == null) {
          RankingGrpc.getAddModelMethod = getAddModelMethod =
              io.grpc.MethodDescriptor.<RankingProto.ModelPath, RankingProto.OperationStatus>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "addModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.ModelPath.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.OperationStatus.getDefaultInstance()))
              .setSchemaDescriptor(new RankingMethodDescriptorSupplier("addModel"))
              .build();
        }
      }
    }
    return getAddModelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<RankingProto.ModelPath,
      RankingProto.OperationStatus> getDeleteModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "deleteModel",
      requestType = RankingProto.ModelPath.class,
      responseType = RankingProto.OperationStatus.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<RankingProto.ModelPath,
      RankingProto.OperationStatus> getDeleteModelMethod() {
    io.grpc.MethodDescriptor<RankingProto.ModelPath, RankingProto.OperationStatus> getDeleteModelMethod;
    if ((getDeleteModelMethod = RankingGrpc.getDeleteModelMethod) == null) {
      synchronized (RankingGrpc.class) {
        if ((getDeleteModelMethod = RankingGrpc.getDeleteModelMethod) == null) {
          RankingGrpc.getDeleteModelMethod = getDeleteModelMethod =
              io.grpc.MethodDescriptor.<RankingProto.ModelPath, RankingProto.OperationStatus>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "deleteModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.ModelPath.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.OperationStatus.getDefaultInstance()))
              .setSchemaDescriptor(new RankingMethodDescriptorSupplier("deleteModel"))
              .build();
        }
      }
    }
    return getDeleteModelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      RankingProto.ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = RankingProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      RankingProto.ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, RankingProto.ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = RankingGrpc.getGetMetricsMethod) == null) {
      synchronized (RankingGrpc.class) {
        if ((getGetMetricsMethod = RankingGrpc.getGetMetricsMethod) == null) {
          RankingGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, RankingProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  RankingProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new RankingMethodDescriptorSupplier("getMetrics"))
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
    if ((getResetMetricsMethod = RankingGrpc.getResetMetricsMethod) == null) {
      synchronized (RankingGrpc.class) {
        if ((getResetMetricsMethod = RankingGrpc.getResetMetricsMethod) == null) {
          RankingGrpc.getResetMetricsMethod = getResetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "resetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new RankingMethodDescriptorSupplier("resetMetrics"))
              .build();
        }
      }
    }
    return getResetMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static RankingStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RankingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RankingStub>() {
        @Override
        public RankingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RankingStub(channel, callOptions);
        }
      };
    return RankingStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static RankingBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RankingBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RankingBlockingStub>() {
        @Override
        public RankingBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RankingBlockingStub(channel, callOptions);
        }
      };
    return RankingBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static RankingFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RankingFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RankingFutureStub>() {
        @Override
        public RankingFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RankingFutureStub(channel, callOptions);
        }
      };
    return RankingFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class RankingImplBase implements io.grpc.BindableService {

    /**
     */
    public void doPredict(RankingProto.Content request,
        io.grpc.stub.StreamObserver<RankingProto.Prediction> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDoPredictMethod(), responseObserver);
    }

    /**
     */
    public void addModel(RankingProto.ModelPath request,
        io.grpc.stub.StreamObserver<RankingProto.OperationStatus> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getAddModelMethod(), responseObserver);
    }

    /**
     */
    public void deleteModel(RankingProto.ModelPath request,
        io.grpc.stub.StreamObserver<RankingProto.OperationStatus> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDeleteModelMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<RankingProto.ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMetricsMethod(), responseObserver);
    }

    /**
     */
    public void resetMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getResetMetricsMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getDoPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                RankingProto.Content,
                RankingProto.Prediction>(
                  this, METHODID_DO_PREDICT)))
          .addMethod(
            getAddModelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                RankingProto.ModelPath,
                RankingProto.OperationStatus>(
                  this, METHODID_ADD_MODEL)))
          .addMethod(
            getDeleteModelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                RankingProto.ModelPath,
                RankingProto.OperationStatus>(
                  this, METHODID_DELETE_MODEL)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                RankingProto.ServerMessage>(
                  this, METHODID_GET_METRICS)))
          .addMethod(
            getResetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.google.protobuf.Empty>(
                  this, METHODID_RESET_METRICS)))
          .build();
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RankingStub extends io.grpc.stub.AbstractAsyncStub<RankingStub> {
    private RankingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RankingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RankingStub(channel, callOptions);
    }

    /**
     */
    public void doPredict(RankingProto.Content request,
        io.grpc.stub.StreamObserver<RankingProto.Prediction> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDoPredictMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void addModel(RankingProto.ModelPath request,
        io.grpc.stub.StreamObserver<RankingProto.OperationStatus> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getAddModelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void deleteModel(RankingProto.ModelPath request,
        io.grpc.stub.StreamObserver<RankingProto.OperationStatus> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDeleteModelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<RankingProto.ServerMessage> responseObserver) {
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
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RankingBlockingStub extends io.grpc.stub.AbstractBlockingStub<RankingBlockingStub> {
    private RankingBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RankingBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RankingBlockingStub(channel, callOptions);
    }

    /**
     */
    public RankingProto.Prediction doPredict(RankingProto.Content request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDoPredictMethod(), getCallOptions(), request);
    }

    /**
     */
    public RankingProto.OperationStatus addModel(RankingProto.ModelPath request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getAddModelMethod(), getCallOptions(), request);
    }

    /**
     */
    public RankingProto.OperationStatus deleteModel(RankingProto.ModelPath request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDeleteModelMethod(), getCallOptions(), request);
    }

    /**
     */
    public RankingProto.ServerMessage getMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty resetMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getResetMetricsMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RankingFutureStub extends io.grpc.stub.AbstractFutureStub<RankingFutureStub> {
    private RankingFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RankingFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RankingFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<RankingProto.Prediction> doPredict(
        RankingProto.Content request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDoPredictMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<RankingProto.OperationStatus> addModel(
        RankingProto.ModelPath request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getAddModelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<RankingProto.OperationStatus> deleteModel(
        RankingProto.ModelPath request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDeleteModelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<RankingProto.ServerMessage> getMetrics(
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
  }

  private static final int METHODID_DO_PREDICT = 0;
  private static final int METHODID_ADD_MODEL = 1;
  private static final int METHODID_DELETE_MODEL = 2;
  private static final int METHODID_GET_METRICS = 3;
  private static final int METHODID_RESET_METRICS = 4;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final RankingImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(RankingImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_DO_PREDICT:
          serviceImpl.doPredict((RankingProto.Content) request,
              (io.grpc.stub.StreamObserver<RankingProto.Prediction>) responseObserver);
          break;
        case METHODID_ADD_MODEL:
          serviceImpl.addModel((RankingProto.ModelPath) request,
              (io.grpc.stub.StreamObserver<RankingProto.OperationStatus>) responseObserver);
          break;
        case METHODID_DELETE_MODEL:
          serviceImpl.deleteModel((RankingProto.ModelPath) request,
              (io.grpc.stub.StreamObserver<RankingProto.OperationStatus>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<RankingProto.ServerMessage>) responseObserver);
          break;
        case METHODID_RESET_METRICS:
          serviceImpl.resetMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @Override
    @SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class RankingBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    RankingBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return RankingProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Ranking");
    }
  }

  private static final class RankingFileDescriptorSupplier
      extends RankingBaseDescriptorSupplier {
    RankingFileDescriptorSupplier() {}
  }

  private static final class RankingMethodDescriptorSupplier
      extends RankingBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    RankingMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (RankingGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new RankingFileDescriptorSupplier())
              .addMethod(getDoPredictMethod())
              .addMethod(getAddModelMethod())
              .addMethod(getDeleteModelMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getResetMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
