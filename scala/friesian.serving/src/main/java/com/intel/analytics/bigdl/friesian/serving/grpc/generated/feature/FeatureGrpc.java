package com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: feature.proto")
public final class FeatureGrpc {

  private FeatureGrpc() {}

  public static final String SERVICE_NAME = "feature.Feature";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getGetUserFeaturesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getUserFeatures",
      requestType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getGetUserFeaturesMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs, com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getGetUserFeaturesMethod;
    if ((getGetUserFeaturesMethod = FeatureGrpc.getGetUserFeaturesMethod) == null) {
      synchronized (FeatureGrpc.class) {
        if ((getGetUserFeaturesMethod = FeatureGrpc.getGetUserFeaturesMethod) == null) {
          FeatureGrpc.getGetUserFeaturesMethod = getGetUserFeaturesMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs, com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getUserFeatures"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureMethodDescriptorSupplier("getUserFeatures"))
              .build();
        }
      }
    }
    return getGetUserFeaturesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getGetItemFeaturesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getItemFeatures",
      requestType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getGetItemFeaturesMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs, com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getGetItemFeaturesMethod;
    if ((getGetItemFeaturesMethod = FeatureGrpc.getGetItemFeaturesMethod) == null) {
      synchronized (FeatureGrpc.class) {
        if ((getGetItemFeaturesMethod = FeatureGrpc.getGetItemFeaturesMethod) == null) {
          FeatureGrpc.getGetItemFeaturesMethod = getGetItemFeaturesMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs, com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getItemFeatures"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureMethodDescriptorSupplier("getItemFeatures"))
              .build();
        }
      }
    }
    return getGetItemFeaturesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = FeatureGrpc.getGetMetricsMethod) == null) {
      synchronized (FeatureGrpc.class) {
        if ((getGetMetricsMethod = FeatureGrpc.getGetMetricsMethod) == null) {
          FeatureGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureMethodDescriptorSupplier("getMetrics"))
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
    if ((getResetMetricsMethod = FeatureGrpc.getResetMetricsMethod) == null) {
      synchronized (FeatureGrpc.class) {
        if ((getResetMetricsMethod = FeatureGrpc.getResetMetricsMethod) == null) {
          FeatureGrpc.getResetMetricsMethod = getResetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "resetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureMethodDescriptorSupplier("resetMetrics"))
              .build();
        }
      }
    }
    return getResetMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static FeatureStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FeatureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FeatureStub>() {
        @Override
        public FeatureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FeatureStub(channel, callOptions);
        }
      };
    return FeatureStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static FeatureBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FeatureBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FeatureBlockingStub>() {
        @Override
        public FeatureBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FeatureBlockingStub(channel, callOptions);
        }
      };
    return FeatureBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static FeatureFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FeatureFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FeatureFutureStub>() {
        @Override
        public FeatureFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FeatureFutureStub(channel, callOptions);
        }
      };
    return FeatureFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class FeatureImplBase implements io.grpc.BindableService {

    /**
     */
    public void getUserFeatures(com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetUserFeaturesMethod(), responseObserver);
    }

    /**
     */
    public void getItemFeatures(com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetItemFeaturesMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage> responseObserver) {
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
            getGetUserFeaturesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features>(
                  this, METHODID_GET_USER_FEATURES)))
          .addMethod(
            getGetItemFeaturesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features>(
                  this, METHODID_GET_ITEM_FEATURES)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage>(
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
  public static final class FeatureStub extends io.grpc.stub.AbstractAsyncStub<FeatureStub> {
    private FeatureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FeatureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FeatureStub(channel, callOptions);
    }

    /**
     */
    public void getUserFeatures(com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetUserFeaturesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getItemFeatures(com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetItemFeaturesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage> responseObserver) {
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
  public static final class FeatureBlockingStub extends io.grpc.stub.AbstractBlockingStub<FeatureBlockingStub> {
    private FeatureBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FeatureBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FeatureBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features getUserFeatures(com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetUserFeaturesMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features getItemFeatures(com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetItemFeaturesMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage getMetrics(com.google.protobuf.Empty request) {
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
  public static final class FeatureFutureStub extends io.grpc.stub.AbstractFutureStub<FeatureFutureStub> {
    private FeatureFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FeatureFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FeatureFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getUserFeatures(
        com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetUserFeaturesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features> getItemFeatures(
        com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetItemFeaturesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage> getMetrics(
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

  private static final int METHODID_GET_USER_FEATURES = 0;
  private static final int METHODID_GET_ITEM_FEATURES = 1;
  private static final int METHODID_GET_METRICS = 2;
  private static final int METHODID_RESET_METRICS = 3;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final FeatureImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(FeatureImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_GET_USER_FEATURES:
          serviceImpl.getUserFeatures((com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features>) responseObserver);
          break;
        case METHODID_GET_ITEM_FEATURES:
          serviceImpl.getItemFeatures((com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage>) responseObserver);
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

  private static abstract class FeatureBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    FeatureBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Feature");
    }
  }

  private static final class FeatureFileDescriptorSupplier
      extends FeatureBaseDescriptorSupplier {
    FeatureFileDescriptorSupplier() {}
  }

  private static final class FeatureMethodDescriptorSupplier
      extends FeatureBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    FeatureMethodDescriptorSupplier(String methodName) {
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
      synchronized (FeatureGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new FeatureFileDescriptorSupplier())
              .addMethod(getGetUserFeaturesMethod())
              .addMethod(getGetItemFeaturesMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getResetMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
