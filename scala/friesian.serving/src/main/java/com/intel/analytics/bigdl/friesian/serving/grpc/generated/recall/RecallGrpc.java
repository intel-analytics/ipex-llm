package com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: recall.proto")
public final class RecallGrpc {

  private RecallGrpc() {}

  public static final String SERVICE_NAME = "recall.Recall";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates> getSearchCandidatesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "searchCandidates",
      requestType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates> getSearchCandidatesMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates> getSearchCandidatesMethod;
    if ((getSearchCandidatesMethod = RecallGrpc.getSearchCandidatesMethod) == null) {
      synchronized (RecallGrpc.class) {
        if ((getSearchCandidatesMethod = RecallGrpc.getSearchCandidatesMethod) == null) {
          RecallGrpc.getSearchCandidatesMethod = getSearchCandidatesMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "searchCandidates"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates.getDefaultInstance()))
              .setSchemaDescriptor(new RecallMethodDescriptorSupplier("searchCandidates"))
              .build();
        }
      }
    }
    return getSearchCandidatesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item,
      com.google.protobuf.Empty> getAddItemMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "addItem",
      requestType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item,
      com.google.protobuf.Empty> getAddItemMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item, com.google.protobuf.Empty> getAddItemMethod;
    if ((getAddItemMethod = RecallGrpc.getAddItemMethod) == null) {
      synchronized (RecallGrpc.class) {
        if ((getAddItemMethod = RecallGrpc.getAddItemMethod) == null) {
          RecallGrpc.getAddItemMethod = getAddItemMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "addItem"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new RecallMethodDescriptorSupplier("addItem"))
              .build();
        }
      }
    }
    return getAddItemMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = RecallGrpc.getGetMetricsMethod) == null) {
      synchronized (RecallGrpc.class) {
        if ((getGetMetricsMethod = RecallGrpc.getGetMetricsMethod) == null) {
          RecallGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new RecallMethodDescriptorSupplier("getMetrics"))
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
    if ((getResetMetricsMethod = RecallGrpc.getResetMetricsMethod) == null) {
      synchronized (RecallGrpc.class) {
        if ((getResetMetricsMethod = RecallGrpc.getResetMetricsMethod) == null) {
          RecallGrpc.getResetMetricsMethod = getResetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "resetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new RecallMethodDescriptorSupplier("resetMetrics"))
              .build();
        }
      }
    }
    return getResetMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static RecallStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecallStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecallStub>() {
        @Override
        public RecallStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecallStub(channel, callOptions);
        }
      };
    return RecallStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static RecallBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecallBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecallBlockingStub>() {
        @Override
        public RecallBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecallBlockingStub(channel, callOptions);
        }
      };
    return RecallBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static RecallFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecallFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecallFutureStub>() {
        @Override
        public RecallFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecallFutureStub(channel, callOptions);
        }
      };
    return RecallFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class RecallImplBase implements io.grpc.BindableService {

    /**
     */
    public void searchCandidates(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSearchCandidatesMethod(), responseObserver);
    }

    /**
     */
    public void addItem(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getAddItemMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage> responseObserver) {
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
            getSearchCandidatesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates>(
                  this, METHODID_SEARCH_CANDIDATES)))
          .addMethod(
            getAddItemMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item,
                com.google.protobuf.Empty>(
                  this, METHODID_ADD_ITEM)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage>(
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
  public static final class RecallStub extends io.grpc.stub.AbstractAsyncStub<RecallStub> {
    private RecallStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RecallStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecallStub(channel, callOptions);
    }

    /**
     */
    public void searchCandidates(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSearchCandidatesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void addItem(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getAddItemMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage> responseObserver) {
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
  public static final class RecallBlockingStub extends io.grpc.stub.AbstractBlockingStub<RecallBlockingStub> {
    private RecallBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RecallBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecallBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates searchCandidates(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSearchCandidatesMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty addItem(com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getAddItemMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage getMetrics(com.google.protobuf.Empty request) {
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
  public static final class RecallFutureStub extends io.grpc.stub.AbstractFutureStub<RecallFutureStub> {
    private RecallFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RecallFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecallFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates> searchCandidates(
        com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSearchCandidatesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> addItem(
        com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getAddItemMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage> getMetrics(
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

  private static final int METHODID_SEARCH_CANDIDATES = 0;
  private static final int METHODID_ADD_ITEM = 1;
  private static final int METHODID_GET_METRICS = 2;
  private static final int METHODID_RESET_METRICS = 3;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final RecallImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(RecallImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_SEARCH_CANDIDATES:
          serviceImpl.searchCandidates((com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates>) responseObserver);
          break;
        case METHODID_ADD_ITEM:
          serviceImpl.addItem((com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage>) responseObserver);
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

  private static abstract class RecallBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    RecallBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Recall");
    }
  }

  private static final class RecallFileDescriptorSupplier
      extends RecallBaseDescriptorSupplier {
    RecallFileDescriptorSupplier() {}
  }

  private static final class RecallMethodDescriptorSupplier
      extends RecallBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    RecallMethodDescriptorSupplier(String methodName) {
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
      synchronized (RecallGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new RecallFileDescriptorSupplier())
              .addMethod(getSearchCandidatesMethod())
              .addMethod(getAddItemMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getResetMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
