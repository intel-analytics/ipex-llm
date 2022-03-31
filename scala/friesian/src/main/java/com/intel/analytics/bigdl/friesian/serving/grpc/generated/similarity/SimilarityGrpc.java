package com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: similarity.proto")
public final class SimilarityGrpc {

  private SimilarityGrpc() {}

  public static final String SERVICE_NAME = "similarity.Similarity";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<SimilarityProto.IDs,
      SimilarityProto.ItemNeighbors> getGetItemNeighborsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getItemNeighbors",
      requestType = SimilarityProto.IDs.class,
      responseType = SimilarityProto.ItemNeighbors.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<SimilarityProto.IDs,
      SimilarityProto.ItemNeighbors> getGetItemNeighborsMethod() {
    io.grpc.MethodDescriptor<SimilarityProto.IDs, SimilarityProto.ItemNeighbors> getGetItemNeighborsMethod;
    if ((getGetItemNeighborsMethod = SimilarityGrpc.getGetItemNeighborsMethod) == null) {
      synchronized (SimilarityGrpc.class) {
        if ((getGetItemNeighborsMethod = SimilarityGrpc.getGetItemNeighborsMethod) == null) {
          SimilarityGrpc.getGetItemNeighborsMethod = getGetItemNeighborsMethod =
              io.grpc.MethodDescriptor.<SimilarityProto.IDs, SimilarityProto.ItemNeighbors>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getItemNeighbors"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  SimilarityProto.IDs.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  SimilarityProto.ItemNeighbors.getDefaultInstance()))
              .setSchemaDescriptor(new SimilarityMethodDescriptorSupplier("getItemNeighbors"))
              .build();
        }
      }
    }
    return getGetItemNeighborsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static SimilarityStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<SimilarityStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<SimilarityStub>() {
        @Override
        public SimilarityStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new SimilarityStub(channel, callOptions);
        }
      };
    return SimilarityStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static SimilarityBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<SimilarityBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<SimilarityBlockingStub>() {
        @Override
        public SimilarityBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new SimilarityBlockingStub(channel, callOptions);
        }
      };
    return SimilarityBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static SimilarityFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<SimilarityFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<SimilarityFutureStub>() {
        @Override
        public SimilarityFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new SimilarityFutureStub(channel, callOptions);
        }
      };
    return SimilarityFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class SimilarityImplBase implements io.grpc.BindableService {

    /**
     */
    public void getItemNeighbors(SimilarityProto.IDs request,
                                 io.grpc.stub.StreamObserver<SimilarityProto.ItemNeighbors> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetItemNeighborsMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getGetItemNeighborsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                SimilarityProto.IDs,
                SimilarityProto.ItemNeighbors>(
                  this, METHODID_GET_ITEM_NEIGHBORS)))
          .build();
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class SimilarityStub extends io.grpc.stub.AbstractAsyncStub<SimilarityStub> {
    private SimilarityStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected SimilarityStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new SimilarityStub(channel, callOptions);
    }

    /**
     */
    public void getItemNeighbors(SimilarityProto.IDs request,
                                 io.grpc.stub.StreamObserver<SimilarityProto.ItemNeighbors> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetItemNeighborsMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class SimilarityBlockingStub extends io.grpc.stub.AbstractBlockingStub<SimilarityBlockingStub> {
    private SimilarityBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected SimilarityBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new SimilarityBlockingStub(channel, callOptions);
    }

    /**
     */
    public SimilarityProto.ItemNeighbors getItemNeighbors(SimilarityProto.IDs request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetItemNeighborsMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class SimilarityFutureStub extends io.grpc.stub.AbstractFutureStub<SimilarityFutureStub> {
    private SimilarityFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected SimilarityFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new SimilarityFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<SimilarityProto.ItemNeighbors> getItemNeighbors(
        SimilarityProto.IDs request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetItemNeighborsMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_GET_ITEM_NEIGHBORS = 0;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final SimilarityImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(SimilarityImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_GET_ITEM_NEIGHBORS:
          serviceImpl.getItemNeighbors((SimilarityProto.IDs) request,
              (io.grpc.stub.StreamObserver<SimilarityProto.ItemNeighbors>) responseObserver);
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

  private static abstract class SimilarityBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    SimilarityBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return SimilarityProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Similarity");
    }
  }

  private static final class SimilarityFileDescriptorSupplier
      extends SimilarityBaseDescriptorSupplier {
    SimilarityFileDescriptorSupplier() {}
  }

  private static final class SimilarityMethodDescriptorSupplier
      extends SimilarityBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    SimilarityMethodDescriptorSupplier(String methodName) {
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
      synchronized (SimilarityGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new SimilarityFileDescriptorSupplier())
              .addMethod(getGetItemNeighborsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
