package com.intel.analytics.bigdl.ppml.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: FLProto.proto")
public final class NNServiceGrpc {

  private NNServiceGrpc() {}

  public static final String SERVICE_NAME = "NNService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<FLProto.UploadRequest,
      FLProto.UploadResponse> getUploadTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTrain",
      requestType = FLProto.UploadRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadRequest,
      FLProto.UploadResponse> getUploadTrainMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadRequest, FLProto.UploadResponse> getUploadTrainMethod;
    if ((getUploadTrainMethod = NNServiceGrpc.getUploadTrainMethod) == null) {
      synchronized (NNServiceGrpc.class) {
        if ((getUploadTrainMethod = NNServiceGrpc.getUploadTrainMethod) == null) {
          NNServiceGrpc.getUploadTrainMethod = getUploadTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new NNServiceMethodDescriptorSupplier("UploadTrain"))
              .build();
        }
      }
    }
    return getUploadTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.DownloadRequest,
      FLProto.DownloadResponse> getDownloadTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "DownloadTrain",
      requestType = FLProto.DownloadRequest.class,
      responseType = FLProto.DownloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.DownloadRequest,
      FLProto.DownloadResponse> getDownloadTrainMethod() {
    io.grpc.MethodDescriptor<FLProto.DownloadRequest, FLProto.DownloadResponse> getDownloadTrainMethod;
    if ((getDownloadTrainMethod = NNServiceGrpc.getDownloadTrainMethod) == null) {
      synchronized (NNServiceGrpc.class) {
        if ((getDownloadTrainMethod = NNServiceGrpc.getDownloadTrainMethod) == null) {
          NNServiceGrpc.getDownloadTrainMethod = getDownloadTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.DownloadRequest, FLProto.DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DownloadTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new NNServiceMethodDescriptorSupplier("DownloadTrain"))
              .build();
        }
      }
    }
    return getDownloadTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.EvaluateRequest,
      FLProto.EvaluateResponse> getUploadEvaluateMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadEvaluate",
      requestType = FLProto.EvaluateRequest.class,
      responseType = FLProto.EvaluateResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.EvaluateRequest,
      FLProto.EvaluateResponse> getUploadEvaluateMethod() {
    io.grpc.MethodDescriptor<FLProto.EvaluateRequest, FLProto.EvaluateResponse> getUploadEvaluateMethod;
    if ((getUploadEvaluateMethod = NNServiceGrpc.getUploadEvaluateMethod) == null) {
      synchronized (NNServiceGrpc.class) {
        if ((getUploadEvaluateMethod = NNServiceGrpc.getUploadEvaluateMethod) == null) {
          NNServiceGrpc.getUploadEvaluateMethod = getUploadEvaluateMethod =
              io.grpc.MethodDescriptor.<FLProto.EvaluateRequest, FLProto.EvaluateResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadEvaluate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.EvaluateRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.EvaluateResponse.getDefaultInstance()))
              .setSchemaDescriptor(new NNServiceMethodDescriptorSupplier("UploadEvaluate"))
              .build();
        }
      }
    }
    return getUploadEvaluateMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static NNServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<NNServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<NNServiceStub>() {
        @Override
        public NNServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new NNServiceStub(channel, callOptions);
        }
      };
    return NNServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static NNServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<NNServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<NNServiceBlockingStub>() {
        @Override
        public NNServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new NNServiceBlockingStub(channel, callOptions);
        }
      };
    return NNServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static NNServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<NNServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<NNServiceFutureStub>() {
        @Override
        public NNServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new NNServiceFutureStub(channel, callOptions);
        }
      };
    return NNServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class NNServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void uploadTrain(FLProto.UploadRequest request,
                            io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTrainMethod(), responseObserver);
    }

    /**
     */
    public void downloadTrain(FLProto.DownloadRequest request,
                              io.grpc.stub.StreamObserver<FLProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadTrainMethod(), responseObserver);
    }

    /**
     */
    public void uploadEvaluate(FLProto.EvaluateRequest request,
                               io.grpc.stub.StreamObserver<FLProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadEvaluateMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_TRAIN)))
          .addMethod(
            getDownloadTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.DownloadRequest,
                FLProto.DownloadResponse>(
                  this, METHODID_DOWNLOAD_TRAIN)))
          .addMethod(
            getUploadEvaluateMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.EvaluateRequest,
                FLProto.EvaluateResponse>(
                  this, METHODID_UPLOAD_EVALUATE)))
          .build();
    }
  }

  /**
   */
  public static final class NNServiceStub extends io.grpc.stub.AbstractAsyncStub<NNServiceStub> {
    private NNServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected NNServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new NNServiceStub(channel, callOptions);
    }

    /**
     */
    public void uploadTrain(FLProto.UploadRequest request,
                            io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadTrain(FLProto.DownloadRequest request,
                              io.grpc.stub.StreamObserver<FLProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadEvaluate(FLProto.EvaluateRequest request,
                               io.grpc.stub.StreamObserver<FLProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadEvaluateMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class NNServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<NNServiceBlockingStub> {
    private NNServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected NNServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new NNServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public FLProto.UploadResponse uploadTrain(FLProto.UploadRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.DownloadResponse downloadTrain(FLProto.DownloadRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.EvaluateResponse uploadEvaluate(FLProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadEvaluateMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class NNServiceFutureStub extends io.grpc.stub.AbstractFutureStub<NNServiceFutureStub> {
    private NNServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected NNServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new NNServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadTrain(
        FLProto.UploadRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.DownloadResponse> downloadTrain(
        FLProto.DownloadRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.EvaluateResponse> uploadEvaluate(
        FLProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadEvaluateMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_TRAIN = 0;
  private static final int METHODID_DOWNLOAD_TRAIN = 1;
  private static final int METHODID_UPLOAD_EVALUATE = 2;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final NNServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(NNServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_UPLOAD_TRAIN:
          serviceImpl.uploadTrain((FLProto.UploadRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_TRAIN:
          serviceImpl.downloadTrain((FLProto.DownloadRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.DownloadResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_EVALUATE:
          serviceImpl.uploadEvaluate((FLProto.EvaluateRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.EvaluateResponse>) responseObserver);
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

  private static abstract class NNServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    NNServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return FLProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("NNService");
    }
  }

  private static final class NNServiceFileDescriptorSupplier
      extends NNServiceBaseDescriptorSupplier {
    NNServiceFileDescriptorSupplier() {}
  }

  private static final class NNServiceMethodDescriptorSupplier
      extends NNServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    NNServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (NNServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new NNServiceFileDescriptorSupplier())
              .addMethod(getUploadTrainMethod())
              .addMethod(getDownloadTrainMethod())
              .addMethod(getUploadEvaluateMethod())
              .build();
        }
      }
    }
    return result;
  }
}
