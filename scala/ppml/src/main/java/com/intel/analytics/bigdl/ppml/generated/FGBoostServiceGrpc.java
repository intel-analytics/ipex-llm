package com.intel.analytics.bigdl.ppml.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: fgboost_service.proto")
public final class FGBoostServiceGrpc {

  private FGBoostServiceGrpc() {}

  public static final String SERVICE_NAME = "fgboost.FGBoostService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.UploadLabelRequest,
      FGBoostServiceProto.UploadResponse> getUploadLabelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadLabel",
      requestType = FGBoostServiceProto.UploadLabelRequest.class,
      responseType = FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.UploadLabelRequest,
      FGBoostServiceProto.UploadResponse> getUploadLabelMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.UploadLabelRequest, FGBoostServiceProto.UploadResponse> getUploadLabelMethod;
    if ((getUploadLabelMethod = FGBoostServiceGrpc.getUploadLabelMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadLabelMethod = FGBoostServiceGrpc.getUploadLabelMethod) == null) {
          FGBoostServiceGrpc.getUploadLabelMethod = getUploadLabelMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.UploadLabelRequest, FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadLabel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadLabelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("uploadLabel"))
              .build();
        }
      }
    }
    return getUploadLabelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.DownloadLabelRequest,
      FGBoostServiceProto.DownloadResponse> getDownloadLabelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "downloadLabel",
      requestType = FGBoostServiceProto.DownloadLabelRequest.class,
      responseType = FGBoostServiceProto.DownloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.DownloadLabelRequest,
      FGBoostServiceProto.DownloadResponse> getDownloadLabelMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.DownloadLabelRequest, FGBoostServiceProto.DownloadResponse> getDownloadLabelMethod;
    if ((getDownloadLabelMethod = FGBoostServiceGrpc.getDownloadLabelMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getDownloadLabelMethod = FGBoostServiceGrpc.getDownloadLabelMethod) == null) {
          FGBoostServiceGrpc.getDownloadLabelMethod = getDownloadLabelMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.DownloadLabelRequest, FGBoostServiceProto.DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "downloadLabel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.DownloadLabelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.DownloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("downloadLabel"))
              .build();
        }
      }
    }
    return getDownloadLabelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.SplitRequest,
      FGBoostServiceProto.SplitResponse> getSplitMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "split",
      requestType = FGBoostServiceProto.SplitRequest.class,
      responseType = FGBoostServiceProto.SplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.SplitRequest,
      FGBoostServiceProto.SplitResponse> getSplitMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.SplitRequest, FGBoostServiceProto.SplitResponse> getSplitMethod;
    if ((getSplitMethod = FGBoostServiceGrpc.getSplitMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getSplitMethod = FGBoostServiceGrpc.getSplitMethod) == null) {
          FGBoostServiceGrpc.getSplitMethod = getSplitMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.SplitRequest, FGBoostServiceProto.SplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "split"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.SplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.SplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("split"))
              .build();
        }
      }
    }
    return getSplitMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.RegisterRequest,
      FGBoostServiceProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "register",
      requestType = FGBoostServiceProto.RegisterRequest.class,
      responseType = FGBoostServiceProto.RegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.RegisterRequest,
      FGBoostServiceProto.RegisterResponse> getRegisterMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.RegisterRequest, FGBoostServiceProto.RegisterResponse> getRegisterMethod;
    if ((getRegisterMethod = FGBoostServiceGrpc.getRegisterMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getRegisterMethod = FGBoostServiceGrpc.getRegisterMethod) == null) {
          FGBoostServiceGrpc.getRegisterMethod = getRegisterMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.RegisterRequest, FGBoostServiceProto.RegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeLeafRequest,
      FGBoostServiceProto.UploadResponse> getUploadTreeLeafMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadTreeLeaf",
      requestType = FGBoostServiceProto.UploadTreeLeafRequest.class,
      responseType = FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeLeafRequest,
      FGBoostServiceProto.UploadResponse> getUploadTreeLeafMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeLeafRequest, FGBoostServiceProto.UploadResponse> getUploadTreeLeafMethod;
    if ((getUploadTreeLeafMethod = FGBoostServiceGrpc.getUploadTreeLeafMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadTreeLeafMethod = FGBoostServiceGrpc.getUploadTreeLeafMethod) == null) {
          FGBoostServiceGrpc.getUploadTreeLeafMethod = getUploadTreeLeafMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.UploadTreeLeafRequest, FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadTreeLeaf"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadTreeLeafRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("uploadTreeLeaf"))
              .build();
        }
      }
    }
    return getUploadTreeLeafMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.EvaluateRequest,
      FGBoostServiceProto.EvaluateResponse> getEvaluateMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "evaluate",
      requestType = FGBoostServiceProto.EvaluateRequest.class,
      responseType = FGBoostServiceProto.EvaluateResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.EvaluateRequest,
      FGBoostServiceProto.EvaluateResponse> getEvaluateMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.EvaluateRequest, FGBoostServiceProto.EvaluateResponse> getEvaluateMethod;
    if ((getEvaluateMethod = FGBoostServiceGrpc.getEvaluateMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getEvaluateMethod = FGBoostServiceGrpc.getEvaluateMethod) == null) {
          FGBoostServiceGrpc.getEvaluateMethod = getEvaluateMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.EvaluateRequest, FGBoostServiceProto.EvaluateResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "evaluate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.EvaluateRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.EvaluateResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("evaluate"))
              .build();
        }
      }
    }
    return getEvaluateMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.PredictRequest,
      FGBoostServiceProto.PredictResponse> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "predict",
      requestType = FGBoostServiceProto.PredictRequest.class,
      responseType = FGBoostServiceProto.PredictResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.PredictRequest,
      FGBoostServiceProto.PredictResponse> getPredictMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.PredictRequest, FGBoostServiceProto.PredictResponse> getPredictMethod;
    if ((getPredictMethod = FGBoostServiceGrpc.getPredictMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getPredictMethod = FGBoostServiceGrpc.getPredictMethod) == null) {
          FGBoostServiceGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.PredictRequest, FGBoostServiceProto.PredictResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.PredictRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.PredictResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("predict"))
              .build();
        }
      }
    }
    return getPredictMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static FGBoostServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FGBoostServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FGBoostServiceStub>() {
        @Override
        public FGBoostServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FGBoostServiceStub(channel, callOptions);
        }
      };
    return FGBoostServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static FGBoostServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FGBoostServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FGBoostServiceBlockingStub>() {
        @Override
        public FGBoostServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FGBoostServiceBlockingStub(channel, callOptions);
        }
      };
    return FGBoostServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static FGBoostServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FGBoostServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FGBoostServiceFutureStub>() {
        @Override
        public FGBoostServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FGBoostServiceFutureStub(channel, callOptions);
        }
      };
    return FGBoostServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class FGBoostServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void uploadLabel(FGBoostServiceProto.UploadLabelRequest request,
                            io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadLabelMethod(), responseObserver);
    }

    /**
     */
    public void downloadLabel(FGBoostServiceProto.DownloadLabelRequest request,
                              io.grpc.stub.StreamObserver<FGBoostServiceProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadLabelMethod(), responseObserver);
    }

    /**
     */
    public void split(FGBoostServiceProto.SplitRequest request,
                      io.grpc.stub.StreamObserver<FGBoostServiceProto.SplitResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSplitMethod(), responseObserver);
    }

    /**
     */
    public void register(FGBoostServiceProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaf(FGBoostServiceProto.UploadTreeLeafRequest request,
                               io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeLeafMethod(), responseObserver);
    }

    /**
     */
    public void evaluate(FGBoostServiceProto.EvaluateRequest request,
                         io.grpc.stub.StreamObserver<FGBoostServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getEvaluateMethod(), responseObserver);
    }

    /**
     */
    public void predict(FGBoostServiceProto.PredictRequest request,
                        io.grpc.stub.StreamObserver<FGBoostServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadLabelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.UploadLabelRequest,
                FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_LABEL)))
          .addMethod(
            getDownloadLabelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.DownloadLabelRequest,
                FGBoostServiceProto.DownloadResponse>(
                  this, METHODID_DOWNLOAD_LABEL)))
          .addMethod(
            getSplitMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.SplitRequest,
                FGBoostServiceProto.SplitResponse>(
                  this, METHODID_SPLIT)))
          .addMethod(
            getRegisterMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.RegisterRequest,
                FGBoostServiceProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeLeafMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.UploadTreeLeafRequest,
                FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAF)))
          .addMethod(
            getEvaluateMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.EvaluateRequest,
                FGBoostServiceProto.EvaluateResponse>(
                  this, METHODID_EVALUATE)))
          .addMethod(
            getPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.PredictRequest,
                FGBoostServiceProto.PredictResponse>(
                  this, METHODID_PREDICT)))
          .build();
    }
  }

  /**
   */
  public static final class FGBoostServiceStub extends io.grpc.stub.AbstractAsyncStub<FGBoostServiceStub> {
    private FGBoostServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FGBoostServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FGBoostServiceStub(channel, callOptions);
    }

    /**
     */
    public void uploadLabel(FGBoostServiceProto.UploadLabelRequest request,
                            io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadLabelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadLabel(FGBoostServiceProto.DownloadLabelRequest request,
                              io.grpc.stub.StreamObserver<FGBoostServiceProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadLabelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void split(FGBoostServiceProto.SplitRequest request,
                      io.grpc.stub.StreamObserver<FGBoostServiceProto.SplitResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSplitMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void register(FGBoostServiceProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaf(FGBoostServiceProto.UploadTreeLeafRequest request,
                               io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeafMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void evaluate(FGBoostServiceProto.EvaluateRequest request,
                         io.grpc.stub.StreamObserver<FGBoostServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predict(FGBoostServiceProto.PredictRequest request,
                        io.grpc.stub.StreamObserver<FGBoostServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class FGBoostServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<FGBoostServiceBlockingStub> {
    private FGBoostServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FGBoostServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FGBoostServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public FGBoostServiceProto.UploadResponse uploadLabel(FGBoostServiceProto.UploadLabelRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadLabelMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.DownloadResponse downloadLabel(FGBoostServiceProto.DownloadLabelRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadLabelMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.SplitResponse split(FGBoostServiceProto.SplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSplitMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.RegisterResponse register(FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.UploadResponse uploadTreeLeaf(FGBoostServiceProto.UploadTreeLeafRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeLeafMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.EvaluateResponse evaluate(FGBoostServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getEvaluateMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.PredictResponse predict(FGBoostServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class FGBoostServiceFutureStub extends io.grpc.stub.AbstractFutureStub<FGBoostServiceFutureStub> {
    private FGBoostServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FGBoostServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FGBoostServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.UploadResponse> uploadLabel(
        FGBoostServiceProto.UploadLabelRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadLabelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.DownloadResponse> downloadLabel(
        FGBoostServiceProto.DownloadLabelRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadLabelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.SplitResponse> split(
        FGBoostServiceProto.SplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSplitMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.RegisterResponse> register(
        FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.UploadResponse> uploadTreeLeaf(
        FGBoostServiceProto.UploadTreeLeafRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeLeafMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.EvaluateResponse> evaluate(
        FGBoostServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.PredictResponse> predict(
        FGBoostServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_LABEL = 0;
  private static final int METHODID_DOWNLOAD_LABEL = 1;
  private static final int METHODID_SPLIT = 2;
  private static final int METHODID_REGISTER = 3;
  private static final int METHODID_UPLOAD_TREE_LEAF = 4;
  private static final int METHODID_EVALUATE = 5;
  private static final int METHODID_PREDICT = 6;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final FGBoostServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(FGBoostServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_UPLOAD_LABEL:
          serviceImpl.uploadLabel((FGBoostServiceProto.UploadLabelRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_LABEL:
          serviceImpl.downloadLabel((FGBoostServiceProto.DownloadLabelRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.DownloadResponse>) responseObserver);
          break;
        case METHODID_SPLIT:
          serviceImpl.split((FGBoostServiceProto.SplitRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.SplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((FGBoostServiceProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAF:
          serviceImpl.uploadTreeLeaf((FGBoostServiceProto.UploadTreeLeafRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_EVALUATE:
          serviceImpl.evaluate((FGBoostServiceProto.EvaluateRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.EvaluateResponse>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((FGBoostServiceProto.PredictRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.PredictResponse>) responseObserver);
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

  private static abstract class FGBoostServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    FGBoostServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return FGBoostServiceProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("FGBoostService");
    }
  }

  private static final class FGBoostServiceFileDescriptorSupplier
      extends FGBoostServiceBaseDescriptorSupplier {
    FGBoostServiceFileDescriptorSupplier() {}
  }

  private static final class FGBoostServiceMethodDescriptorSupplier
      extends FGBoostServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    FGBoostServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (FGBoostServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new FGBoostServiceFileDescriptorSupplier())
              .addMethod(getUploadLabelMethod())
              .addMethod(getDownloadLabelMethod())
              .addMethod(getSplitMethod())
              .addMethod(getRegisterMethod())
              .addMethod(getUploadTreeLeafMethod())
              .addMethod(getEvaluateMethod())
              .addMethod(getPredictMethod())
              .build();
        }
      }
    }
    return result;
  }
}
