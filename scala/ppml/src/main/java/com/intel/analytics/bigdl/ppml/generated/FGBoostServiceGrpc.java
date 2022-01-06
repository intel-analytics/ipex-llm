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
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> getUploadTableMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadTable",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> getUploadTableMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> getUploadTableMethod;
    if ((getUploadTableMethod = FGBoostServiceGrpc.getUploadTableMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadTableMethod = FGBoostServiceGrpc.getUploadTableMethod) == null) {
          FGBoostServiceGrpc.getUploadTableMethod = getUploadTableMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadTable"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("uploadTable"))
              .build();
        }
      }
    }
    return getUploadTableMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse> getDownloadTableMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "downloadTable",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse> getDownloadTableMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse> getDownloadTableMethod;
    if ((getDownloadTableMethod = FGBoostServiceGrpc.getDownloadTableMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getDownloadTableMethod = FGBoostServiceGrpc.getDownloadTableMethod) == null) {
          FGBoostServiceGrpc.getDownloadTableMethod = getDownloadTableMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "downloadTable"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("downloadTable"))
              .build();
        }
      }
    }
    return getDownloadTableMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse> getSplitMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "split",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse> getSplitMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse> getSplitMethod;
    if ((getSplitMethod = FGBoostServiceGrpc.getSplitMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getSplitMethod = FGBoostServiceGrpc.getSplitMethod) == null) {
          FGBoostServiceGrpc.getSplitMethod = getSplitMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "split"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("split"))
              .build();
        }
      }
    }
    return getSplitMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "register",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse> getRegisterMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse> getRegisterMethod;
    if ((getRegisterMethod = FGBoostServiceGrpc.getRegisterMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getRegisterMethod = FGBoostServiceGrpc.getRegisterMethod) == null) {
          FGBoostServiceGrpc.getRegisterMethod = getRegisterMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> getUploadTreeLeavesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadTreeLeaves",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> getUploadTreeLeavesMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> getUploadTreeLeavesMethod;
    if ((getUploadTreeLeavesMethod = FGBoostServiceGrpc.getUploadTreeLeavesMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadTreeLeavesMethod = FGBoostServiceGrpc.getUploadTreeLeavesMethod) == null) {
          FGBoostServiceGrpc.getUploadTreeLeavesMethod = getUploadTreeLeavesMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadTreeLeaves"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("uploadTreeLeaves"))
              .build();
        }
      }
    }
    return getUploadTreeLeavesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse> getEvaluateMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "evaluate",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse> getEvaluateMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse> getEvaluateMethod;
    if ((getEvaluateMethod = FGBoostServiceGrpc.getEvaluateMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getEvaluateMethod = FGBoostServiceGrpc.getEvaluateMethod) == null) {
          FGBoostServiceGrpc.getEvaluateMethod = getEvaluateMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "evaluate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("evaluate"))
              .build();
        }
      }
    }
    return getEvaluateMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "predict",
      requestType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest,
      com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse> getPredictMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse> getPredictMethod;
    if ((getPredictMethod = FGBoostServiceGrpc.getPredictMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getPredictMethod = FGBoostServiceGrpc.getPredictMethod) == null) {
          FGBoostServiceGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest, com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse.getDefaultInstance()))
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
        @java.lang.Override
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
        @java.lang.Override
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
        @java.lang.Override
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
    public void uploadTable(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTableMethod(), responseObserver);
    }

    /**
     */
    public void downloadTable(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadTableMethod(), responseObserver);
    }

    /**
     */
    public void split(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSplitMethod(), responseObserver);
    }

    /**
     */
    public void register(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeLeavesMethod(), responseObserver);
    }

    /**
     */
    public void evaluate(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getEvaluateMethod(), responseObserver);
    }

    /**
     */
    public void predict(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadTableMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TABLE)))
          .addMethod(
            getDownloadTableMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse>(
                  this, METHODID_DOWNLOAD_TABLE)))
          .addMethod(
            getSplitMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse>(
                  this, METHODID_SPLIT)))
          .addMethod(
            getRegisterMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeLeavesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAVES)))
          .addMethod(
            getEvaluateMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse>(
                  this, METHODID_EVALUATE)))
          .addMethod(
            getPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest,
                com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse>(
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

    @java.lang.Override
    protected FGBoostServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FGBoostServiceStub(channel, callOptions);
    }

    /**
     */
    public void uploadTable(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTableMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadTable(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadTableMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void split(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSplitMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void register(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void evaluate(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predict(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse> responseObserver) {
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

    @java.lang.Override
    protected FGBoostServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FGBoostServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse uploadTable(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTableMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse downloadTable(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadTableMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse split(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSplitMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse register(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse uploadTreeLeaves(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeLeavesMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse evaluate(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getEvaluateMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse predict(com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest request) {
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

    @java.lang.Override
    protected FGBoostServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FGBoostServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> uploadTable(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTableMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse> downloadTable(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadTableMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse> split(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSplitMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse> register(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse> uploadTreeLeaves(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse> evaluate(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse> predict(
        com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_TABLE = 0;
  private static final int METHODID_DOWNLOAD_TABLE = 1;
  private static final int METHODID_SPLIT = 2;
  private static final int METHODID_REGISTER = 3;
  private static final int METHODID_UPLOAD_TREE_LEAVES = 4;
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

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_UPLOAD_TABLE:
          serviceImpl.uploadTable((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTableRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_TABLE:
          serviceImpl.downloadTable((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadTableRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.DownloadResponse>) responseObserver);
          break;
        case METHODID_SPLIT:
          serviceImpl.split((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.SplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAVES:
          serviceImpl.uploadTreeLeaves((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadTreeLeavesRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_EVALUATE:
          serviceImpl.evaluate((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.EvaluateResponse>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.PredictResponse>) responseObserver);
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

  private static abstract class FGBoostServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    FGBoostServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.getDescriptor();
    }

    @java.lang.Override
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

    @java.lang.Override
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
              .addMethod(getUploadTableMethod())
              .addMethod(getDownloadTableMethod())
              .addMethod(getSplitMethod())
              .addMethod(getRegisterMethod())
              .addMethod(getUploadTreeLeavesMethod())
              .addMethod(getEvaluateMethod())
              .addMethod(getPredictMethod())
              .build();
        }
      }
    }
    return result;
  }
}
