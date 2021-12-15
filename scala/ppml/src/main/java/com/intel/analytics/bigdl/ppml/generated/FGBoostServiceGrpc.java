package com.intel.analytics.bigdl.ppml.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: gbdt_service.proto")
public final class FGBoostServiceGrpc {

  private FGBoostServiceGrpc() {}

  public static final String SERVICE_NAME = "FGBoostService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.UploadSplitRequest,
      FGBoostServiceProto.UploadResponse> getUploadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadSplitTrain",
      requestType = FGBoostServiceProto.UploadSplitRequest.class,
      responseType = FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.UploadSplitRequest,
      FGBoostServiceProto.UploadResponse> getUploadSplitTrainMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.UploadSplitRequest, FGBoostServiceProto.UploadResponse> getUploadSplitTrainMethod;
    if ((getUploadSplitTrainMethod = FGBoostServiceGrpc.getUploadSplitTrainMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadSplitTrainMethod = FGBoostServiceGrpc.getUploadSplitTrainMethod) == null) {
          FGBoostServiceGrpc.getUploadSplitTrainMethod = getUploadSplitTrainMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.UploadSplitRequest, FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("UploadSplitTrain"))
              .build();
        }
      }
    }
    return getUploadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.DownloadSplitRequest,
      FGBoostServiceProto.DownloadSplitResponse> getDownloadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "DownloadSplitTrain",
      requestType = FGBoostServiceProto.DownloadSplitRequest.class,
      responseType = FGBoostServiceProto.DownloadSplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.DownloadSplitRequest,
      FGBoostServiceProto.DownloadSplitResponse> getDownloadSplitTrainMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.DownloadSplitRequest, FGBoostServiceProto.DownloadSplitResponse> getDownloadSplitTrainMethod;
    if ((getDownloadSplitTrainMethod = FGBoostServiceGrpc.getDownloadSplitTrainMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getDownloadSplitTrainMethod = FGBoostServiceGrpc.getDownloadSplitTrainMethod) == null) {
          FGBoostServiceGrpc.getDownloadSplitTrainMethod = getDownloadSplitTrainMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.DownloadSplitRequest, FGBoostServiceProto.DownloadSplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DownloadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.DownloadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.DownloadSplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("DownloadSplitTrain"))
              .build();
        }
      }
    }
    return getDownloadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.RegisterRequest,
      FGBoostServiceProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Register",
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
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("Register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeEvalRequest,
      FGBoostServiceProto.UploadResponse> getUploadTreeEvalMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeEval",
      requestType = FGBoostServiceProto.UploadTreeEvalRequest.class,
      responseType = FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeEvalRequest,
      FGBoostServiceProto.UploadResponse> getUploadTreeEvalMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeEvalRequest, FGBoostServiceProto.UploadResponse> getUploadTreeEvalMethod;
    if ((getUploadTreeEvalMethod = FGBoostServiceGrpc.getUploadTreeEvalMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadTreeEvalMethod = FGBoostServiceGrpc.getUploadTreeEvalMethod) == null) {
          FGBoostServiceGrpc.getUploadTreeEvalMethod = getUploadTreeEvalMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.UploadTreeEvalRequest, FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeEval"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadTreeEvalRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("UploadTreeEval"))
              .build();
        }
      }
    }
    return getUploadTreeEvalMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeLeavesRequest,
      FGBoostServiceProto.UploadResponse> getUploadTreeLeavesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeLeaves",
      requestType = FGBoostServiceProto.UploadTreeLeavesRequest.class,
      responseType = FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeLeavesRequest,
      FGBoostServiceProto.UploadResponse> getUploadTreeLeavesMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.UploadTreeLeavesRequest, FGBoostServiceProto.UploadResponse> getUploadTreeLeavesMethod;
    if ((getUploadTreeLeavesMethod = FGBoostServiceGrpc.getUploadTreeLeavesMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadTreeLeavesMethod = FGBoostServiceGrpc.getUploadTreeLeavesMethod) == null) {
          FGBoostServiceGrpc.getUploadTreeLeavesMethod = getUploadTreeLeavesMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.UploadTreeLeavesRequest, FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeLeaves"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadTreeLeavesRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("UploadTreeLeaves"))
              .build();
        }
      }
    }
    return getUploadTreeLeavesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FGBoostServiceProto.PredictTreeRequest,
      FGBoostServiceProto.PredictTreeResponse> getPredictTreeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "PredictTree",
      requestType = FGBoostServiceProto.PredictTreeRequest.class,
      responseType = FGBoostServiceProto.PredictTreeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FGBoostServiceProto.PredictTreeRequest,
      FGBoostServiceProto.PredictTreeResponse> getPredictTreeMethod() {
    io.grpc.MethodDescriptor<FGBoostServiceProto.PredictTreeRequest, FGBoostServiceProto.PredictTreeResponse> getPredictTreeMethod;
    if ((getPredictTreeMethod = FGBoostServiceGrpc.getPredictTreeMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getPredictTreeMethod = FGBoostServiceGrpc.getPredictTreeMethod) == null) {
          FGBoostServiceGrpc.getPredictTreeMethod = getPredictTreeMethod =
              io.grpc.MethodDescriptor.<FGBoostServiceProto.PredictTreeRequest, FGBoostServiceProto.PredictTreeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "PredictTree"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.PredictTreeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FGBoostServiceProto.PredictTreeResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("PredictTree"))
              .build();
        }
      }
    }
    return getPredictTreeMethod;
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
    public void uploadSplitTrain(FGBoostServiceProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(FGBoostServiceProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<FGBoostServiceProto.DownloadSplitResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void register(FGBoostServiceProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeEval(FGBoostServiceProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeEvalMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(FGBoostServiceProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeLeavesMethod(), responseObserver);
    }

    /**
     */
    public void predictTree(FGBoostServiceProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<FGBoostServiceProto.PredictTreeResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictTreeMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadSplitTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.UploadSplitRequest,
                FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_SPLIT_TRAIN)))
          .addMethod(
            getDownloadSplitTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.DownloadSplitRequest,
                FGBoostServiceProto.DownloadSplitResponse>(
                  this, METHODID_DOWNLOAD_SPLIT_TRAIN)))
          .addMethod(
            getRegisterMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.RegisterRequest,
                FGBoostServiceProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeEvalMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.UploadTreeEvalRequest,
                FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_EVAL)))
          .addMethod(
            getUploadTreeLeavesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.UploadTreeLeavesRequest,
                FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAVES)))
          .addMethod(
            getPredictTreeMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FGBoostServiceProto.PredictTreeRequest,
                FGBoostServiceProto.PredictTreeResponse>(
                  this, METHODID_PREDICT_TREE)))
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
    public void uploadSplitTrain(FGBoostServiceProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(FGBoostServiceProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<FGBoostServiceProto.DownloadSplitResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request, responseObserver);
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
    public void uploadTreeEval(FGBoostServiceProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(FGBoostServiceProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predictTree(FGBoostServiceProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<FGBoostServiceProto.PredictTreeResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request, responseObserver);
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
    public FGBoostServiceProto.UploadResponse uploadSplitTrain(FGBoostServiceProto.UploadSplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.DownloadSplitResponse downloadSplitTrain(FGBoostServiceProto.DownloadSplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.RegisterResponse register(FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.UploadResponse uploadTreeEval(FGBoostServiceProto.UploadTreeEvalRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeEvalMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.UploadResponse uploadTreeLeaves(FGBoostServiceProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeLeavesMethod(), getCallOptions(), request);
    }

    /**
     */
    public FGBoostServiceProto.PredictTreeResponse predictTree(FGBoostServiceProto.PredictTreeRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictTreeMethod(), getCallOptions(), request);
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
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.UploadResponse> uploadSplitTrain(
        FGBoostServiceProto.UploadSplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.DownloadSplitResponse> downloadSplitTrain(
        FGBoostServiceProto.DownloadSplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request);
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
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.UploadResponse> uploadTreeEval(
        FGBoostServiceProto.UploadTreeEvalRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.UploadResponse> uploadTreeLeaves(
        FGBoostServiceProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FGBoostServiceProto.PredictTreeResponse> predictTree(
        FGBoostServiceProto.PredictTreeRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_SPLIT_TRAIN = 0;
  private static final int METHODID_DOWNLOAD_SPLIT_TRAIN = 1;
  private static final int METHODID_REGISTER = 2;
  private static final int METHODID_UPLOAD_TREE_EVAL = 3;
  private static final int METHODID_UPLOAD_TREE_LEAVES = 4;
  private static final int METHODID_PREDICT_TREE = 5;

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
        case METHODID_UPLOAD_SPLIT_TRAIN:
          serviceImpl.uploadSplitTrain((FGBoostServiceProto.UploadSplitRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_SPLIT_TRAIN:
          serviceImpl.downloadSplitTrain((FGBoostServiceProto.DownloadSplitRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.DownloadSplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((FGBoostServiceProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_EVAL:
          serviceImpl.uploadTreeEval((FGBoostServiceProto.UploadTreeEvalRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAVES:
          serviceImpl.uploadTreeLeaves((FGBoostServiceProto.UploadTreeLeavesRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_PREDICT_TREE:
          serviceImpl.predictTree((FGBoostServiceProto.PredictTreeRequest) request,
              (io.grpc.stub.StreamObserver<FGBoostServiceProto.PredictTreeResponse>) responseObserver);
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
              .addMethod(getUploadSplitTrainMethod())
              .addMethod(getDownloadSplitTrainMethod())
              .addMethod(getRegisterMethod())
              .addMethod(getUploadTreeEvalMethod())
              .addMethod(getUploadTreeLeavesMethod())
              .addMethod(getPredictTreeMethod())
              .build();
        }
      }
    }
    return result;
  }
}
