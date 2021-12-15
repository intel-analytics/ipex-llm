package com.intel.analytics.bigdl.ppml.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: gbdt_service.proto")
public final class GBDTServiceGrpc {

  private GBDTServiceGrpc() {}

  public static final String SERVICE_NAME = "GBDTService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<GBDTServiceProto.UploadSplitRequest,
      GBDTServiceProto.UploadResponse> getUploadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadSplitTrain",
      requestType = GBDTServiceProto.UploadSplitRequest.class,
      responseType = GBDTServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<GBDTServiceProto.UploadSplitRequest,
      GBDTServiceProto.UploadResponse> getUploadSplitTrainMethod() {
    io.grpc.MethodDescriptor<GBDTServiceProto.UploadSplitRequest, GBDTServiceProto.UploadResponse> getUploadSplitTrainMethod;
    if ((getUploadSplitTrainMethod = GBDTServiceGrpc.getUploadSplitTrainMethod) == null) {
      synchronized (GBDTServiceGrpc.class) {
        if ((getUploadSplitTrainMethod = GBDTServiceGrpc.getUploadSplitTrainMethod) == null) {
          GBDTServiceGrpc.getUploadSplitTrainMethod = getUploadSplitTrainMethod =
              io.grpc.MethodDescriptor.<GBDTServiceProto.UploadSplitRequest, GBDTServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.UploadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBDTServiceMethodDescriptorSupplier("UploadSplitTrain"))
              .build();
        }
      }
    }
    return getUploadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<GBDTServiceProto.DownloadSplitRequest,
      GBDTServiceProto.DownloadSplitResponse> getDownloadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "DownloadSplitTrain",
      requestType = GBDTServiceProto.DownloadSplitRequest.class,
      responseType = GBDTServiceProto.DownloadSplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<GBDTServiceProto.DownloadSplitRequest,
      GBDTServiceProto.DownloadSplitResponse> getDownloadSplitTrainMethod() {
    io.grpc.MethodDescriptor<GBDTServiceProto.DownloadSplitRequest, GBDTServiceProto.DownloadSplitResponse> getDownloadSplitTrainMethod;
    if ((getDownloadSplitTrainMethod = GBDTServiceGrpc.getDownloadSplitTrainMethod) == null) {
      synchronized (GBDTServiceGrpc.class) {
        if ((getDownloadSplitTrainMethod = GBDTServiceGrpc.getDownloadSplitTrainMethod) == null) {
          GBDTServiceGrpc.getDownloadSplitTrainMethod = getDownloadSplitTrainMethod =
              io.grpc.MethodDescriptor.<GBDTServiceProto.DownloadSplitRequest, GBDTServiceProto.DownloadSplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DownloadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.DownloadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.DownloadSplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBDTServiceMethodDescriptorSupplier("DownloadSplitTrain"))
              .build();
        }
      }
    }
    return getDownloadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<GBDTServiceProto.RegisterRequest,
      GBDTServiceProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Register",
      requestType = GBDTServiceProto.RegisterRequest.class,
      responseType = GBDTServiceProto.RegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<GBDTServiceProto.RegisterRequest,
      GBDTServiceProto.RegisterResponse> getRegisterMethod() {
    io.grpc.MethodDescriptor<GBDTServiceProto.RegisterRequest, GBDTServiceProto.RegisterResponse> getRegisterMethod;
    if ((getRegisterMethod = GBDTServiceGrpc.getRegisterMethod) == null) {
      synchronized (GBDTServiceGrpc.class) {
        if ((getRegisterMethod = GBDTServiceGrpc.getRegisterMethod) == null) {
          GBDTServiceGrpc.getRegisterMethod = getRegisterMethod =
              io.grpc.MethodDescriptor.<GBDTServiceProto.RegisterRequest, GBDTServiceProto.RegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBDTServiceMethodDescriptorSupplier("Register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<GBDTServiceProto.UploadTreeEvalRequest,
      GBDTServiceProto.UploadResponse> getUploadTreeEvalMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeEval",
      requestType = GBDTServiceProto.UploadTreeEvalRequest.class,
      responseType = GBDTServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<GBDTServiceProto.UploadTreeEvalRequest,
      GBDTServiceProto.UploadResponse> getUploadTreeEvalMethod() {
    io.grpc.MethodDescriptor<GBDTServiceProto.UploadTreeEvalRequest, GBDTServiceProto.UploadResponse> getUploadTreeEvalMethod;
    if ((getUploadTreeEvalMethod = GBDTServiceGrpc.getUploadTreeEvalMethod) == null) {
      synchronized (GBDTServiceGrpc.class) {
        if ((getUploadTreeEvalMethod = GBDTServiceGrpc.getUploadTreeEvalMethod) == null) {
          GBDTServiceGrpc.getUploadTreeEvalMethod = getUploadTreeEvalMethod =
              io.grpc.MethodDescriptor.<GBDTServiceProto.UploadTreeEvalRequest, GBDTServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeEval"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.UploadTreeEvalRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBDTServiceMethodDescriptorSupplier("UploadTreeEval"))
              .build();
        }
      }
    }
    return getUploadTreeEvalMethod;
  }

  private static volatile io.grpc.MethodDescriptor<GBDTServiceProto.UploadTreeLeavesRequest,
      GBDTServiceProto.UploadResponse> getUploadTreeLeavesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeLeaves",
      requestType = GBDTServiceProto.UploadTreeLeavesRequest.class,
      responseType = GBDTServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<GBDTServiceProto.UploadTreeLeavesRequest,
      GBDTServiceProto.UploadResponse> getUploadTreeLeavesMethod() {
    io.grpc.MethodDescriptor<GBDTServiceProto.UploadTreeLeavesRequest, GBDTServiceProto.UploadResponse> getUploadTreeLeavesMethod;
    if ((getUploadTreeLeavesMethod = GBDTServiceGrpc.getUploadTreeLeavesMethod) == null) {
      synchronized (GBDTServiceGrpc.class) {
        if ((getUploadTreeLeavesMethod = GBDTServiceGrpc.getUploadTreeLeavesMethod) == null) {
          GBDTServiceGrpc.getUploadTreeLeavesMethod = getUploadTreeLeavesMethod =
              io.grpc.MethodDescriptor.<GBDTServiceProto.UploadTreeLeavesRequest, GBDTServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeLeaves"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.UploadTreeLeavesRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBDTServiceMethodDescriptorSupplier("UploadTreeLeaves"))
              .build();
        }
      }
    }
    return getUploadTreeLeavesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<GBDTServiceProto.PredictTreeRequest,
      GBDTServiceProto.PredictTreeResponse> getPredictTreeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "PredictTree",
      requestType = GBDTServiceProto.PredictTreeRequest.class,
      responseType = GBDTServiceProto.PredictTreeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<GBDTServiceProto.PredictTreeRequest,
      GBDTServiceProto.PredictTreeResponse> getPredictTreeMethod() {
    io.grpc.MethodDescriptor<GBDTServiceProto.PredictTreeRequest, GBDTServiceProto.PredictTreeResponse> getPredictTreeMethod;
    if ((getPredictTreeMethod = GBDTServiceGrpc.getPredictTreeMethod) == null) {
      synchronized (GBDTServiceGrpc.class) {
        if ((getPredictTreeMethod = GBDTServiceGrpc.getPredictTreeMethod) == null) {
          GBDTServiceGrpc.getPredictTreeMethod = getPredictTreeMethod =
              io.grpc.MethodDescriptor.<GBDTServiceProto.PredictTreeRequest, GBDTServiceProto.PredictTreeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "PredictTree"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.PredictTreeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  GBDTServiceProto.PredictTreeResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBDTServiceMethodDescriptorSupplier("PredictTree"))
              .build();
        }
      }
    }
    return getPredictTreeMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static GBDTServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GBDTServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GBDTServiceStub>() {
        @Override
        public GBDTServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GBDTServiceStub(channel, callOptions);
        }
      };
    return GBDTServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static GBDTServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GBDTServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GBDTServiceBlockingStub>() {
        @Override
        public GBDTServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GBDTServiceBlockingStub(channel, callOptions);
        }
      };
    return GBDTServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static GBDTServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GBDTServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GBDTServiceFutureStub>() {
        @Override
        public GBDTServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GBDTServiceFutureStub(channel, callOptions);
        }
      };
    return GBDTServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class GBDTServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void uploadSplitTrain(GBDTServiceProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(GBDTServiceProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<GBDTServiceProto.DownloadSplitResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void register(GBDTServiceProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<GBDTServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeEval(GBDTServiceProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeEvalMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(GBDTServiceProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeLeavesMethod(), responseObserver);
    }

    /**
     */
    public void predictTree(GBDTServiceProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<GBDTServiceProto.PredictTreeResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictTreeMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadSplitTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                GBDTServiceProto.UploadSplitRequest,
                GBDTServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_SPLIT_TRAIN)))
          .addMethod(
            getDownloadSplitTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                GBDTServiceProto.DownloadSplitRequest,
                GBDTServiceProto.DownloadSplitResponse>(
                  this, METHODID_DOWNLOAD_SPLIT_TRAIN)))
          .addMethod(
            getRegisterMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                GBDTServiceProto.RegisterRequest,
                GBDTServiceProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeEvalMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                GBDTServiceProto.UploadTreeEvalRequest,
                GBDTServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_EVAL)))
          .addMethod(
            getUploadTreeLeavesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                GBDTServiceProto.UploadTreeLeavesRequest,
                GBDTServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAVES)))
          .addMethod(
            getPredictTreeMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                GBDTServiceProto.PredictTreeRequest,
                GBDTServiceProto.PredictTreeResponse>(
                  this, METHODID_PREDICT_TREE)))
          .build();
    }
  }

  /**
   */
  public static final class GBDTServiceStub extends io.grpc.stub.AbstractAsyncStub<GBDTServiceStub> {
    private GBDTServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected GBDTServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GBDTServiceStub(channel, callOptions);
    }

    /**
     */
    public void uploadSplitTrain(GBDTServiceProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(GBDTServiceProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<GBDTServiceProto.DownloadSplitResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void register(GBDTServiceProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<GBDTServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeEval(GBDTServiceProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(GBDTServiceProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predictTree(GBDTServiceProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<GBDTServiceProto.PredictTreeResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class GBDTServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<GBDTServiceBlockingStub> {
    private GBDTServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected GBDTServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GBDTServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public GBDTServiceProto.UploadResponse uploadSplitTrain(GBDTServiceProto.UploadSplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public GBDTServiceProto.DownloadSplitResponse downloadSplitTrain(GBDTServiceProto.DownloadSplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public GBDTServiceProto.RegisterResponse register(GBDTServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public GBDTServiceProto.UploadResponse uploadTreeEval(GBDTServiceProto.UploadTreeEvalRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeEvalMethod(), getCallOptions(), request);
    }

    /**
     */
    public GBDTServiceProto.UploadResponse uploadTreeLeaves(GBDTServiceProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeLeavesMethod(), getCallOptions(), request);
    }

    /**
     */
    public GBDTServiceProto.PredictTreeResponse predictTree(GBDTServiceProto.PredictTreeRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictTreeMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class GBDTServiceFutureStub extends io.grpc.stub.AbstractFutureStub<GBDTServiceFutureStub> {
    private GBDTServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected GBDTServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GBDTServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<GBDTServiceProto.UploadResponse> uploadSplitTrain(
        GBDTServiceProto.UploadSplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<GBDTServiceProto.DownloadSplitResponse> downloadSplitTrain(
        GBDTServiceProto.DownloadSplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<GBDTServiceProto.RegisterResponse> register(
        GBDTServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<GBDTServiceProto.UploadResponse> uploadTreeEval(
        GBDTServiceProto.UploadTreeEvalRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<GBDTServiceProto.UploadResponse> uploadTreeLeaves(
        GBDTServiceProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<GBDTServiceProto.PredictTreeResponse> predictTree(
        GBDTServiceProto.PredictTreeRequest request) {
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
    private final GBDTServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(GBDTServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_UPLOAD_SPLIT_TRAIN:
          serviceImpl.uploadSplitTrain((GBDTServiceProto.UploadSplitRequest) request,
              (io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_SPLIT_TRAIN:
          serviceImpl.downloadSplitTrain((GBDTServiceProto.DownloadSplitRequest) request,
              (io.grpc.stub.StreamObserver<GBDTServiceProto.DownloadSplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((GBDTServiceProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<GBDTServiceProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_EVAL:
          serviceImpl.uploadTreeEval((GBDTServiceProto.UploadTreeEvalRequest) request,
              (io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAVES:
          serviceImpl.uploadTreeLeaves((GBDTServiceProto.UploadTreeLeavesRequest) request,
              (io.grpc.stub.StreamObserver<GBDTServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_PREDICT_TREE:
          serviceImpl.predictTree((GBDTServiceProto.PredictTreeRequest) request,
              (io.grpc.stub.StreamObserver<GBDTServiceProto.PredictTreeResponse>) responseObserver);
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

  private static abstract class GBDTServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    GBDTServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return GBDTServiceProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("GBDTService");
    }
  }

  private static final class GBDTServiceFileDescriptorSupplier
      extends GBDTServiceBaseDescriptorSupplier {
    GBDTServiceFileDescriptorSupplier() {}
  }

  private static final class GBDTServiceMethodDescriptorSupplier
      extends GBDTServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    GBDTServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (GBDTServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new GBDTServiceFileDescriptorSupplier())
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
