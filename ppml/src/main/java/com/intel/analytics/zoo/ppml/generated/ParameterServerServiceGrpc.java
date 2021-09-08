package com.intel.analytics.zoo.ppml.generated;

import io.grpc.stub.ClientCalls;

import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;

/**
 * <pre>
 * Parameter Server Proto
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.33.0)",
    comments = "Source: FLProto.proto")
public final class ParameterServerServiceGrpc {

  private ParameterServerServiceGrpc() {}

  public static final String SERVICE_NAME = "ParameterServerService";

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
    if ((getUploadTrainMethod = ParameterServerServiceGrpc.getUploadTrainMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getUploadTrainMethod = ParameterServerServiceGrpc.getUploadTrainMethod) == null) {
          ParameterServerServiceGrpc.getUploadTrainMethod = getUploadTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("UploadTrain"))
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
    if ((getDownloadTrainMethod = ParameterServerServiceGrpc.getDownloadTrainMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getDownloadTrainMethod = ParameterServerServiceGrpc.getDownloadTrainMethod) == null) {
          ParameterServerServiceGrpc.getDownloadTrainMethod = getDownloadTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.DownloadRequest, FLProto.DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DownloadTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("DownloadTrain"))
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
    if ((getUploadEvaluateMethod = ParameterServerServiceGrpc.getUploadEvaluateMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getUploadEvaluateMethod = ParameterServerServiceGrpc.getUploadEvaluateMethod) == null) {
          ParameterServerServiceGrpc.getUploadEvaluateMethod = getUploadEvaluateMethod =
              io.grpc.MethodDescriptor.<FLProto.EvaluateRequest, FLProto.EvaluateResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadEvaluate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.EvaluateRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.EvaluateResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("UploadEvaluate"))
              .build();
        }
      }
    }
    return getUploadEvaluateMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.UploadSplitRequest,
      FLProto.UploadResponse> getUploadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadSplitTrain",
      requestType = FLProto.UploadSplitRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadSplitRequest,
      FLProto.UploadResponse> getUploadSplitTrainMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadSplitRequest, FLProto.UploadResponse> getUploadSplitTrainMethod;
    if ((getUploadSplitTrainMethod = ParameterServerServiceGrpc.getUploadSplitTrainMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getUploadSplitTrainMethod = ParameterServerServiceGrpc.getUploadSplitTrainMethod) == null) {
          ParameterServerServiceGrpc.getUploadSplitTrainMethod = getUploadSplitTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadSplitRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("UploadSplitTrain"))
              .build();
        }
      }
    }
    return getUploadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.DownloadSplitRequest,
      FLProto.DownloadSplitResponse> getDownloadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "DownloadSplitTrain",
      requestType = FLProto.DownloadSplitRequest.class,
      responseType = FLProto.DownloadSplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.DownloadSplitRequest,
      FLProto.DownloadSplitResponse> getDownloadSplitTrainMethod() {
    io.grpc.MethodDescriptor<FLProto.DownloadSplitRequest, FLProto.DownloadSplitResponse> getDownloadSplitTrainMethod;
    if ((getDownloadSplitTrainMethod = ParameterServerServiceGrpc.getDownloadSplitTrainMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getDownloadSplitTrainMethod = ParameterServerServiceGrpc.getDownloadSplitTrainMethod) == null) {
          ParameterServerServiceGrpc.getDownloadSplitTrainMethod = getDownloadSplitTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.DownloadSplitRequest, FLProto.DownloadSplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DownloadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadSplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("DownloadSplitTrain"))
              .build();
        }
      }
    }
    return getDownloadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.RegisterRequest,
      FLProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Register",
      requestType = FLProto.RegisterRequest.class,
      responseType = FLProto.RegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.RegisterRequest,
      FLProto.RegisterResponse> getRegisterMethod() {
    io.grpc.MethodDescriptor<FLProto.RegisterRequest, FLProto.RegisterResponse> getRegisterMethod;
    if ((getRegisterMethod = ParameterServerServiceGrpc.getRegisterMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getRegisterMethod = ParameterServerServiceGrpc.getRegisterMethod) == null) {
          ParameterServerServiceGrpc.getRegisterMethod = getRegisterMethod =
              io.grpc.MethodDescriptor.<FLProto.RegisterRequest, FLProto.RegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("Register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.UploadTreeEvalRequest,
      FLProto.UploadResponse> getUploadTreeEvalMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeEval",
      requestType = FLProto.UploadTreeEvalRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadTreeEvalRequest,
      FLProto.UploadResponse> getUploadTreeEvalMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadTreeEvalRequest, FLProto.UploadResponse> getUploadTreeEvalMethod;
    if ((getUploadTreeEvalMethod = ParameterServerServiceGrpc.getUploadTreeEvalMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getUploadTreeEvalMethod = ParameterServerServiceGrpc.getUploadTreeEvalMethod) == null) {
          ParameterServerServiceGrpc.getUploadTreeEvalMethod = getUploadTreeEvalMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadTreeEvalRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeEval"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadTreeEvalRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("UploadTreeEval"))
              .build();
        }
      }
    }
    return getUploadTreeEvalMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.UploadTreeLeavesRequest,
      FLProto.UploadResponse> getUploadTreeLeavesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeLeaves",
      requestType = FLProto.UploadTreeLeavesRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadTreeLeavesRequest,
      FLProto.UploadResponse> getUploadTreeLeavesMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadTreeLeavesRequest, FLProto.UploadResponse> getUploadTreeLeavesMethod;
    if ((getUploadTreeLeavesMethod = ParameterServerServiceGrpc.getUploadTreeLeavesMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getUploadTreeLeavesMethod = ParameterServerServiceGrpc.getUploadTreeLeavesMethod) == null) {
          ParameterServerServiceGrpc.getUploadTreeLeavesMethod = getUploadTreeLeavesMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadTreeLeavesRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeLeaves"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadTreeLeavesRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("UploadTreeLeaves"))
              .build();
        }
      }
    }
    return getUploadTreeLeavesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.PredictTreeRequest,
      FLProto.PredictTreeResponse> getPredictTreeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "PredictTree",
      requestType = FLProto.PredictTreeRequest.class,
      responseType = FLProto.PredictTreeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.PredictTreeRequest,
      FLProto.PredictTreeResponse> getPredictTreeMethod() {
    io.grpc.MethodDescriptor<FLProto.PredictTreeRequest, FLProto.PredictTreeResponse> getPredictTreeMethod;
    if ((getPredictTreeMethod = ParameterServerServiceGrpc.getPredictTreeMethod) == null) {
      synchronized (ParameterServerServiceGrpc.class) {
        if ((getPredictTreeMethod = ParameterServerServiceGrpc.getPredictTreeMethod) == null) {
          ParameterServerServiceGrpc.getPredictTreeMethod = getPredictTreeMethod =
              io.grpc.MethodDescriptor.<FLProto.PredictTreeRequest, FLProto.PredictTreeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "PredictTree"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.PredictTreeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.PredictTreeResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ParameterServerServiceMethodDescriptorSupplier("PredictTree"))
              .build();
        }
      }
    }
    return getPredictTreeMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static ParameterServerServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ParameterServerServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ParameterServerServiceStub>() {
        @Override
        public ParameterServerServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ParameterServerServiceStub(channel, callOptions);
        }
      };
    return ParameterServerServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static ParameterServerServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ParameterServerServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ParameterServerServiceBlockingStub>() {
        @Override
        public ParameterServerServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ParameterServerServiceBlockingStub(channel, callOptions);
        }
      };
    return ParameterServerServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static ParameterServerServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ParameterServerServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ParameterServerServiceFutureStub>() {
        @Override
        public ParameterServerServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ParameterServerServiceFutureStub(channel, callOptions);
        }
      };
    return ParameterServerServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Parameter Server Proto
   * </pre>
   */
  public static abstract class ParameterServerServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     * NN
     * </pre>
     */
    public void uploadTrain(FLProto.UploadRequest request,
                            io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadTrainMethod(), responseObserver);
    }

    /**
     */
    public void downloadTrain(FLProto.DownloadRequest request,
                              io.grpc.stub.StreamObserver<FLProto.DownloadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getDownloadTrainMethod(), responseObserver);
    }

    /**
     */
    public void uploadEvaluate(FLProto.EvaluateRequest request,
                               io.grpc.stub.StreamObserver<FLProto.EvaluateResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadEvaluateMethod(), responseObserver);
    }

    /**
     * <pre>
     * Gradient Boosting Tree
     * </pre>
     */
    public void uploadSplitTrain(FLProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(FLProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<FLProto.DownloadSplitResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getDownloadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void register(FLProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FLProto.RegisterResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeEval(FLProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadTreeEvalMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(FLProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadTreeLeavesMethod(), responseObserver);
    }

    /**
     */
    public void predictTree(FLProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<FLProto.PredictTreeResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getPredictTreeMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadTrainMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_TRAIN)))
          .addMethod(
            getDownloadTrainMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.DownloadRequest,
                FLProto.DownloadResponse>(
                  this, METHODID_DOWNLOAD_TRAIN)))
          .addMethod(
            getUploadEvaluateMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.EvaluateRequest,
                FLProto.EvaluateResponse>(
                  this, METHODID_UPLOAD_EVALUATE)))
          .addMethod(
            getUploadSplitTrainMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadSplitRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_SPLIT_TRAIN)))
          .addMethod(
            getDownloadSplitTrainMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.DownloadSplitRequest,
                FLProto.DownloadSplitResponse>(
                  this, METHODID_DOWNLOAD_SPLIT_TRAIN)))
          .addMethod(
            getRegisterMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.RegisterRequest,
                FLProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeEvalMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadTreeEvalRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_EVAL)))
          .addMethod(
            getUploadTreeLeavesMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadTreeLeavesRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAVES)))
          .addMethod(
            getPredictTreeMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                FLProto.PredictTreeRequest,
                FLProto.PredictTreeResponse>(
                  this, METHODID_PREDICT_TREE)))
          .build();
    }
  }

  /**
   * <pre>
   * Parameter Server Proto
   * </pre>
   */
  public static final class ParameterServerServiceStub extends io.grpc.stub.AbstractAsyncStub<ParameterServerServiceStub> {
    private ParameterServerServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected ParameterServerServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ParameterServerServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * NN
     * </pre>
     */
    public void uploadTrain(FLProto.UploadRequest request,
                            io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadTrain(FLProto.DownloadRequest request,
                              io.grpc.stub.StreamObserver<FLProto.DownloadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadEvaluate(FLProto.EvaluateRequest request,
                               io.grpc.stub.StreamObserver<FLProto.EvaluateResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadEvaluateMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Gradient Boosting Tree
     * </pre>
     */
    public void uploadSplitTrain(FLProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(FLProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<FLProto.DownloadSplitResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void register(FLProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FLProto.RegisterResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeEval(FLProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(FLProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predictTree(FLProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<FLProto.PredictTreeResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * Parameter Server Proto
   * </pre>
   */
  public static final class ParameterServerServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<ParameterServerServiceBlockingStub> {
    private ParameterServerServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected ParameterServerServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ParameterServerServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * NN
     * </pre>
     */
    public FLProto.UploadResponse uploadTrain(FLProto.UploadRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.DownloadResponse downloadTrain(FLProto.DownloadRequest request) {
      return blockingUnaryCall(
          getChannel(), getDownloadTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.EvaluateResponse uploadEvaluate(FLProto.EvaluateRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadEvaluateMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Gradient Boosting Tree
     * </pre>
     */
    public FLProto.UploadResponse uploadSplitTrain(FLProto.UploadSplitRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.DownloadSplitResponse downloadSplitTrain(FLProto.DownloadSplitRequest request) {
      return blockingUnaryCall(
          getChannel(), getDownloadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.RegisterResponse register(FLProto.RegisterRequest request) {
      return blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.UploadResponse uploadTreeEval(FLProto.UploadTreeEvalRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadTreeEvalMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.UploadResponse uploadTreeLeaves(FLProto.UploadTreeLeavesRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadTreeLeavesMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.PredictTreeResponse predictTree(FLProto.PredictTreeRequest request) {
      return blockingUnaryCall(
          getChannel(), getPredictTreeMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Parameter Server Proto
   * </pre>
   */
  public static final class ParameterServerServiceFutureStub extends io.grpc.stub.AbstractFutureStub<ParameterServerServiceFutureStub> {
    private ParameterServerServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected ParameterServerServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ParameterServerServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * NN
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadTrain(
        FLProto.UploadRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.DownloadResponse> downloadTrain(
        FLProto.DownloadRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getDownloadTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.EvaluateResponse> uploadEvaluate(
        FLProto.EvaluateRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadEvaluateMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Gradient Boosting Tree
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadSplitTrain(
        FLProto.UploadSplitRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.DownloadSplitResponse> downloadSplitTrain(
        FLProto.DownloadSplitRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.RegisterResponse> register(
        FLProto.RegisterRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadTreeEval(
        FLProto.UploadTreeEvalRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadTreeLeaves(
        FLProto.UploadTreeLeavesRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.PredictTreeResponse> predictTree(
        FLProto.PredictTreeRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_TRAIN = 0;
  private static final int METHODID_DOWNLOAD_TRAIN = 1;
  private static final int METHODID_UPLOAD_EVALUATE = 2;
  private static final int METHODID_UPLOAD_SPLIT_TRAIN = 3;
  private static final int METHODID_DOWNLOAD_SPLIT_TRAIN = 4;
  private static final int METHODID_REGISTER = 5;
  private static final int METHODID_UPLOAD_TREE_EVAL = 6;
  private static final int METHODID_UPLOAD_TREE_LEAVES = 7;
  private static final int METHODID_PREDICT_TREE = 8;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final ParameterServerServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(ParameterServerServiceImplBase serviceImpl, int methodId) {
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
        case METHODID_UPLOAD_SPLIT_TRAIN:
          serviceImpl.uploadSplitTrain((FLProto.UploadSplitRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_SPLIT_TRAIN:
          serviceImpl.downloadSplitTrain((FLProto.DownloadSplitRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.DownloadSplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((FLProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_EVAL:
          serviceImpl.uploadTreeEval((FLProto.UploadTreeEvalRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAVES:
          serviceImpl.uploadTreeLeaves((FLProto.UploadTreeLeavesRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_PREDICT_TREE:
          serviceImpl.predictTree((FLProto.PredictTreeRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.PredictTreeResponse>) responseObserver);
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

  private static abstract class ParameterServerServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    ParameterServerServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return FLProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("ParameterServerService");
    }
  }

  private static final class ParameterServerServiceFileDescriptorSupplier
      extends ParameterServerServiceBaseDescriptorSupplier {
    ParameterServerServiceFileDescriptorSupplier() {}
  }

  private static final class ParameterServerServiceMethodDescriptorSupplier
      extends ParameterServerServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    ParameterServerServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (ParameterServerServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new ParameterServerServiceFileDescriptorSupplier())
              .addMethod(getUploadTrainMethod())
              .addMethod(getDownloadTrainMethod())
              .addMethod(getUploadEvaluateMethod())
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
