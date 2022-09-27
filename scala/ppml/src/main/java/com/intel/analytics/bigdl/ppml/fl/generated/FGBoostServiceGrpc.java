package com.intel.analytics.bigdl.ppml.fl.generated;

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
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> getUploadLabelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadLabel",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> getUploadLabelMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> getUploadLabelMethod;
    if ((getUploadLabelMethod = FGBoostServiceGrpc.getUploadLabelMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadLabelMethod = FGBoostServiceGrpc.getUploadLabelMethod) == null) {
          FGBoostServiceGrpc.getUploadLabelMethod = getUploadLabelMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadLabel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("uploadLabel"))
              .build();
        }
      }
    }
    return getUploadLabelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse> getDownloadLabelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "downloadLabel",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse> getDownloadLabelMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse> getDownloadLabelMethod;
    if ((getDownloadLabelMethod = FGBoostServiceGrpc.getDownloadLabelMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getDownloadLabelMethod = FGBoostServiceGrpc.getDownloadLabelMethod) == null) {
          FGBoostServiceGrpc.getDownloadLabelMethod = getDownloadLabelMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "downloadLabel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("downloadLabel"))
              .build();
        }
      }
    }
    return getDownloadLabelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse> getSplitMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "split",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse> getSplitMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse> getSplitMethod;
    if ((getSplitMethod = FGBoostServiceGrpc.getSplitMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getSplitMethod = FGBoostServiceGrpc.getSplitMethod) == null) {
          FGBoostServiceGrpc.getSplitMethod = getSplitMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "split"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("split"))
              .build();
        }
      }
    }
    return getSplitMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "register",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse> getRegisterMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse> getRegisterMethod;
    if ((getRegisterMethod = FGBoostServiceGrpc.getRegisterMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getRegisterMethod = FGBoostServiceGrpc.getRegisterMethod) == null) {
          FGBoostServiceGrpc.getRegisterMethod = getRegisterMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> getUploadTreeLeafMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadTreeLeaf",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> getUploadTreeLeafMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> getUploadTreeLeafMethod;
    if ((getUploadTreeLeafMethod = FGBoostServiceGrpc.getUploadTreeLeafMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getUploadTreeLeafMethod = FGBoostServiceGrpc.getUploadTreeLeafMethod) == null) {
          FGBoostServiceGrpc.getUploadTreeLeafMethod = getUploadTreeLeafMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadTreeLeaf"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("uploadTreeLeaf"))
              .build();
        }
      }
    }
    return getUploadTreeLeafMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse> getEvaluateMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "evaluate",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse> getEvaluateMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse> getEvaluateMethod;
    if ((getEvaluateMethod = FGBoostServiceGrpc.getEvaluateMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getEvaluateMethod = FGBoostServiceGrpc.getEvaluateMethod) == null) {
          FGBoostServiceGrpc.getEvaluateMethod = getEvaluateMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "evaluate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("evaluate"))
              .build();
        }
      }
    }
    return getEvaluateMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "predict",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse> getPredictMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse> getPredictMethod;
    if ((getPredictMethod = FGBoostServiceGrpc.getPredictMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getPredictMethod = FGBoostServiceGrpc.getPredictMethod) == null) {
          FGBoostServiceGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("predict"))
              .build();
        }
      }
    }
    return getPredictMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse> getSaveServerModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "saveServerModel",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse> getSaveServerModelMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse> getSaveServerModelMethod;
    if ((getSaveServerModelMethod = FGBoostServiceGrpc.getSaveServerModelMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getSaveServerModelMethod = FGBoostServiceGrpc.getSaveServerModelMethod) == null) {
          FGBoostServiceGrpc.getSaveServerModelMethod = getSaveServerModelMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "saveServerModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("saveServerModel"))
              .build();
        }
      }
    }
    return getSaveServerModelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse> getLoadServerModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "loadServerModel",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse> getLoadServerModelMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse> getLoadServerModelMethod;
    if ((getLoadServerModelMethod = FGBoostServiceGrpc.getLoadServerModelMethod) == null) {
      synchronized (FGBoostServiceGrpc.class) {
        if ((getLoadServerModelMethod = FGBoostServiceGrpc.getLoadServerModelMethod) == null) {
          FGBoostServiceGrpc.getLoadServerModelMethod = getLoadServerModelMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest, com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "loadServerModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse.getDefaultInstance()))
              .setSchemaDescriptor(new FGBoostServiceMethodDescriptorSupplier("loadServerModel"))
              .build();
        }
      }
    }
    return getLoadServerModelMethod;
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
    public void uploadLabel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadLabelMethod(), responseObserver);
    }

    /**
     */
    public void downloadLabel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadLabelMethod(), responseObserver);
    }

    /**
     */
    public void split(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSplitMethod(), responseObserver);
    }

    /**
     */
    public void register(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaf(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeLeafMethod(), responseObserver);
    }

    /**
     */
    public void evaluate(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getEvaluateMethod(), responseObserver);
    }

    /**
     */
    public void predict(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    /**
     */
    public void saveServerModel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSaveServerModelMethod(), responseObserver);
    }

    /**
     */
    public void loadServerModel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getLoadServerModelMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadLabelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_LABEL)))
          .addMethod(
            getDownloadLabelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse>(
                  this, METHODID_DOWNLOAD_LABEL)))
          .addMethod(
            getSplitMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse>(
                  this, METHODID_SPLIT)))
          .addMethod(
            getRegisterMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeLeafMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAF)))
          .addMethod(
            getEvaluateMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse>(
                  this, METHODID_EVALUATE)))
          .addMethod(
            getPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse>(
                  this, METHODID_PREDICT)))
          .addMethod(
            getSaveServerModelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse>(
                  this, METHODID_SAVE_SERVER_MODEL)))
          .addMethod(
            getLoadServerModelMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse>(
                  this, METHODID_LOAD_SERVER_MODEL)))
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
    public void uploadLabel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadLabelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadLabel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadLabelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void split(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSplitMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void register(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaf(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeafMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void evaluate(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predict(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void saveServerModel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSaveServerModelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void loadServerModel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getLoadServerModelMethod(), getCallOptions()), request, responseObserver);
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
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse uploadLabel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadLabelMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse downloadLabel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadLabelMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse split(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSplitMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse register(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse uploadTreeLeaf(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeLeafMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse evaluate(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getEvaluateMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse predict(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse saveServerModel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSaveServerModelMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse loadServerModel(com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getLoadServerModelMethod(), getCallOptions(), request);
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
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> uploadLabel(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadLabelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse> downloadLabel(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadLabelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse> split(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSplitMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse> register(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse> uploadTreeLeaf(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeLeafMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse> evaluate(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse> predict(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse> saveServerModel(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSaveServerModelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse> loadServerModel(
        com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getLoadServerModelMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_LABEL = 0;
  private static final int METHODID_DOWNLOAD_LABEL = 1;
  private static final int METHODID_SPLIT = 2;
  private static final int METHODID_REGISTER = 3;
  private static final int METHODID_UPLOAD_TREE_LEAF = 4;
  private static final int METHODID_EVALUATE = 5;
  private static final int METHODID_PREDICT = 6;
  private static final int METHODID_SAVE_SERVER_MODEL = 7;
  private static final int METHODID_LOAD_SERVER_MODEL = 8;

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
        case METHODID_UPLOAD_LABEL:
          serviceImpl.uploadLabel((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadLabelRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_LABEL:
          serviceImpl.downloadLabel((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadLabelRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.DownloadResponse>) responseObserver);
          break;
        case METHODID_SPLIT:
          serviceImpl.split((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAF:
          serviceImpl.uploadTreeLeaf((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadTreeLeafRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.UploadResponse>) responseObserver);
          break;
        case METHODID_EVALUATE:
          serviceImpl.evaluate((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.EvaluateResponse>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.PredictResponse>) responseObserver);
          break;
        case METHODID_SAVE_SERVER_MODEL:
          serviceImpl.saveServerModel((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.SaveModelResponse>) responseObserver);
          break;
        case METHODID_LOAD_SERVER_MODEL:
          serviceImpl.loadServerModel((com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.LoadModelResponse>) responseObserver);
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
      return com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.getDescriptor();
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
              .addMethod(getUploadLabelMethod())
              .addMethod(getDownloadLabelMethod())
              .addMethod(getSplitMethod())
              .addMethod(getRegisterMethod())
              .addMethod(getUploadTreeLeafMethod())
              .addMethod(getEvaluateMethod())
              .addMethod(getPredictMethod())
              .addMethod(getSaveServerModelMethod())
              .addMethod(getLoadServerModelMethod())
              .build();
        }
      }
    }
    return result;
  }
}
