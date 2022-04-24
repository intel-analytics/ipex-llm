package com.intel.analytics.bigdl.ppml.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: nn_service.proto")
public final class NNServiceGrpc {

  private NNServiceGrpc() {}

  public static final String SERVICE_NAME = "nn.NNService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse> getTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "train",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse> getTrainMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest, com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse> getTrainMethod;
    if ((getTrainMethod = NNServiceGrpc.getTrainMethod) == null) {
      synchronized (NNServiceGrpc.class) {
        if ((getTrainMethod = NNServiceGrpc.getTrainMethod) == null) {
          NNServiceGrpc.getTrainMethod = getTrainMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest, com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "train"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse.getDefaultInstance()))
              .setSchemaDescriptor(new NNServiceMethodDescriptorSupplier("train"))
              .build();
        }
      }
    }
    return getTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse> getEvaluateMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "evaluate",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse> getEvaluateMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest, com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse> getEvaluateMethod;
    if ((getEvaluateMethod = NNServiceGrpc.getEvaluateMethod) == null) {
      synchronized (NNServiceGrpc.class) {
        if ((getEvaluateMethod = NNServiceGrpc.getEvaluateMethod) == null) {
          NNServiceGrpc.getEvaluateMethod = getEvaluateMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest, com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "evaluate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse.getDefaultInstance()))
              .setSchemaDescriptor(new NNServiceMethodDescriptorSupplier("evaluate"))
              .build();
        }
      }
    }
    return getEvaluateMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "predict",
      requestType = com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest.class,
      responseType = com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest,
      com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse> getPredictMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest, com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse> getPredictMethod;
    if ((getPredictMethod = NNServiceGrpc.getPredictMethod) == null) {
      synchronized (NNServiceGrpc.class) {
        if ((getPredictMethod = NNServiceGrpc.getPredictMethod) == null) {
          NNServiceGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest, com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse.getDefaultInstance()))
              .setSchemaDescriptor(new NNServiceMethodDescriptorSupplier("predict"))
              .build();
        }
      }
    }
    return getPredictMethod;
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
    public void train(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getTrainMethod(), responseObserver);
    }

    /**
     */
    public void evaluate(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getEvaluateMethod(), responseObserver);
    }

    /**
     */
    public void predict(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse>(
                  this, METHODID_TRAIN)))
          .addMethod(
            getEvaluateMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse>(
                  this, METHODID_EVALUATE)))
          .addMethod(
            getPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest,
                com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse>(
                  this, METHODID_PREDICT)))
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
    public void train(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void evaluate(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predict(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request, responseObserver);
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
    public com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse train(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse evaluate(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getEvaluateMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse predict(com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictMethod(), getCallOptions(), request);
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
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse> train(
        com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse> evaluate(
        com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getEvaluateMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse> predict(
        com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_TRAIN = 0;
  private static final int METHODID_EVALUATE = 1;
  private static final int METHODID_PREDICT = 2;

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
        case METHODID_TRAIN:
          serviceImpl.train((com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.TrainResponse>) responseObserver);
          break;
        case METHODID_EVALUATE:
          serviceImpl.evaluate((com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.EvaluateResponse>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.PredictResponse>) responseObserver);
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
      return com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.getDescriptor();
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
              .addMethod(getTrainMethod())
              .addMethod(getEvaluateMethod())
              .addMethod(getPredictMethod())
              .build();
        }
      }
    }
    return result;
  }
}
