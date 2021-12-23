package com.intel.analytics.bigdl.ppml.service

import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.generated.{FGBoostServiceGrpc, FGBoostServiceProto}
import io.grpc.stub.StreamObserver

class FGBoostServiceImpl extends FGBoostServiceGrpc.FGBoostServiceImplBase{
  override def downloadTable(request: DownloadTableRequest,
                             responseObserver: StreamObserver[DownloadResponse]): Unit = {

  }

  override def uploadTable(request: UploadTableRequest, responseObserver: StreamObserver[UploadResponse]): Unit = {


  }

  override def split(request: SplitRequest, responseObserver: StreamObserver[SplitResponse]): Unit = {


  }
}
