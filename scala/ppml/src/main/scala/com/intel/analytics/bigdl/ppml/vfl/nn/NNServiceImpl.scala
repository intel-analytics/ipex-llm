
package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy
import com.intel.analytics.bigdl.ppml.common.Aggregator
import com.intel.analytics.bigdl.ppml.generated.FLProto._
import com.intel.analytics.bigdl.ppml.generated.NNServiceGrpc
import io.grpc.stub.StreamObserver
import java.util
import java.util.{HashMap, Map}

import com.intel.analytics.bigdl.dllib.nn.{BCECriterion, MSECriterion, Sigmoid, View}
import com.intel.analytics.bigdl.ppml.common.FLPhase._
import com.intel.analytics.bigdl.ppml.common.FLPhase.TRAIN
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNAggregator


class NNServiceImpl() extends NNServiceGrpc.NNServiceImplBase {
  private var aggregatorMap: Map[String, Aggregator] = null
  initAggregatorMap()


  private def initAggregatorMap(): Unit = {
    aggregatorMap = new util.HashMap[String, Aggregator]
    aggregatorMap.put("logistic_regression", VflNNAggregator(1, Sigmoid[Float](),
      null, BCECriterion[Float](), Array(new Top1Accuracy())))
    aggregatorMap.put("linear_regression", VflNNAggregator(1, View[Float](),
      null, MSECriterion[Float](), Array(new Top1Accuracy())))
  }


  override def downloadTrain(request: DownloadRequest, responseObserver: StreamObserver[DownloadResponse]): Unit = {
    val version = request.getMetaData.getVersion
    val aggregator = aggregatorMap.get(request.getAlgorithm)
    val data = aggregator.getServerData(TRAIN).serverData
    if (data == null) {
      val response = "Your required data doesn't exist"
      responseObserver.onNext(DownloadResponse.newBuilder.setResponse(response).setCode(0).build)
    }
    else {
      val response = "Download data successfully"
      responseObserver.onNext(DownloadResponse.newBuilder.setResponse(response).setData(data).setCode(1).build)
    }
    responseObserver.onCompleted()
  }

  override def uploadTrain(request: UploadRequest,
                           responseObserver: StreamObserver[UploadResponse]): Unit = {
    // check data version, drop all the unmatched version
    val clientUUID = request.getClientuuid
    val data = request.getData
    val version = data.getMetaData.getVersion
    val aggregator = aggregatorMap.get(request.getAlgorithm)
    try {
      aggregator.putClientData(TRAIN, clientUUID, version, data)
      val response = UploadResponse.newBuilder.setResponse("Data received").setCode(0).build
      responseObserver.onNext(response)
      responseObserver.onCompleted()
    } catch {
      case e: Exception =>
        val response = UploadResponse.newBuilder.setResponse(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }
  }
}

