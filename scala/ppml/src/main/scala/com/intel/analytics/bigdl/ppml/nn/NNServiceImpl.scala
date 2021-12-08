/*
 * Copyright 2021 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.bigdl.ppml.nn

import java.util
import java.util.Map

import com.intel.analytics.bigdl.dllib.nn.{BCECriterion, MSECriterion, Sigmoid, View}
import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy
import com.intel.analytics.bigdl.ppml.common.{Aggregator, AverageAggregator}
import com.intel.analytics.bigdl.ppml.common.FLPhase.TRAIN
import com.intel.analytics.bigdl.ppml.generated.FLProto._
import com.intel.analytics.bigdl.ppml.generated.NNServiceGrpc
import com.intel.analytics.bigdl.ppml.hfl.nn.HflNNAggregator
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNAggregator
import io.grpc.stub.StreamObserver


class NNServiceImpl() extends NNServiceGrpc.NNServiceImplBase {
  private var aggregatorMap: Map[String, Aggregator[Table]] = null
  initAggregatorMap()


  private def initAggregatorMap(): Unit = {
    aggregatorMap = new util.HashMap[String, Aggregator[Table]]
    aggregatorMap.put("vfl_logistic_regression", VflNNAggregator(1, Sigmoid[Float](),
      null, BCECriterion[Float](), Array(new Top1Accuracy())))
    aggregatorMap.put("hfl_linear_regression", VflNNAggregator(1, View[Float](),
      null, MSECriterion[Float](), Array(new Top1Accuracy())))
    aggregatorMap.put("hfl_logistic_regression", new HflNNAggregator())
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

