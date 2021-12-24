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

package com.intel.analytics.bigdl.ppml.fgboost

import com.intel.analytics.bigdl.ppml.base.DataHolder
import com.intel.analytics.bigdl.ppml.common.FLPhase
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.generated.{FGBoostServiceGrpc, FGBoostServiceProto}
import com.intel.analytics.bigdl.ppml.vfl.fgboost.VflGBoostAggregator
import io.grpc.stub.StreamObserver

class FGBoostServiceImpl(clientNum: Int) extends FGBoostServiceGrpc.FGBoostServiceImplBase{
  val aggregator = new VflGBoostAggregator()
  aggregator.setClientNum(clientNum)
  override def downloadTable(request: DownloadTableRequest,
                             responseObserver: StreamObserver[DownloadResponse]): Unit = {
    val version = request.getMetaData.getVersion
    val data = aggregator.getServerData(FLPhase.LABEL).getTableStorage().serverData
    if (data == null) {
      val response = "Your required data doesn't exist"
      responseObserver.onNext(DownloadResponse.newBuilder.setResponse(response).setCode(0).build)
      responseObserver.onCompleted()
    }
    else {
      val response = "Download data successfully"
      responseObserver.onNext(
        DownloadResponse.newBuilder.setResponse(response).setData(data).setCode(1).build)
      responseObserver.onCompleted()
    }
  }

  override def uploadTable(request: UploadTableRequest,
                           responseObserver: StreamObserver[UploadResponse]): Unit = {
    val clientUUID = request.getClientuuid
    val data = request.getData
    val version = data.getMetaData.getVersion
    try {
      aggregator.putClientData(FLPhase.LABEL, clientUUID, version, new DataHolder(data))
      val response = s"Table uploaded to server at clientID: $clientUUID, version: $version"
      responseObserver.onNext(UploadResponse.newBuilder.setResponse(response).setCode(0).build)
    } catch {
      case e: Exception =>
        val response = UploadResponse.newBuilder.setResponse(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }

  }

  override def split(request: SplitRequest, responseObserver: StreamObserver[SplitResponse]): Unit = {
    val clientUUID = request.getClientuuid
    val split = request.getSplit

    val version = -1 // version is not needed in fgboost
    try {
      aggregator.putClientData(FLPhase.SPLIT, clientUUID, version, new DataHolder(split))
      val bestSplit = aggregator.getBestSplit()
      if (split == null) {
        val response = "Your required data doesn't exist"
        responseObserver.onNext(SplitResponse.newBuilder.setResponse(response).setCode(0).build)
        responseObserver.onCompleted()
      }
      else {
        val response = "Download data successfully"
        responseObserver.onNext(SplitResponse.newBuilder.setResponse(response).setSplit(split).setCode(1).build)
        responseObserver.onCompleted()
      }
    } catch {
      case e: Exception =>
        val response = SplitResponse.newBuilder.setResponse(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }

  }

  override def uploadTreeLeaves(request: UploadTreeLeavesRequest,
                                responseObserver: StreamObserver[UploadResponse]): Unit = {
    val clientUUID = request.getClientuuid
    val leaves = request.getTreeLeaves

    val version = -1 // version is not needed in fgboost
    try {
      aggregator.putClientData(FLPhase.LEAF, clientUUID, version, new DataHolder(leaves))
      val bestSplit = aggregator.getBestSplit()
      val response = "Your required data doesn't exist"
      responseObserver.onNext(UploadResponse.newBuilder.setResponse(response).setCode(0).build)
      responseObserver.onCompleted()
    } catch {
      case e: Exception =>
        val response = UploadResponse.newBuilder.setResponse(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }
  }

  override def evaluate(request: EvaluateRequest,
                        responseObserver: StreamObserver[EvaluateResponse]): Unit = {

  }

  override def predict(request: PredictRequest,
                       responseObserver: StreamObserver[PredictResponse]): Unit = {

  }


}
