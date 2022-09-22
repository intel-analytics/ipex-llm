/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.fl.fgboost

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.FLConfig
import com.intel.analytics.bigdl.ppml.fl.base.DataHolder
import com.intel.analytics.bigdl.ppml.fl.common.FLPhase
import com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceGrpc
import com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.fl.utils.ServerUtils
import io.grpc.stub.StreamObserver
import org.apache.commons.lang3.exception.ExceptionUtils
import org.apache.logging.log4j.LogManager

import java.util
import java.util.concurrent.ConcurrentHashMap
import collection.JavaConverters._


class FGBoostServiceImpl(clientNum: Int, config: FLConfig)
  extends FGBoostServiceGrpc.FGBoostServiceImplBase{
  val logger = LogManager.getLogger(getClass)
  val aggregator = new FGBoostAggregator(config)
  aggregator.setClientNum(clientNum)

  // store client id as key and client data as value
  val evalBufferMap = new ConcurrentHashMap[Int, util.ArrayList[BoostEval]]()
  var predBufferMap = new ConcurrentHashMap[Int, util.ArrayList[BoostEval]]()

  override def downloadLabel(request: DownloadLabelRequest,
                             responseObserver: StreamObserver[DownloadResponse]): Unit = {
    val version = request.getMetaData.getVersion
    logger.debug(s"Server received downloadLabel request of version: $version")
    synchronized {
      if (aggregator.getLabelStorage().version < version) {
        logger.debug(s"Download label: server version is " +
          s"${aggregator.getLabelStorage().version}, waiting")
        wait()
      } else if (aggregator.getLabelStorage().version > version) {
        logger.error(s"Server version could never advance client version, something is wrong.")
      } else {
        notifyAll()
      }
    }
    val data = aggregator.getLabelStorage().serverData
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

  override def uploadLabel(request: UploadLabelRequest,
                           responseObserver: StreamObserver[UploadResponse]): Unit = {
    val clientUUID = request.getClientuuid
    Log4Error.invalidInputError(ServerUtils.checkClientId(clientNum, clientUUID),
      s"Invalid client ID, should be in range of [1, $clientNum], got $clientUUID")

    val data = request.getData
    val version = data.getMetaData.getVersion
    try {
      aggregator.putClientData(FLPhase.LABEL, clientUUID, version, new DataHolder(data))
      val response = s"TensorMap uploaded to server at clientID: $clientUUID, version: $version"
      responseObserver.onNext(UploadResponse.newBuilder.setResponse(response).setCode(0).build)
      responseObserver.onCompleted()
    } catch {
      case e: Exception =>
        val errorMsg = ExceptionUtils.getStackTrace(e)
        val response = UploadResponse.newBuilder.setResponse(errorMsg).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }

  }

  override def split(request: SplitRequest,
                     responseObserver: StreamObserver[SplitResponse]): Unit = {
    val clientUUID = request.getClientuuid
    Log4Error.invalidInputError(ServerUtils.checkClientId(clientNum, clientUUID),
      s"Invalid client ID, should be in range of [1, $clientNum], got $clientUUID")
    val split = request.getSplit
    try {
      aggregator.putClientData(FLPhase.SPLIT, clientUUID, split.getVersion, new DataHolder(split))
      val bestSplit = aggregator.getBestSplit(split.getTreeID, split.getNodeID)
      if (bestSplit == null) {
        val response = "Your required bestSplit data doesn't exist"
        responseObserver.onNext(SplitResponse.newBuilder.setResponse(response).setCode(1).build)
        responseObserver.onCompleted()
      }
      else {
        val response = SplitResponse.newBuilder
          .setResponse("Split success").setSplit(bestSplit).setCode(0).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
      }
    } catch {
      case e: Exception =>
        val errorMsg = ExceptionUtils.getStackTrace(e)
        val response = SplitResponse.newBuilder.setResponse(errorMsg).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }

  }

  override def uploadTreeLeaf(request: UploadTreeLeafRequest,
                              responseObserver: StreamObserver[UploadResponse]): Unit = {
    val clientUUID = request.getClientuuid
    Log4Error.invalidInputError(ServerUtils.checkClientId(clientNum, clientUUID),
      s"Invalid client ID, should be in range of [1, $clientNum], got $clientUUID")
    val treeLeaf = request.getTreeLeaf

    try {
      val response = "Upload tree leaf successfully"
      aggregator.putClientData(FLPhase.TREE_LEAF,
        clientUUID, treeLeaf.getVersion, new DataHolder(treeLeaf))
      responseObserver.onNext(UploadResponse.newBuilder.setResponse(response).setCode(0).build)
      responseObserver.onCompleted()
    } catch {
      case e: Exception =>
        val response = UploadResponse.newBuilder.setResponse(
          e.getStackTrace.toString).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }
  }

  override def evaluate(request: EvaluateRequest,
                        responseObserver: StreamObserver[EvaluateResponse]): Unit = {
    val clientUUID = request.getClientuuid
    Log4Error.invalidInputError(ServerUtils.checkClientId(clientNum, clientUUID),
      s"Invalid client ID, should be in range of [1, $clientNum], got $clientUUID")
    val version = request.getVersion
    logger.debug(s"Server received Evaluate request of version: $version")
    if (!evalBufferMap.containsKey(clientUUID)) {
      evalBufferMap.put(clientUUID, new util.ArrayList[BoostEval]())
    }
    val evalBuffer = evalBufferMap.get(clientUUID)
    try {
      // If not last batch, add to buffer, else put data into data map and trigger aggregate
      evalBuffer.addAll(request.getTreeEvalList)
      if (!request.getLastBatch) {
        logger.info(s"Added evaluate data to buffer, current size: ${evalBuffer.size()}")
        val response = "Add data successfully"
        responseObserver.onNext(
          EvaluateResponse.newBuilder.setResponse(response).setCode(1).build)
        responseObserver.onCompleted()
      }
      else {
        logger.info(s"Last batch data received, put buffer to clientData map in server")
        synchronized {
          if (aggregator.getEvalStorage().version != version) {
            logger.debug(s"Evaluate: server version is " +
              s"${aggregator.getEvalStorage().version}, waiting")
            wait()
          } else {
            notifyAll()
          }
        }
        aggregator.putClientData(FLPhase.EVAL,
          clientUUID, request.getVersion, new DataHolder(evalBuffer))
        evalBuffer.clear()
        val result = aggregator.getResultStorage().serverData
        if (result == null) {
          val response = "Server evaluate complete"
          responseObserver.onNext(EvaluateResponse.newBuilder
            .setResponse(response).setCode(0).build)
          responseObserver.onCompleted()
        }
        else {
          val response = "Download data successfully"
          responseObserver.onNext(
            EvaluateResponse.newBuilder
              .setResponse(response).setData(result).setCode(1).build)
          responseObserver.onCompleted()
        }
      }
    } catch {
      case e: Exception =>
        val errorMsg = ExceptionUtils.getStackTrace(e)
        val response = EvaluateResponse.newBuilder.setResponse(errorMsg).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }
  }

  override def predict(request: PredictRequest,
                       responseObserver: StreamObserver[PredictResponse]): Unit = {
    val clientUUID = request.getClientuuid
    Log4Error.invalidInputError(ServerUtils.checkClientId(clientNum, clientUUID),
      s"Invalid client ID, should be in range of [1, $clientNum], got $clientUUID")
    val predicts: java.util.List[BoostEval] = request.getTreeEvalList
    // TODO: add same logic with evaluate
    try {
      aggregator.putClientData(FLPhase.PREDICT, clientUUID, request.getVersion,
        new DataHolder(predicts))
      val result = aggregator.getResultStorage().serverData
      if (result == null) {
        val response = "Your required data doesn't exist"
        responseObserver.onNext(PredictResponse.newBuilder.setResponse(response).setCode(0).build)
        responseObserver.onCompleted()
      }
      else {
        val response = "Download data successfully"
        responseObserver.onNext(
          PredictResponse.newBuilder.setResponse(response).setData(result).setCode(1).build)
        responseObserver.onCompleted()
      }
    } catch {
      case e: Exception =>
        val error = e.getStackTrace.map(_.toString).mkString("\n")
        logger.error(e.getMessage + "\n" + error)
        val response = PredictResponse.newBuilder.setResponse(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    } finally {

    }
  }

  override def saveServerModel(request: SaveModelRequest,
                               responseObserver: StreamObserver[SaveModelResponse]): Unit = {
    try {
      aggregator.saveModel(request.getModelPath)
      val response = "Save model on server successfully"
      responseObserver.onNext(
        SaveModelResponse.newBuilder.setMessage(response).setCode(1).build)
      responseObserver.onCompleted()
    } catch {
      case e: Exception =>
        val error = e.getStackTrace.map(_.toString).mkString("\n")
        logger.error(e.getMessage + "\n" + error)
        val response = SaveModelResponse.newBuilder.setMessage(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    }

  }

  override def loadServerModel(request: LoadModelRequest,
                               responseObserver: StreamObserver[LoadModelResponse]): Unit = {
    try {
      aggregator.loadModel(request.getModelPath)
      val response = "Save model on server successfully"
      responseObserver.onNext(
        LoadModelResponse.newBuilder.setMessage(response).setCode(1).build)
      responseObserver.onCompleted()
    } catch {
      case e: Exception =>
        val error = e.getStackTrace.map(_.toString).mkString("\n")
        logger.error(e.getMessage + "\n" + error)
        val response = LoadModelResponse.newBuilder.setMessage(e.getMessage).setCode(1).build
        responseObserver.onNext(response)
        responseObserver.onCompleted()
    }

  }

}
