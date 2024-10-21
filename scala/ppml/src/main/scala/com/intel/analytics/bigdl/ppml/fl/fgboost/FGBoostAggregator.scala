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


import com.intel.analytics.bigdl.dllib.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.fl.base.StorageHolder
import com.intel.analytics.bigdl.ppml.fl.common.{Aggregator, FLDataType, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.fl.fgboost.common.{RMSEObjective, TreeObjective}
import com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.fl.generated.FlBaseProto._
import com.intel.analytics.bigdl.ppml.fl.utils.ProtoUtils._
import org.apache.logging.log4j.LogManager

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.FLConfig
import org.apache.commons.io.serialization.ValidatingObjectInputStream
import org.json4s.FileInput

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.file.Files

class FGBoostAggregator(config: FLConfig,
                        validationMethods: Array[ValidationMethod[Float]] = null)
  extends Aggregator {

  val logger = LogManager.getLogger(this.getClass)

  var obj: TreeObjective = new RMSEObjective
  var nLabel = 1

  var serverTreeLeaf = new ArrayBuffer[Map[Int, Float]]()
  var validationSize = -1
  var basePrediction: Array[Float] = null
  val bestSplit = new java.util.HashMap[String, DataSplit]()
  var validationResult = Array[ValidationResult]()

  // wrapper methods to simplify data access
  def getLabelStorage(): Storage[TensorMap] =
    aggregateTypeMap.get(FLPhase.LABEL).getTensorMapStorage()
  def getSplitStorage(): Storage[DataSplit] =
    aggregateTypeMap.get(FLPhase.SPLIT).getSplitStorage()
  def getTreeLeafStorage(): Storage[TreeLeaf] =
    aggregateTypeMap.get(FLPhase.TREE_LEAF).getLeafStorage()
  def getEvalStorage(): Storage[java.util.List[BoostEval]] =
    aggregateTypeMap.get(FLPhase.EVAL).getTreeEvalStorage()
  def getPredictStorage(): Storage[java.util.List[BoostEval]] =
    aggregateTypeMap.get(FLPhase.PREDICT).getTreeEvalStorage()
  def getResultStorage(): Storage[TensorMap] =
    aggregateTypeMap.get(FLPhase.RESULT).getTensorMapStorage()

  def loadModel(modelPath: String): Unit = {
    if (new File(modelPath).exists()) {
      val is = new ValidatingObjectInputStream(new FileInputStream(modelPath))
      is.accept(classOf[ArrayBuffer[Map[Int, Float]]])
      // val is = new ObjectInputStream(new FileInputStream(modelPath))
      serverTreeLeaf = is.readObject().asInstanceOf[ArrayBuffer[Map[Int, Float]]]
    } else {
      logger.warn(s"$modelPath does not exist, will create new model")
    }
  }

  def saveModel(modelPath: String): Unit = {
    if (modelPath != null) {
      logger.info(s"Saving FGBoostAggregator model to ${modelPath}")
      val os = new ObjectOutputStream(new FileOutputStream(modelPath))
      os.writeObject(serverTreeLeaf)
      os.close()
    }
  }

  override def initStorage(): Unit = {
    aggregateTypeMap.put(FLPhase.LABEL, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(FLPhase.RESULT, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(FLPhase.SPLIT, new StorageHolder(FLDataType.TREE_SPLIT))
    aggregateTypeMap.put(FLPhase.TREE_LEAF, new StorageHolder(FLDataType.TREE_LEAF))
    aggregateTypeMap.put(FLPhase.PREDICT, new StorageHolder(FLDataType.TREE_EVAL))
    aggregateTypeMap.put(FLPhase.EVAL, new StorageHolder(FLDataType.TREE_EVAL))
  }

  override def aggregate(flPhase: FLPhase): Unit = {
    flPhase match {
      case FLPhase.LABEL => initGradient()
      case FLPhase.SPLIT => aggregateSplit()
      case FLPhase.TREE_LEAF => aggregateTreeLeaf()
      case FLPhase.EVAL => aggEvaluate()
      case FLPhase.PREDICT => aggPredict()
      case _ => throw new NotImplementedError()
    }
  }

  def getBestSplit(treeID: String, nodeID: String): DataSplit = {
    synchronized(bestSplit.asScala) {
      val id = getTreeNodeId(treeID, nodeID)
      while ( {
        !bestSplit.containsKey(id)
      }) try bestSplit.wait()
      catch {
        case ie: InterruptedException =>
          throw new RuntimeException(ie)
      }
      return bestSplit.get(id)
    }
  }


  def setObj(objective: TreeObjective): Unit = {
    obj = objective
  }


  def setNumLabel(numLabel: Int): Unit = {
    nLabel = numLabel
  }


  def initGradient(): Unit = {
    logger.debug(s"Server init gradiant from label")
    val labelClientData = getLabelStorage().clientData
    val aggData = labelClientData.asScala.mapValues(_.getTensorMapMap).values.flatMap(_.asScala)
      .map { data =>
        (data._1, data._2.getTensorList.asScala.toArray.map(_.toFloat))
      }.toMap
    logger.debug(s"$aggData")
    val label = aggData("label")
    validationSize = label.length
    basePrediction = if (aggData.contains("predict")) {
      aggData("predict")
    } else {
      Array.fill[Float](validationSize)(0.5f)
    }
    val gradients = obj.getGradient(basePrediction, label)
    val grad = gradients(0)
    val hess = gradients(1)
    val loss = obj.getLoss(basePrediction, label)

    logger.info(s"====Loss ${loss} ====")
//    logger.info("Label" + label.mkString("Array(", ", ", ")"))

    val metaData = MetaData.newBuilder()
      .setName("xgboost_grad")
      .setVersion(0)
      .build()
    val aggregatedModel = TensorMap.newBuilder()
      .setMetaData(metaData)
      .putTensorMap("grad", toFloatTensor(grad))
      .putTensorMap("hess", toFloatTensor(hess))
      .putTensorMap("predict", toFloatTensor(basePrediction))
      .putTensorMap("label", toFloatTensor(label))
      .putTensorMap("loss", toFloatTensor(Array(loss)))
      .build()
    // Update gradient
    getLabelStorage().clearClientAndUpdateServer(aggregatedModel)
    logger.debug("Init gradient completed.")
  }

  def setClientNum(clientNum: Int): this.type = {
    this.clientNum = clientNum
    this
  }

  def aggEvaluate(): Unit = {
    val aggPredict = aggregatePredict(FLPhase.EVAL) // array dimension: record * trees
    val newPredict = predictWithTree(aggPredict)
    val targetProto = getLabelStorage().serverData
    // Compute new residual
    updateGradient(newPredict)
  }

  def predictWithEncoding(encoding: Array[java.lang.Boolean], treeID: Int): Float = {
    val treeLeaf = serverTreeLeaf(treeID)
//    logger.debug("Predict with encoding" + encoding.mkString("Array(", ", ", ")"))
//    logger.debug("Tree map encoding" + treeLeaf.mkString("Array(", ", ", ")"))
    // go through the path by encoding to the leaf node
    var currIndex = 0
    while (!treeLeaf.contains(currIndex)) {
//      logger.debug("CurrIndex " + currIndex)
      if (currIndex > encoding.length - 1) {
        logger.error("Exception " + currIndex)
        logger.error("encoding " + encoding.mkString("Array(", ", ", ")"))
        logger.error("Leaves " + treeLeaf.toString)
      }
      if (encoding(currIndex).booleanValue()) {
        currIndex = currIndex * 2 + 1
      } else {
        currIndex = currIndex * 2 + 2
      }
    }
//    logger.debug("Reach leaf " + treeLeaf(currIndex))
    treeLeaf(currIndex)
  }


  def predictWithTree(
        aggPredict: Array[Array[(String, Array[java.lang.Boolean])]]): Array[Float] = {
    logger.debug("Predict with new Tree")
    if (aggPredict.head.length == 1) {
      // Last tree
      aggPredict.map { predict =>
        predictWithEncoding(predict.last._2, serverTreeLeaf.length - 1)
      }
    } else {
      // n label tree or full tree
      aggPredict.flatMap { predict =>
        predict.map(x =>
          predictWithEncoding(x._2, x._1.toInt))
      }
    }
  }

  def aggregateTreeLeaf(): Unit = {
    logger.info(s"Adding new Tree ${serverTreeLeaf.length}")
    val leafMap = getTreeLeafStorage().clientData.asScala

    val treeIndexes = leafMap.values.head.getLeafIndexList.asScala.map(Integer2int).toArray
    val treeOutputs = leafMap.values.head.getLeafOutputList.asScala.map(Float2float).toArray
    val treeLeaf = treeIndexes.zip(treeOutputs).toMap
    // Add new tree leaves to server
    serverTreeLeaf += treeLeaf
    leafMap.clear()
    getEvalStorage().version += 1
  }

  def updateGradient(newPredict: Array[Float]): Unit = {
    // For XGBoost Regression with squared loss
    // g = y' - y, h = 1
    logger.info(s"Updating Gradient with new Predict, new predict size: ${newPredict.size}")
    val tableStorage = getLabelStorage()
    val gradTable = tableStorage.serverData.getTensorMapMap
    val predict = gradTable.get("predict").getTensorList.asScala.toArray.map(_.toFloat)
    val label = gradTable.get("label").getTensorList.asScala.toArray.map(_.toFloat)
    // Update Predict
    for (i <- predict.indices) {
      predict(i) = predict(i) + newPredict(i)
    }
    // Compute Gradients
    val gradients = obj.getGradient(predict, label)
    // Compute loss
    val loss = obj.getLoss(predict, label)

    logger.info(s"New predict loss: ${loss}")
//    logger.debug("New Predict" + predict.mkString("Array(", ", ", ")"))
//    logger.debug("New Grad" + gradients(0).mkString("Array(", ", ", ")"))

    val metaData = MetaData.newBuilder()
      .setName("xgboost_grad")
      .setVersion(tableStorage.version + 1)
      .build()
    val aggregatedModel = TensorMap.newBuilder()
      .setMetaData(metaData)
      .putTensorMap("grad", toFloatTensor(gradients(0)))
      .putTensorMap("predict", toFloatTensor(predict))
      .putTensorMap("hess", toFloatTensor(gradients(1)))
      .putTensorMap("label", toFloatTensor(label))
      .putTensorMap("loss", toFloatTensor(Array(loss)))
      .build()
    // Update gradient

    tableStorage.clearClientAndUpdateServer(aggregatedModel)
    logger.info(s"current label storage version: ${getLabelStorage().version}")
  }

  def getTreeNodeId(treeID: String, nodeID: String): String = treeID + "_" + nodeID
  def aggregateSplit(): Unit = {
    val splitMap = getSplitStorage().clientData
    var bestGain = Float.MinValue
    bestSplit.synchronized {
      splitMap.values.asScala.foreach { split =>
        if (split.getGain > bestGain) {
          val id = getTreeNodeId(split.getTreeID, split.getNodeID);
          bestSplit.put(id, split)
          bestGain = split.getGain
        }
      }
      splitMap.clear()
      bestSplit.notifyAll()
    }
    getSplitStorage().version += 1
  }


  def aggregatePredict(flPhase: FLPhase): Array[Array[(String, Array[java.lang.Boolean])]] = {
    // get proto and convert to scala object
//    logger.info("Aggregate Predict")
    val boostEvalBranchMap = flPhase match {
      case FLPhase.EVAL =>
        getEvalStorage().clientData
      case FLPhase.PREDICT =>
        getPredictStorage().clientData
      case _ => throw new IllegalArgumentException()
    }
    val evalResults = boostEvalBranchMap.asScala.mapValues { list =>
      list.asScala.toArray.map { be =>
        be.getEvaluatesList.asScala.toArray.map { treePredict =>
          (treePredict.getTreeID, treePredict.getPredictsList.asScala.toArray)
        }
      }
    }
    val clientsIterator = boostEvalBranchMap.asScala.keys.toIterator
    val result = evalResults(clientsIterator.next())
    while (clientsIterator.hasNext) {
      result.zip(evalResults(clientsIterator.next())).foreach { be =>
        be._1.zip(be._2).foreach { case (clientPredict1, clientPredict2) =>
          Log4Error.unKnowExceptionError(clientPredict1._1 == clientPredict2._1,
            "Tree id miss match." +
            s" Got ${clientPredict1._1} ${clientPredict2._1}.")
          clientPredict1._2.indices.foreach { i =>
            clientPredict1._2(i) = clientPredict1._2(i) && clientPredict2._2(i)
          }
        }
      }
    }
    flPhase match {
      case FLPhase.EVAL =>
        getEvalStorage().clientData.clear()
      case FLPhase.PREDICT =>
        getPredictStorage().clientData.clear()
      case _ => throw new IllegalArgumentException()
    }
    result
  }

  def aggPredict(): Unit = {
    val aggedPredict = aggregatePredict(FLPhase.PREDICT)
    val newPredict = aggedPredict.zip(basePrediction).map(p =>
      // Predict value of each boosting tree
      p._1.map(x => predictWithEncoding(x._2, x._1.toInt)).sum + p._2
    )
    val tableStorage = getResultStorage()
    val metaData = MetaData.newBuilder()
      .setName("predictResult")
      .setVersion(tableStorage.version)
      .build()
    val aggResult = TensorMap.newBuilder()
      .setMetaData(metaData)
      .putTensorMap("predictResult", toFloatTensor(newPredict)).build()
    tableStorage.clearClientAndUpdateServer(aggResult)
    getPredictStorage().version += 1
  }
}

