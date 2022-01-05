package com.intel.analytics.bigdl.ppml.vfl.fgboost

import java.util.HashMap

import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLPhase}
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import org.apache.logging.log4j.LogManager

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class VflGBoostAggregator extends Aggregator[Table] {

  val logger = LogManager.getLogger(this.getClass)

  var obj: TreeObjective = new RMSEObjective
  var nLabel = 1

  val trainMap = Map[String, Table]()
  protected val bestSplit = new HashMap[String, DataSplit]
  protected val leafMap = new HashMap[String, TreeLeaves]
  protected val splitMap = new HashMap[String, DataSplit]
  protected val gradMap = new HashMap[String, List[Float]]
  val _evalMap = new HashMap[String, java.util.List[BoostEval]]
  val _predMap = new HashMap[String, java.util.List[BoostEval]]
  val serverTreeLeaves = new ArrayBuffer[Map[Int, Float]]()
  var validationSize = -1
  var basePrediction: Array[Float] = null

  def setObj(objective: TreeObjective): Unit = {
    obj = objective
  }


  def setNumLabel(numLabel: Int): Unit = {
    nLabel = numLabel
  }


  def init(): Unit = {
    val aggData = trainMap.mapValues(_.getTableMap).values.flatMap(_.asScala)
      .map { data =>
        (data._1, data._2.getTensorList.asScala.toArray.map(_.toFloat))
      }.toMap
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
    logger.info("Label" + label.mkString("Array(", ", ", ")"))

    val metaData = TableMetaData.newBuilder()
      .setName("xgboost_grad")
      .setVersion(0)
      .build()
    val aggregatedModel = Table.newBuilder()
      .setMetaData(metaData)
      .putTable("grad", toFloatTensor(grad))
      .putTable("hess", toFloatTensor(hess))
      .putTable("predict", toFloatTensor(basePrediction))
      .putTable("label", toFloatTensor(label))
      .build()
    // Update gradient
    trainStorage.updateStorage(aggregatedModel)
    trainMap.clear()
  }

  def setClientNum(clientNum: Int): this.type = {
    this.clientNum = clientNum
    this
  }

  override def aggregate(flPhase: FLPhase): Unit = {
    // TODO aggregate split
    aggregateSplit()
    if (leafMap.size >= clientNum) {
      aggregateTree()
    }
  }

  def aggEvaluate(agg: Boolean): Unit = {
    if (leafMap.size >= clientNum) {
      aggregateTree()
    }
    val aggPredict = aggregatePredict()
    val newPredict = predictWithTree(aggPredict)
    // Compute new residual
    updateGradient(newPredict)
  }

  def predictWithEncoding(encoding: Array[java.lang.Boolean], treeID: Int): Float = {
    val treeLeaves = serverTreeLeaves(treeID)
    logger.debug("Predict with encoding" + encoding.mkString("Array(", ", ", ")"))
    logger.debug("Tree map encoding" + treeLeaves.mkString("Array(", ", ", ")"))
    var currIndex = 0
    while (!treeLeaves.contains(currIndex)) {
      logger.debug("CurrIndex " + currIndex)
      if (currIndex > encoding.length - 1) {
        logger.error("Exception " + currIndex)
        logger.error("encoding " + encoding.mkString("Array(", ", ", ")"))
        logger.error("Leaves " + treeLeaves.toString)
      }
      if (encoding(currIndex).booleanValue()) {
        currIndex = currIndex * 2 + 1
      } else {
        currIndex = currIndex * 2 + 2
      }
    }
    logger.debug("Reach leaf " + treeLeaves(currIndex))
    treeLeaves(currIndex)
  }


  def predictWithTree(aggPredict: Array[Array[(String, Array[java.lang.Boolean])]]): Array[Float] = {
    logger.info("Predict with new Tree")
    if (aggPredict.head.length == 1) {
      // Last tree
      aggPredict.map { predict =>
        predictWithEncoding(predict.last._2, serverTreeLeaves.length - 1)
      }
    } else {
      // n label tree or full tree
      aggPredict.flatMap { predict =>
        predict.map(x =>
          predictWithEncoding(x._2, x._1.toInt))
      }
    }
  }

  def aggregateTree(): Unit = {
    logger.info(s"Add new Tree ${serverTreeLeaves.length}")
    val treeIndexes = leafMap.asScala.values.head.getLeafIndexList.map(Integer2int).toArray
    val treeOutputs = leafMap.asScala.values.head.getLeafOutputList.map(Float2float).toArray
    val treeLeaves = treeIndexes.zip(treeOutputs).toMap
    // Add new tree leaves to server
    serverTreeLeaves += treeLeaves
    leafMap.clear()
  }

  def updateGradient(newPredict: Array[Float]): Unit = {
    // For XGBoost Regression with squared loss
    // g = y' - y, h = 1
    logger.info("Updating Gradient with new Predict")
    val gradTable = trainStorage.serverData.getTableMap
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

    logger.info(s"========Loss ${loss} =======")
    logger.debug("New Predict" + predict.mkString("Array(", ", ", ")"))
    logger.debug("New Grad" + gradients(0).mkString("Array(", ", ", ")"))

    val metaData = TableMetaData.newBuilder()
      .setName("xgboost_grad")
      .setVersion(trainStorage.version + 1)
      .build()
    val aggregatedModel = Table.newBuilder()
      .setMetaData(metaData)
      .putTable("grad", toFloatTensor(gradients(0)))
      .putTable("predict", toFloatTensor(predict))
      .putTable("hess", toFloatTensor(gradients(1)))
      .putTable("label", toFloatTensor(label))
      .build()
    // Update gradient
    _evalMap.clear()
    trainStorage.updateStorage(aggregatedModel)
  }

  def getTreeNodeId(treeID: String, nodeID: String): String = treeID + "_" + nodeID
  def aggregateSplit(): Unit = {
    var bestGain = Float.MinValue
    if (splitMap.nonEmpty) {
      bestSplit.synchronized {
        splitMap.values.foreach { split =>
          if (split.getGain > bestGain) {
            val id = getTreeNodeId(split.getTreeID, split.getNodeID);
            bestSplit.put(id, split)
            bestGain = split.getGain
          }
        }
        splitMap.clear()
        bestSplit.notifyAll()
      }
    } else {
      if (trainMap.nonEmpty && trainStorage.version == 0) {
        // Initializing gradient
        init()
      }
    }
  }

  def aggregatePredict(): Array[Array[(String, Array[java.lang.Boolean])]] = {
    // Aggregate from temp predict
    logger.info("Aggregate Predict")
    val version = evalStorage.version
    val clients = _evalMap.keys.toIterator

    // extract from evalMap
    val evalResults = _evalMap.mapValues { list =>
      list.asScala.toArray.map { be =>
        be.getEvaluatesList.asScala.toArray.map { treePredict =>
          (treePredict.getTreeID, treePredict.getPredictsList.asScala.toArray)
        }
      }
    }

    val result = evalResults(clients.next())
    // Join predict result.
    while (clients.hasNext) {
      result.zip(evalResults(clients.next())).foreach { be =>
        be._1.zip(be._2).foreach { treePredict =>
          val left = treePredict._1
          val right = treePredict._2
          require(left._1 == right._1, "Tree id miss match." +
            s" Got ${left._1} ${right._1}.")
          left._2.indices.foreach { i =>
            left._2(i) = (left._2(i) && right._2(i))
          }
        }
      }
    }
    result
  }

  def aggPredict(): Unit = {
    _evalMap.putAll(_predMap)
    val aggedPredict = aggregatePredict()
    val newPredict = aggedPredict.zip(basePrediction).map(p =>
      // agged prediction + base prediction
      p._1.map(x => predictWithEncoding(x._2, x._1.toInt)).sum + p._2
    )
    _predMap.clear()
    val metaData = TableMetaData.newBuilder()
      .setName("predictResult")
      .setVersion(predictStorage.version)
      .build()
    val aggResult = Table.newBuilder()
      .setMetaData(metaData)
      .putTable("predictResult", toFloatTensor(newPredict)).build()
    predictStorage.updateStorage(aggResult)
  }
}

