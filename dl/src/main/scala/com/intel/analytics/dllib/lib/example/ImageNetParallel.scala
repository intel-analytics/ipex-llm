package com.intel.analytics.dllib.lib.example

import com.intel.analytics.dllib.lib.example.ImageNetUtils._
import com.intel.analytics.dllib.lib.nn._
import com.intel.analytics.dllib.lib.optim.EpochOptimizer.Regime
import com.intel.analytics.dllib.lib.optim._
import com.intel.analytics.dllib.lib.tensor._
import org.apache.hadoop.io.Text
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import com.intel.analytics.dllib.lib.example.Utils._

object ImageNetParallel {

  val dataOffset = 8

  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.dllib.lib.optim").setLevel(Level.INFO)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      params.dataType match {
        case "float" =>
          trainFloat(params)
        case "double" =>
          train(params)
        case _ =>
          ???
      }
    }}.getOrElse{sys.exit(1)}
  }


  def train(params : Params) = {
    val trainFiles = params.folder + "/train"
    val testFiles = params.folder + "/validation"
    val labelFile = params.labelsFile
    val partitionNum = params.partitionNum
    val optType = params.masterOptM
    val netType = params.net
    val classNum = params.classNum

    val conf = new SparkConf().setAppName(s"ImageNet class: ${params.classNum} Parallelism: ${params.parallelism.toString}, partition : ${params.partitionNum}, " +
          s"masterConfig: ${params.masterConfig}, workerConfig: ${params.workerConfig}")

    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_BLOCKTIME", "0")
    conf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10")  // akka networking speed is slow
    val sc = new SparkContext(conf)

    val labelsMap = getLabels(labelFile)
    val trainData = loadCroppedData(trainFiles, sc, labelsMap, classNum + 0.5)
    println(s"Count train data...")
    val totalSize = trainData.count()
    println(s"train data size is ${totalSize}")
    println("Estimate mean and std...")
    val sum = computeMean(trainData)
    val meanR = sum._1 / totalSize
    val meanG = sum._2 / totalSize
    val meanB = sum._3 / totalSize
    val mean = (meanR, meanG, meanB)
    println(s"meanR is $meanR meanG is $meanG meanB is $meanB")
    val varSum = computeVar(trainData, meanR, meanG, meanB)
    val varR = varSum._1 / totalSize
    val varG = varSum._2 / totalSize
    val varB = varSum._3 / totalSize
    val std = (varR, varG, varB)
    println(s"varR is $varR varG is $varG varB is $varB")

    val criterion = new ClassNLLCriterion[Double]()
    val model = netType match {
      case "alexnet" => AlexNet.getModel[Double](classNum)
      case "googlenet" => GoogleNet.getModel[Double](classNum)
      case "googlenet-bn" => GoogleNet.getModel[Double](classNum, "googlenet-bn")
    }
    println(model)

    val driverConfig = params.masterConfig.clone()
    val workerConfig = params.workerConfig.clone()
    workerConfig("profile") = true

    val regime : Array[Regime] = Array(
      Regime(1, 4, T("learningRate" -> 4e-2, "weightDecay" -> 5e-4)),
      Regime(5, 18, T("learningRate" -> 2e-2, "weightDecay" -> 5e-4)),
      Regime(19, 29, T("learningRate" -> 5e-3, "weightDecay" -> 5e-4)),
      Regime(30, 43, T("learningRate" -> 1e-3, "weightDecay" -> 0.0)),
      Regime(44, 52, T("learningRate" -> 5e-4, "weightDecay" -> 0.0)),
      Regime(53, 100000000, T("learningRate" -> 1e-4, "weightDecay" -> 0.0))
    )

    val croppedData = loadCroppedData(trainFiles, sc, labelsMap, classNum + 0.5).coalesce(partitionNum, true)
      .setName("Cropped Train Data").cache()
    println("Cache cropped data...")
    croppedData.count()

    val communicator = new CompressedCommunicator[Double](croppedData, params.partitionNum)
    val dataSets = new ShuffleBatchDataSet[(Double, Array[Byte]), Double](
      croppedData, toTensorWithoutCrop(mean, std),
      params.workerConfig[Int]("batch"), params.workerConfig[Int]("batch"))

    val validation = croppedData.zipPartitions(loadCroppedData(testFiles, sc, labelsMap, classNum + 0.5)
      .coalesce(partitionNum, true))((train, test) => test).cache()
    validation.setName("Validation Data")
    println(s"load validation data")
    val size = validation.count()
    println(s"validation data size is ${size}")

    val testDataSets = new ShuffleBatchDataSet[(Double, Array[Byte]), Double](
      validation, toTensorWithoutCrop(mean, std), 4, 4)

    val optimizer = new GradAggEpochOptimizer[Double](model, criterion, getOptimMethod(params.masterOptM),
      communicator, dataSets, driverConfig)
    optimizer.setRegimes(regime)
    optimizer.setEvaluation(EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(testDataSets)
    optimizer.setMaxEpoch(100)

    conf.set("spark.task.cpus", params.parallelism.toString)
    optimizer.optimize()
  }

  def trainFloat(params : Params) = {
    val trainFiles = params.folder + "/train"
    val testFiles = params.folder + "/validation"
    val labelFile = params.labelsFile
    val partitionNum = params.partitionNum
    val optType = params.masterOptM
    val netType = params.net
    val classNum = params.classNum

    val conf = new SparkConf().setAppName(s"ImageNet class[Float]: ${params.classNum} Parallelism: ${params.parallelism.toString}, partition : ${params.partitionNum}, " +
      s"masterConfig: ${params.masterConfig}, workerConfig: ${params.workerConfig}")

    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_BLOCKTIME", "0")
    conf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10")  // akka networking speed is slow
    val sc = new SparkContext(conf)

    val labelsMap = getLabels(labelFile)
    val trainData = loadCroppedData(trainFiles, sc, labelsMap, classNum + 0.5)
    println(s"Count train data...")
    val totalSize = trainData.count()
    println(s"train data size is ${totalSize}")
    println("Estimate mean and std...")
    val sum = computeMean(trainData)
    val meanR = sum._1.toFloat / totalSize
    val meanG = sum._2.toFloat / totalSize
    val meanB = sum._3.toFloat / totalSize
    val mean = (meanR, meanG, meanB)
    println(s"meanR is $meanR meanG is $meanG meanB is $meanB")
    val varSum = computeVar(trainData, meanR, meanG, meanB)
    val varR = varSum._1.toFloat / totalSize
    val varG = varSum._2.toFloat / totalSize
    val varB = varSum._3.toFloat / totalSize
    val std = (varR, varG, varB)
    println(s"varR is $varR varG is $varG varB is $varB")

    val criterion = new ClassNLLCriterion[Float]()
    val model = netType match {
      case "alexnet" => AlexNet.getModel[Float](classNum)
      case "googlenet" => GoogleNet.getModel[Float](classNum)
      case "googlenet-bn" => GoogleNet.getModel[Float](classNum, "googlenet-bn")
    }
    println(model)

    val driverConfig = params.masterConfig.clone()
    val workerConfig = params.workerConfig.clone()
    workerConfig("profile") = true

    val regime : Array[Regime] = Array(
      Regime(1, 4, T("learningRate" -> 4e-2, "weightDecay" -> 5e-4)),
      Regime(5, 18, T("learningRate" -> 2e-2, "weightDecay" -> 5e-4)),
      Regime(19, 29, T("learningRate" -> 5e-3, "weightDecay" -> 5e-4)),
      Regime(30, 43, T("learningRate" -> 1e-3, "weightDecay" -> 0.0)),
      Regime(44, 52, T("learningRate" -> 5e-4, "weightDecay" -> 0.0)),
      Regime(53, 100000000, T("learningRate" -> 1e-4, "weightDecay" -> 0.0))
    )

    val croppedData = loadCroppedDataFloat(trainFiles, sc, labelsMap, classNum + 0.5).coalesce(partitionNum, true)
      .setName("Cropped Train Data").cache()
    println("Cache cropped data...")
    croppedData.count()

    val communicator = new CompressedCommunicator[Float](croppedData, params.partitionNum)
    val dataSets = new ShuffleBatchDataSet[(Float, Array[Byte]), Float](
      croppedData, toTensorWithoutCropFloat(mean, std),
      params.workerConfig[Int]("batch"), params.workerConfig[Int]("batch"))

    val validation = croppedData.zipPartitions(loadCroppedDataFloat(testFiles, sc, labelsMap, classNum + 0.5)
      .coalesce(partitionNum, true))((train, test) => test).cache()
    validation.setName("Validation Data")
    println(s"load validation data")
    val size = validation.count()
    println(s"validation data size is ${size}")

    val testDataSets = new ShuffleBatchDataSet[(Float, Array[Byte]), Float](
      validation, toTensorWithoutCropFloat(mean, std), 4, 4)

    val optimizer = new GradAggEpochOptimizer[Float](model, criterion, getOptimMethodFloat(params.masterOptM),
      communicator, dataSets, driverConfig)
    optimizer.setRegimes(regime)
    optimizer.setEvaluation(EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(testDataSets)
    optimizer.setMaxEpoch(100)

    conf.set("spark.task.cpus", params.parallelism.toString)
    optimizer.optimize()
  }

  def toTensor2(mean : (Double, Double, Double),
    std : (Double, Double, Double))(images : Seq[(Double, Array[Byte])]) : (Tensor[Double], Tensor[Double]) = {
    val size = images.size
    val featureShape = Array(3, 224, 224)
    val featureSize = featureShape.product
    val input = torch.Tensor[Double](Array(size) ++ featureShape)
    val target = torch.Tensor[Double](size)
    val features = input.storage().array()
    val targets = target.storage().array()
    var i = 0
    while(i < size) {
      val (label, data) = images(i)
      cropDouble(data, input.size(3), input.size(4), mean, std, features, i * featureSize)
      targets(i) = label
      i += 1
    }
    (input, target)
  }

  def toTensorWithoutCrop(mean : (Double, Double, Double),
               std : (Double, Double, Double))(images : Seq[(Double, Array[Byte])], input : Tensor[Double], target : Tensor[Double]) : (Tensor[Double], Tensor[Double]) = {
    val size = images.size
    val featureShape = Array(3, 224, 224)
    val featureSize = featureShape.product
    input.resize(Array(size) ++ featureShape)
    target.resize(Array(size))
    val features = input.storage().array()
    val targets = target.storage().array()
    var i = 0
    while(i < size) {
      val (label, data) = images(i)
      normalizeDouble(data, 224 * 224, mean, std, features, i * featureSize)
      targets(i) = label
      i += 1
    }
    (input, target)
  }

  def toTensorWithoutCropFloat(mean : (Float, Float, Float),
                          std : (Float, Float, Float))(images : Seq[(Float, Array[Byte])], input : Tensor[Float], target : Tensor[Float]) : (Tensor[Float], Tensor[Float]) = {
    val size = images.size
    val featureShape = Array(3, 224, 224)
    val featureSize = featureShape.product
    input.resize(Array(size) ++ featureShape)
    target.resize(Array(size))
    val features = input.storage().array()
    val targets = target.storage().array()
    var i = 0
    while(i < size) {
      val (label, data) = images(i)
      normalizeFloat(data, 224 * 224, mean, std, features, i * featureSize)
      targets(i) = label
      i += 1
    }
    (input, target)
  }

  def computeMean(data : RDD[(Double, Array[Byte])]) : (Double, Double, Double) = {
    data.map(d => ImageNetUtils.computeMean(d._2, 0)).
      reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def computeVar(data : RDD[(Double, Array[Byte])], meanR : Double, meanG : Double, meanB : Double)
  : (Double, Double, Double) = {
    data.map(d => ImageNetUtils.computeVar(d._2, meanR, meanG, meanB, 0)).
      reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def loadData(url : String, sc : SparkContext, labelMap : Map[String, Double], classes : Double)
  : RDD[(Double, Array[Byte])] = {
    val labelMapBroadCast = sc.broadcast(labelMap)
    sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      // extract label
      val localLabelMap = labelMapBroadCast.value
      val label = image._1.toString.split("_")(0)
      require(label.matches("n\\d\\d\\d\\d\\d\\d\\d\\d"), "Invalid label format")
      (localLabelMap(label), image)
    }).filter(_._1 < classes).map(labelAndimage => {
      val data = labelAndimage._2._2.getBytes
      // Hadoop Reader always resuse the same key value reader so let's make a copy of it
      val newData = new Array[Byte](data.length)
      System.arraycopy(data, 0, newData, 0, data.length)

      (labelAndimage._1, newData)
    })
  }

  def loadCroppedData(url : String, sc : SparkContext, labelMap : Map[String, Double], classes : Double)
  : RDD[(Double, Array[Byte])] = {
    val labelMapBroadCast = sc.broadcast(labelMap)
    sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      // extract label
      val localLabelMap = labelMapBroadCast.value
      val label = image._1.toString.split("_")(0)
      require(label.matches("n\\d\\d\\d\\d\\d\\d\\d\\d"), "Invalid label format")
      (localLabelMap(label), image)
    }).filter(_._1 < classes).map(labelAndimage => {
      val data = labelAndimage._2._2.getBytes
      // Hadoop Reader always resuse the same key value reader so let's make a copy of it
      val result = new Array[Byte](3 * 224 * 224)
      crop(data, 224, 224, result)

      (labelAndimage._1, result)
    })
  }

  def loadCroppedDataFloat(url : String, sc : SparkContext, labelMap : Map[String, Double], classes : Double)
  : RDD[(Float, Array[Byte])] = {
    val labelMapBroadCast = sc.broadcast(labelMap)
    sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      // extract label
      val localLabelMap = labelMapBroadCast.value
      val label = image._1.toString.split("_")(0)
      require(label.matches("n\\d\\d\\d\\d\\d\\d\\d\\d"), "Invalid label format")
      (localLabelMap(label), image)
    }).filter(_._1 < classes).map(labelAndimage => {
      val data = labelAndimage._2._2.getBytes
      // Hadoop Reader always resuse the same key value reader so let's make a copy of it
      val result = new Array[Byte](3 * 224 * 224)
      crop(data, 224, 224, result)

      (labelAndimage._1.toFloat, result)
    })
  }
}
