package com.intel.webscaleml.nn.example

import java.net.URI

import breeze.optimize.BatchSize
import com.intel.ml.linalg.benchmarks.logisticregression.TrainingProto.TrainingSample
import com.intel.webscaleml.nn.example.Utils._
import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric.TensorNumericDouble
import com.intel.webscaleml.nn.tensor.{SparseTensorBLAS, torch, Tensor, SparseTensor}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{Text, BytesWritable, LongWritable}
import org.apache.hadoop.util.LineReader
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.log4j.{Level, Logger}

import scala.collection.mutable.ArrayBuffer
import scopt.OptionParser

object LogisticRegression {

  def getModel(file: String): Module[Double] = {
    val model = torch.load[Module[Double]](file)
    model
  }

  def getModel(featureSize: Int): Module[Double] = {
    val model = new Sequential[Double]()
    model.add(new SparseLinear(featureSize, 1))
    model.add(new Sigmoid())
    model
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
  }

  def train(params: Params) : Int = {

    val conf = new SparkConf().setAppName(s"LogisticRegression, worker: ${params.workerNum}, partition: ${params.partitionNum}, " +
      s"masterConfig: ${params.masterConfig}, workerConfig: ${params.workerConfig}")

    conf.setExecutorEnv("OPENBLAS_MAIN_FREE", " 1")
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_AFFINITY", "compact,1,0")
    conf.setExecutorEnv("MKL_DYNAMIC", "false")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${(if(params.parallelism == 1) 1 else params.parallelism / 2).toString}")
    conf.set("spark.kryoserializer.buffer.max", "2047m")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.shuffle.spill", "false")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10")  // akka networking speed is slow
//    conf.setMaster("local[2]")

    val featureWhiteList = params.folder + "/feature_id_white_list"
    val trainFiles = params.folder + "/tencent-lr"
    val testFiles = params.folder + "/tencent-lr"
    val sc = new SparkContext(conf)
    val whiteList = readFeatureIds(featureWhiteList)
    val bcWhiteList = sc.broadcast(whiteList)
    val dataFormat = params.workerConfig.get[String]("dataFormat").getOrElse("CSR")

    val stackSize = params.workerConfig.get[Int]("stack").getOrElse(1000)
    println(s"statistic analyze")

    val (max, min, nnz, noc, mean, m2n, totalCount) = sc.sequenceFile[LongWritable, BytesWritable](trainFiles, 1).map(l => parse(l, bcWhiteList.value)).filter(l => l != null).coalesce(params.partitionNum, true).mapPartitions[(Array[Double], Array[Double], Array[Int], Array[Int], Array[Double], Array[Double], Int)] { samples =>
      val curMax = new Array[Double](whiteList.size)
      val curMin = new Array[Double](whiteList.size)
      val curNNZ = new Array[Int](whiteList.size)
      val noOneCount = new Array[Int](whiteList.size)
      val curMean = new Array[Double](whiteList.size)
      val curM2N = new Array[Double](whiteList.size)
      var count = 0

      while(samples.hasNext) {
        val sample = samples.next()
        sample._2 match {
          case st : SparseTensor[Double] => {
            val indices = st._indices(0)
            val values = st._values
            var i = 0
            while(i < indices.length) {
              val value = values(i)
              val index = indices(i) - 1
              if(curMax(index) < value) {
                curMax(index) = value
              }
              if(curMin(index) > value) {
                curMin(index) = value
              }
              if(value != 0.0 && value != 1.0) {
                noOneCount(index) += 1
              }

              val prevMean = curMean(index)
              val diff = values(i) - prevMean
              curMean(index) = prevMean + diff / (curNNZ(index) + 1.0)

              curM2N(index) += (values(i) - curMean(index)) * diff

              curNNZ(index) += 1
              i += 1
            }
            count += 1
          }
        }
      }
      Iterator((curMax, curMin, curNNZ, noOneCount, curMean, curM2N, count))
    }.reduce { (a, b) =>
      require(a._1.size == b._1.size)
      val n = a._1.size
      var i = 0
      while(i < n) {
        if(a._3(i) != 0 && b._3(i) != 0) {
          if(a._1(i) < b._1(i))
            a._1(i) = b._1(i)
          if(a._2(i) > b._2(i))
            a._2(i) = b._2(i)
          val deltaMean = b._5(i) - a._5(i)
          val totalNnz = a._3(i) + b._3(i)
          a._5(i) += deltaMean * b._5(i) / totalNnz
          a._6(i) += b._6(i) + deltaMean * deltaMean * a._3(i) * b._3(i) / totalNnz
          a._3(i) += b._3(i)
          a._4(i) += b._4(i)
        } else if(a._3(i) == 0 && b._3(i) != 0) {
          a._1(i) = b._1(i)
          a._2(i) = b._2(i)
          a._3(i) = b._3(i)
          a._4(i) = b._4(i)
          a._5(i) = b._5(i)
          a._6(i) = b._6(i)
        }
        i += 1
      }

      (a._1, a._2, a._3, a._4, a._5, a._6, a._7 + b._7)
    }

    var i = 0
    var blankColumn = 0
    var bitColumn = 0
    var maxData = 0.0
    var minData = 0.0
    var inited = false
    var constant = 0
    var noScaleColum = 0
    while(i < nnz.length) {
      if(nnz(i) == 0) {
        blankColumn += 1
      }

      if(noc(i) == 0) {
        bitColumn += 1
      }

      if(!inited || maxData < max(i)) {
        maxData = max(i)
      }
      if(!inited || minData > min(i)) {
        minData = min(i)
        inited = true
      }

      if(max(i) != 1.0 || min(i) != 0.0) {
        noScaleColum += 1
      }

      if(max(i) == min(i)) {
        constant += 1
      }

      i += 1
    }
    println(s"blank column number is ${blankColumn}; bit column number is ${bitColumn}; total column number is ${nnz.length}; max value is ${maxData}; min value is ${minData}; Constant column is ${constant} No scale column is ${noScaleColum}")
    println("Calculate max and min complete")
    val bcStat = sc.broadcast((max, min, mean, m2n, nnz, totalCount))
    val trainData = sc.sequenceFile[LongWritable, BytesWritable](trainFiles, 1).map(l => parse(l, bcWhiteList.value)).filter(l => l != null).coalesce(params.partitionNum, true).mapPartitions(toTensorRDD(stackSize, bcStat.value._1, bcStat.value._2, bcStat.value._3, bcStat.value._4, bcStat.value._5, bcStat.value._6, dataFormat)(_)).persist()
    trainData.setName("Train Data RDD")
    val validationData = sc.sequenceFile[LongWritable, BytesWritable](testFiles, 1).map(l => parse(l, bcWhiteList.value)).filter(l => l != null).coalesce(params.workerNum, true).mapPartitions(toTensorRDD(stackSize,  bcStat.value._1, bcStat.value._2, bcStat.value._3, bcStat.value._4, bcStat.value._5, bcStat.value._6, dataFormat)(_)).persist()
    validationData.setName("Validation Data RDD")

    val driverConfig = params.masterConfig.clone()
    val workerConfig = params.workerConfig.clone()
    workerConfig("profile") = true

    val communicator = new CompressedCommunicator[Double](trainData, trainData.partitions.length, Some(params.workerNum))
    val dataSets = new ShuffleFullBatchDataSet[(Tensor[Double], Tensor[Double]), Double](trainData, toTensor() _, 1, 1, 1)
    val optimizer = new WeightAvgEpochOptimizer[Double](getModel(whiteList.size), new BCECriterion(), getOptimMethod(params.masterOptM), communicator, dataSets, driverConfig)

    optimizer.setMaxEpoch(params.masterConfig.get[Int]("epoch").getOrElse(10))
    optimizer.setTestInterval(params.valIter)
    optimizer.setPath("./logisticRegression.model")
    optimizer.setEvaluation(EvaluateMethods.calcLrAccuracy(TensorNumericDouble))
    optimizer.setTestDataSet(dataSets)
    conf.set("spark.task.cpus", params.parallelism.toString)
    optimizer.optimize()

    return 0
  }


  def parse(l: (LongWritable, BytesWritable), whiteList: scala.collection.mutable.Map[Int, Int]): (Double, Tensor[Double]) = {
    //    val startTime = System.nanoTime()
    val trainingSample = TrainingSample.parseFrom(l._2.copyBytes())
    val label = trainingSample.getYValue.toDouble
    val indices = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Double]()

    for (i <- 0 until trainingSample.getFeaturesCount) {
      val idValuePair = trainingSample.getFeatures(i)
      val key = idValuePair.getId()
      val value = idValuePair.getValue()
      //            println(key + ":" + value)
      if (value != 0) {
        if (whiteList.size == 0) {
          indices.append(key + 1)
          values.append(value)
        } else if (whiteList.contains(key)) {
          indices.append(whiteList(key) + 1)
          values.append(value)
        }
      }
    }

    val indicesArray = indices.toArray
    val valuesArray = values.toArray

    sortWithIndices(indicesArray, valuesArray)

    val features = torch.Tensor(Array(indicesArray), torch.storage[Double](valuesArray), Array(whiteList.size))
    //    val endTime = System.nanoTime()
    //A funny thing here. If doesn't print here, the job will lost some executors.
    //    for(i <- 0 until indicesArray.length){
    //      print(indicesArray(i) + ":" + valuesArray(i) + " ")
    //    }
    print()
    //    println(s"parse to sparseVector run time is ${((endTime - startTime): Double) / 1e9} second, vector length: ${indices.size}, total features: ${whiteList.size}")
    //    new LabeledPoint(label, new SparseVector(whiteList.size, indices.toArray, values.toArray))
    (label, features)
  }


  def readFeatureIds(fileName: String): scala.collection.mutable.Map[Int, Int] = {
    val featureIds = scala.collection.mutable.Map[Int, Int]()
    val conf = new Configuration()
    if (fileName.startsWith("hdfs"))
      conf.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem")
    else
      conf.set("fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")

    val fileSystem = FileSystem.get(new URI(fileName), conf)
    val path = new Path(fileName)
    val inputStream = fileSystem.open(path)
    val lineReader = new LineReader(inputStream)
    val line = new Text()

    var i = 0
    while(lineReader.readLine(line) != 0) {
      val featureId = Integer.parseInt(line.toString())
      featureIds += (featureId -> i)
      i += 1
    }
    lineReader.close()

    featureIds
  }

  def to2DTensorCoo(input: Seq[(Double, Tensor[Double])]): (Tensor[Double], Tensor[Double]) = {
    val size = input.size
    val featureSize = input(0)._2.size(1)
    var totalElements = 0
    var i = 0
    while (i < size) {
      val record = input(i)._2
      totalElements += record.nElement()
      i += 1
    }

    val featureRowIndices = new Array[Int](totalElements)
    val featureColIndices = new Array[Int](totalElements)
    val featureValues = new Array[Double](totalElements)
    val labels = new Array[Double](size)
    i = 0
    var index = 0
    while (i < size) {
      labels(i) = input(i)._1
      input(i)._2 match {
        case record: SparseTensor[Double] => {
          val length = record._indices(0).length
          System.arraycopy(record._indices(0), 0, featureColIndices, index, length)
          System.arraycopy(record._values.array(), 0, featureValues, index, length)
          var j = 0
          while (j < length) {
            featureRowIndices(index + j) = i + 1
            j += 1
          }
          index += length
        }
      }
      i += 1
    }
    (torch.Tensor(Array(featureRowIndices, featureColIndices), torch.storage[Double](featureValues), Array(size, featureSize)),
      torch.Tensor[Double](torch.storage(labels), 1, Array(size)))
  }


  def to2DTensorCsr(input: Seq[(Double, Tensor[Double])]): (Tensor[Double], Tensor[Double]) = {
    val size = input.size
    val featureSize = input(0)._2.size(1)
    var totalElements = 0
    var i = 0
    while (i < size) {
      val record = input(i)._2
      totalElements += record.nElement()
      i += 1
    }

    val featureRowIndices = new Array[Int](size + 1)
    val featureColIndices = new Array[Int](totalElements)
    val featureValues = new Array[Double](totalElements)
    val labels = new Array[Double](size)
    i = 0
    var index = 0
    featureRowIndices(0) = 1
    while (i < size) {
      labels(i) = input(i)._1
      input(i)._2 match {
        case record: SparseTensor[Double] => {
          val length = record._indices(0).length
          featureRowIndices(i + 1) = length + featureRowIndices(i)
          System.arraycopy(record._values.array(), 0, featureValues, index, length)
          var j = 0
          while (j < length) {
            featureColIndices(index + j) = record._indices(0)(j)
            j += 1
          }
          index += length
        }
      }
      i += 1
    }
    (torch.Tensor(featureRowIndices, featureColIndices, torch.storage[Double](featureValues), Array(size, featureSize)),
      torch.Tensor[Double](torch.storage(labels), 1, Array(size)))
  }

  def toTensorRDD(batchSize: Int, max: Array[Double], min: Array[Double], mean: Array[Double], m2n: Array[Double], nnz: Array[Int], totalCount : Int, dataFormat: String = "CSR")(samples: Iterator[(Double, Tensor[Double])]): Iterator[(Tensor[Double], Tensor[Double])] ={
    val records = new ArrayBuffer[(Double, Tensor[Double])]()
    val tensors = new ArrayBuffer[(Tensor[Double], Tensor[Double])]
    var i = 0
    while(samples.hasNext){
      val sample = samples.next()
      sample._2 match{
        case st: SparseTensor[Double] => {
          val indices = st._indices(0)
          val values = st._values
          var j = 0
          while(j < values.length()){
            val col = indices(j) - 1
            val maxValue = max(indices(j) - 1)
            val minValue = min(indices(j) - 1)
            //if(maxValue != 1.0 || minValue != 0.0) {
            //  values(j) = (values(j) - minValue) / (maxValue - minValue)
            //}
            var variance =
              m2n(col) + mean(col) * mean(col) * nnz(col) * (totalCount - nnz(col)) / totalCount
            variance /= totalCount - 1
            variance = math.sqrt(variance)
            values(j) *= (if (variance != 0.0) 1.0 / variance else 0.0)
            j += 1
          }
        }
      }

      records.append(sample)
      i += 1
      if(i % batchSize == 0){
        tensors.append(if("CSR" == dataFormat)to2DTensorCsr(records) else to2DTensorCoo(records))
        records.clear()
      }
    }
    if(i != 0){
      tensors.append(if("CSR" == dataFormat)to2DTensorCsr(records) else to2DTensorCoo(records))
    }
    tensors.toIterator
  }

  def toTensor()(inputs: Seq[(Tensor[Double], Tensor[Double])], input : Tensor[Double], target : Tensor[Double]): (Tensor[Double], Tensor[Double]) = {
    inputs(0)
  }

  def sortWithIndices(indices: Array[Int], values: Array[Double]): Unit ={
    var i = 0
    while(i < indices.length){
      var j = 0
      while(j < indices.length - 1 - i){
        if(indices(j) > indices(j + 1)){
          val tmp = indices(j)
          indices(j) = indices(j + 1)
          indices(j+1) = tmp
          val valueTmp = values(j)
          values(j) = values(j + 1)
          values(j + 1) = valueTmp
        }
        j += 1
      }
      i += 1
    }
  }
}
