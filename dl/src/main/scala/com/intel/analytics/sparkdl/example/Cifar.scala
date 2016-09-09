/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.example

import com.intel.analytics.sparkdl.example.Utils._
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.ps.{OneReduceParameterManager, ParameterManager}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.Table
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


object Cifar {
  val dataOffset = 0
  val classNumber = 10


  def getOptim(model: Module[Double], params: Params, pm: ParameterManager[Double],
    dataSets: DataSet[_, Double] with HasEpoch, config: Table,
    metrics: Metrics): Optimizer[Double] = {
    val optim = params.masterOptM match {
      case "adagrad" => new Adagrad[Double]()
      case "sgd" => new SGD[Double]()
      case "lbfgs" => new LBFGS[Double]()
      case _ =>
        throw new UnsupportedOperationException(s"optim ${params.distribute}")
    }
    println(model)
    params.distribute match {
      case "parallel" =>
        new WeightAvgEpochOptimizer[Double](model, getCriterion, optim, pm, dataSets,
          metrics, config).setMaxEpoch(config.get[Int]("epoch").getOrElse(20))
      case "serial" =>
        new GradAggEpochOptimizer[Double](model, getCriterion, optim, pm, dataSets,
          metrics, config).setMaxEpoch(config.get[Int]("epoch").getOrElse(20))
      case _ =>
        throw new UnsupportedOperationException(s"${params.distribute} optimizer")
    }
  }

  private def train(params: Params) = {
    val conf = new SparkConf().
      setAppName(s"cifar-10, worker: $params.workerNum, partition: $params.partitionNum")
    conf.setExecutorEnv("OPENBLAS_MAIN_FREE", "1")
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_AFFINITY", "compact,1,0")
    conf.setExecutorEnv("MKL_DYNAMIC", "false")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")
    conf.set("spark.kryoserializer.buffer.max", "512m")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.shuffle.spill", "false")

    val sc = new SparkContext(conf)
    val featureShape = Array(3, 32, 32)
    val targetShape = Array(1)

    val driverConfig = params.masterConfig.clone()
    val workerConfig = params.workerConfig.clone()
    val stackSize = 100
    val batchSize = workerConfig.get("batchSize").getOrElse[Int](1)
    val batchNum = workerConfig.get("batchNum").getOrElse[Int](1)


    println(s"load train data")
    val trainData = loadData(params.folder + "/train.vgg.txt", sc, 1, classNumber + 0.5).
      mapPartitions[(Tensor[Double], Tensor[Double])](toTensorRDD(stackSize, featureShape)(_)).
      filter(_._2.nElement() != 0).coalesce(params.partitionNum, true).persist()
    println("Estimate mean and std")
    val totalSize = trainData.count()
    println(s"train data size is ${totalSize}")


    val validationData = loadData(params.folder + "/test.vgg.txt", sc, 1, classNumber + 0.5).
      mapPartitions[(Tensor[Double], Tensor[Double])](toTensorRDD(stackSize, featureShape)(_)).
      filter(_._2.nElement() != 0).coalesce(params.partitionNum, true)
    val adjustedValidationData = trainData.zipPartitions(validationData)((train, test) => test).
      persist()
    adjustedValidationData.setName("Validation Data")
    val size = adjustedValidationData.count()
    println(s"validation data size is ${size}")

    val model = getModel[Double](params.classNum, params.net)
    val parameters = model.getParameters()._1
    val metrics = new Metrics()
    val dataSets = new ShuffleBatchDataSet[(Tensor[Double], Tensor[Double]), Double](trainData,
      toTensor2(), batchSize, batchSize, batchNum)
    val pm = new OneReduceParameterManager[Double](parameters, dataSets.partitions(), metrics)
    val optimizer = getOptim(model, params, pm, dataSets, driverConfig, metrics)
    optimizer.setTestInterval(params.valIter)
    optimizer.setPath("./cifarnet.obj")
    optimizer.addEvaluation("top", EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(dataSets)

    optimizer.optimize()

  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
  }

  def toTensor2()(images: Seq[(Tensor[Double], Tensor[Double])], input: Tensor[Double],
    target: Tensor[Double]): (Tensor[Double], Tensor[Double]) = {
    images(0)
  }

  def toTensor()(images: Seq[(Double, Array[Double])]): (Tensor[Double], Tensor[Double]) = {
    val size = images.size
    val featureShape = Array(3, 32, 32)
    val featureSize = featureShape.product
    val input = torch.Tensor[Double](Array(size) ++ featureShape)
    val target = torch.Tensor[Double](size)
    val features = input.storage().asInstanceOf[Storage[Double]].array()
    val targets = target.storage().asInstanceOf[Storage[Double]].array()
    var i = 0
    while (i < size) {
      val (label, data) = images(i)
      System.arraycopy(data, 0, features, i * data.length, data.length)
      targets(i) = label
      i += 1
    }
    (input, target)
  }

  def toTensorRDD(stackSize: Int, featureShape: Array[Int])(
    samples: Iterator[(Double, Array[Double])]): Iterator[(Tensor[Double], Tensor[Double])] = {
    val records = new ArrayBuffer[(Double, Array[Double])]()
    val tensors = new ArrayBuffer[(Tensor[Double], Tensor[Double])]
    def toTensor(batchSize: Int, featureShape: Array[Int])(
      input: Seq[(Double, Array[Double])]): (Tensor[Double], Tensor[Double]) = {
      val size = batchSize
      val featureShape = Array(3, 32, 32)
      val result = torch.Tensor[Double](Array(size) ++ featureShape)
      val target = torch.Tensor[Double](size)
      val features = result.storage().array()
      val targets = target.storage().array()
      var i = 0
      while (i < size) {
        val (label, data) = input(i)
        System.arraycopy(data, 0, features, i * data.length, data.length)
        targets(i) = label
        i += 1
      }
      (result, target)
    }

    var i = 0
    while (samples.hasNext) {
      val sample = samples.next()
      records.append(sample)
      i += 1
      if (i % stackSize == 0) {
        tensors.append(toTensor(stackSize, featureShape)(records))
        records.clear()
        i = 0
      }
    }
    if (i != 0) {
      tensors.append(toTensor(records.length, featureShape)(records))
    }
    tensors.toIterator
  }

  def computeMean(data: RDD[(Double, Array[Double])]): (Double, Double, Double) = {
    data.map { d =>
      val image = d._2
      require((image.length - dataOffset) % 3 == 0, "invalid data")
      var (sumY, sumU, sumV) = (0.0, 0.0, 0.0)
      var i = dataOffset
      while (i < image.length) {
        sumY += image(i + 2)
        sumU += image(i + 1)
        sumV += image(i + 0)
        i += 3
      }
      val number = (i - dataOffset) / 3.0
      (sumY / number, sumU / number, sumV / number)
    }.reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def computeVar(data: RDD[(Double, Array[Double])],
    meanR: Double, meanG: Double, meanB: Double)
  : (Double, Double, Double) = {
    data.map { d =>
      val image = d._2
      require((image.length - dataOffset) % 3 == 0, "invalid data")
      var (sumY, sumU, sumV) = (0.0, 0.0, 0.0)
      var i = dataOffset
      while (i < image.length) {
        val y = image(i + 2)
        val u = image(i + 1)
        val v = image(i + 0)
        sumY += y * y
        sumU += u * u
        sumV += v * v
        i += 3
      }
      val number = (i - dataOffset) / 3.0
      (math.sqrt(sumY / number), math.sqrt(sumU / number), math.sqrt(sumV / number))
    }.reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def normalization(dataSets: RDD[(Double, Array[Double])],
    totalSize: Long): RDD[(Double, Array[Double])] = {
    val sum = computeMean(dataSets)
    val meanY = sum._1 / totalSize
    val meanU = sum._2 / totalSize
    val meanV = sum._3 / totalSize
    val mean = (meanY, meanU, meanV)
    println(s"meanY is $meanY meanU is $meanU meanV is $meanV")
    val varSum = computeVar(dataSets, meanY, meanU, meanV)
    val varY = varSum._1 / totalSize
    val varU = varSum._2 / totalSize
    val varV = varSum._3 / totalSize
    val std = (varY, varU, varV)
    println(s"varY is $varY varU is $varU varV is $varV")
    println(s"load validation data")

    dataSets.map { v =>
      var i = 0
      while (i < v._2.length) {
        v._2(i + 1) = (v._2(i + 1) - meanU) / -varU
        v._2(i + 2) = (v._2(i + 1) - meanV) / -varV
        i += 3
      }
      (v._1, v._2)
    }
  }

  def nextBatch(dataSets: RDD[(Double, Array[Double])], total: Int,
    metaRDD: RDD[(Int, Tensor[Double], Tensor[Double])], featureShape: Array[Int],
    labelShape: Array[Int], batchSize: Int, innerLoop: Int): RDD[(Tensor[Double],
    Tensor[Double])] = {
    val partitionNumber = dataSets.partitions.length
    dataSets.sample(false, innerLoop * batchSize * partitionNumber.toDouble / total,
      System.nanoTime()).zipPartitions(metaRDD)((dataIterator, metaIterator) => {
      require(metaIterator.hasNext, "empty metaIterator")
      val (length, input, target) = metaIterator.next()
      require(!metaIterator.hasNext, "metaIterator has more than one elements")

      dataIterator.grouped(batchSize).map(seq => {
        val size = seq.size
        require(input.nElement() >= size * featureShape.product)
        require(target.nElement() >= size)
        var i = 0
        val features = input.storage().asInstanceOf[Storage[Double]].array()
        val targets = target.storage().asInstanceOf[Storage[Double]].array()
        seq.foreach { case (label, data) =>
          System.arraycopy(data, 0, features, i * data.length, data.length)
          targets(i) = label
          i += 1
        }
        (input, target)
      })
    })
  }

  def fullBatch(data: RDD[(Double, Array[Double])], total: Int,
    metaRDD: RDD[(Int, Tensor[Double], Tensor[Double])], featureShape: Array[Int],
    labelShape: Array[Int], batchSize: Int, innerLoop: Int): RDD[(Tensor[Double],
    Tensor[Double])] = {
    val partitionNumber = data.partitions.length
    data.zipPartitions(metaRDD)((dataIterator, metaIterator) => {
      require(metaIterator.hasNext, "empty metaIterator")
      val (length, input, target) = metaIterator.next()
      require(!metaIterator.hasNext, "metaIterator has more than one elements")

      dataIterator.grouped(batchSize).map(seq => {
        val size = seq.size
        require(input.nElement() >= size * featureShape.product)
        require(target.nElement() >= size)
        var i = 0
        val features = input.storage().array()
        val targets = target.storage().array()
        seq.foreach { case (label, data) =>
          System.arraycopy(data, 0, features, i * data.length, data.length)
          targets(i) = label
          i += 1
        }
        (input, target)
      })
    })
  }

  def loadData(url: String, sc: SparkContext, numPartition: Int, classes: Double)
  : RDD[(Double, Array[Double])] = {
    sc.textFile(url, numPartition).map(records => {
      val record = records.split("\\|")
      val label = record(0).toDouble
      val image = record.slice(1, record.length).map(_.toDouble)
      (label, image)
    })
  }

  def rgb2yuv(input: Array[Double], pixelNumber: Int): Unit = {
    require(pixelNumber == input.length / 3, "invalid number of pixels")
    var y = 0.0
    var u = 0.0
    var v = 0.0
    var i = 0
    while (i < pixelNumber) {
      y = 0.299 * input(i) + 0.587 * input(i + pixelNumber) + 0.114 * input(i + pixelNumber * 2)
      u = -0.147 * input(i) - 0.289 * input(i + pixelNumber) + 0.436 * input(i + pixelNumber * 2)
      v = 0.615 * input(i) - 0.51499 * input(i + pixelNumber) - 0.10001 * input(i + pixelNumber * 2)
      input(i) = y.toByte
      input(i + pixelNumber) = u.toByte
      input(i + pixelNumber * 2) = v.toByte
      i = i + 1
    }
  }

  def getCriterion(): Criterion[Double] = {
    new ClassNLLCriterion[Double]()
  }

  def getModel(file: String): Module[Double] = {
    val model = torch.load[Module[Double]](file)
    model
  }

  def getModel[T: ClassTag](classNumber: Int, netType: String)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val model = netType match {
      case "vggBnDo" =>
        val vggBnDo = new Sequential[T]()

        def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[T] = {
          vggBnDo.add(new SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
          vggBnDo.add(new SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
          vggBnDo.add(new ReLU[T](true))
          vggBnDo
        }
        convBNReLU(3, 64).add(new Dropout[T]((0.3)))
        convBNReLU(64, 64)
        vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(64, 128).add(new Dropout[T](0.4))
        convBNReLU(128, 128)
        vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(128, 256).add(new Dropout[T](0.4))
        convBNReLU(256, 256).add(new Dropout[T](0.4))
        convBNReLU(256, 256)
        vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(256, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512)
        vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(512, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512)
        vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())
        vggBnDo.add(new View[T](512))

        val classifier = new Sequential[T]()
        classifier.add(new Dropout[T](0.5))
        classifier.add(new Linear[T](512, 512))
        classifier.add(new BatchNormalization[T](512))
        classifier.add(new ReLU[T](true))
        classifier.add(new Dropout[T](0.5))
        classifier.add(new Linear[T](512, classNumber))
        classifier.add(new LogSoftMax[T])
        vggBnDo.add(classifier)

        vggBnDo

      case "vggBn" =>
        val vggBn = new Sequential[T]()

        def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[T] = {
          vggBn.add(new SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
          vggBn.add(new SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
          vggBn.add(new ReLU[T](true))
          vggBn
        }
        convBNReLU(3, 64)
        convBNReLU(64, 64)
        vggBn.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(64, 128)
        convBNReLU(128, 128)
        vggBn.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(128, 256)
        convBNReLU(256, 256)
        convBNReLU(256, 256)
        vggBn.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(256, 512)
        convBNReLU(512, 512)
        convBNReLU(512, 512)
        vggBn.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(512, 512)
        convBNReLU(512, 512)
        convBNReLU(512, 512)
        vggBn.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())
        vggBn.add(new View[T](512))

        val classifier = new Sequential[T]()
        classifier.add(new Linear[T](512, 512))
        classifier.add(new BatchNormalization[T](512))
        classifier.add(new ReLU[T](true))
        classifier.add(new Linear[T](512, classNumber))
        classifier.add(new LogSoftMax[T])
        vggBn.add(classifier)

        vggBn

      case "vggDo" =>
        val vggDo = new Sequential[T]()

        def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[T] = {
          vggDo.add(new SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
          vggDo.add(new ReLU[T](true))
          vggDo
        }
        convBNReLU(3, 64).add(new Dropout[T]((0.3)))
        convBNReLU(64, 64)
        vggDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(64, 128).add(new Dropout[T](0.4))
        convBNReLU(128, 128)
        vggDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(128, 256).add(new Dropout[T](0.4))
        convBNReLU(256, 256).add(new Dropout[T](0.4))
        convBNReLU(256, 256)
        vggDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(256, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512)
        vggDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

        convBNReLU(512, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512).add(new Dropout[T](0.4))
        convBNReLU(512, 512)
        vggDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())
        vggDo.add(new View[T](512))

        val classifier = new Sequential[T]()
        classifier.add(new Dropout[T](0.5))
        classifier.add(new Linear[T](512, 512))
        classifier.add(new ReLU[T](true))
        classifier.add(new Dropout[T](0.5))
        classifier.add(new Linear[T](512, classNumber))
        classifier.add(new LogSoftMax[T])
        vggDo.add(classifier)

        vggDo
      case _ =>
        val model = new Sequential[T]

        /** *
         * https://github.com/torch/demos/blob/master/train-on-cifar/train-on-cifar.lua
         */
        model.add(new SpatialConvolutionMap[T](SpatialConvolutionMap.random[T](3, 16, 1), 5, 5))
        model.add(new Tanh[T]())
        model.add(new SpatialMaxPooling[T](2, 2, 2, 2))
        /* stage 2 : filter bank -> squashing -> max pooling */
        model.add(new SpatialConvolutionMap[T](SpatialConvolutionMap.random[T](16, 256, 4), 5, 5))
        model.add(new Tanh[T]())
        model.add(new SpatialMaxPooling[T](2, 2, 2, 2))
        /* stage 3 : standard 2-layer neural network */
        model.add(new Reshape[T](Array(256 * 5 * 5)))
        model.add(new Linear[T](256 * 5 * 5, 128))
        model.add(new Tanh[T]())
        model.add(new Linear[T](128, classNumber))
        model.add(new LogSoftMax[T]())

        model
    }

    val (masterWeights, masterGrad) = model.getParameters()
    println(s"model length is ${masterWeights.nElement()}")

    model
  }

}
