package com.intel.analytics.dllib.lib.dataSet

import com.intel.analytics.dllib.lib.nn.{Module, ClassNLLCriterion, Criterion}
import com.intel.analytics.dllib.lib.optim._
import com.intel.analytics.dllib.lib.tensor.{T, Table, Tensor}
import scopt.OptionParser

import scala.reflect.ClassTag


/*
*(1) train files, test files, labelsFile in the same folder path, and train files in folder/train,
* test files in folder/validation, labelsFile in folder/labels.
*
*/

object Utils {
  case class Params(
     folder : String = "./",
     dataSetType : String = "ImageNet",
     masterOptM : String = "null",
     workerOptM : String = "null",
     net : String = "ann",
     distribute : String = "parallel",
     partitionNum : Int = 4,
     workerNum : Int = 8,
     valIter : Int = 20,
     masterConfig : Table = T(),
     workerConfig : Table = T(),
     isLocalSpark : Boolean = true,
     parallelism : Int = 1,
     classNum : Int = 100,
     labelsFile : String = "./labels",
     dataType : String = "float",
     testInterval : Int = 0,
     trainSize : Int = 0,
     testSize : Int = 0
   )

  val defaultParams = Params()

  def getParser() : OptionParser[Params] = {
    new OptionParser[Params]("Neural network") {
      head("Neural network examples")
      opt[String]('o', "masterOptM")
        .text("optm : sgd | lbfgs | adagrad")
        .action((x, c) => c.copy(masterOptM = x.toLowerCase))
      opt[String]("workerOptM")
        .text("optm : sgd | lbfgs")
        .action((x, c) => c.copy(workerOptM = x.toLowerCase))
      opt[Double]('r', "learningRate")
        .text("master learning rate")
        .action((x, c) => {c.masterConfig("learningRate") = x; c})
      opt[String]("dataFormat")
        .text("dataFormat")
        .action((x, c) => {c.workerConfig("dataFormat") = x; c})
      opt[Double]("workerLearningRate")
        .text("worker learning rate")
        .action((x, c) => {c.workerConfig("learningRate") = x; c})
      opt[Double]("learningRateDecay")
        .text("master learning rate decay")
        .action((x, c) => {c.masterConfig("learningRateDecay") = x; c})
      opt[Double]("workerlearningRateDecay")
        .text("worker learning rate decay")
        .action((x, c) => {c.workerConfig("learningRateDecay") = x; c})
      opt[Double]("momentum")
        .text("master momentum")
        .action((x, c) => {c.masterConfig("momentum") = x; c})
      opt[Double]("dampening")
        .text("master dampening")
        .action((x, c) => {c.masterConfig("dampening") = x; c})
      opt[Double]("workerMomentum")
        .text("worker momentum")
        .action((x, c) => {c.workerConfig("momentum") = x; c})
      opt[Double]("workerDampening")
        .text("worker dampening")
        .action((x, c) => {c.workerConfig("dampening") = x; c})
      opt[Double]("weightDecay")
        .text("master weightDecay")
        .action((x, c) => {c.masterConfig("weightDecay") = x; c})
      opt[Double]("workerWeightDecay")
        .text("worker weightDecay")
        .action((x, c) => {c.workerConfig("weightDecay") = x; c})
      opt[String]('n', "net")
        .text("net type : ann | cnn")
        .action((x, c) => c.copy(net = x))
      opt[Int]('b', "batch")
        .text("batch size")
        .action((x, c) => {c.workerConfig("batch") = x; c})
      opt[Int]('s', "stackSize")
        .text("stack size")
        .action((x, c) => {c.workerConfig("stack") = x; c })
      opt[Int]("innerloop")
        .text("how many times update on worker")
        .action((x, c) => {c.workerConfig("innerloop") = x; c})
      opt[Int]("batchnum")
        .text("how many times update on worker")
        .action((x, c) => {c.workerConfig("batchnum") = x; c})
      opt[Int]("valIter")
        .text("iterations number each validation")
        .action((x, c) => c.copy(valIter = x))
      opt[Int]("epoch")
        .text("epoch number")
        .action((x, c) => {c.masterConfig("epoch") = x; c})
      opt[Int]("maxIter")
        .text("max iteration number")
        .action((x, c) => {c.masterConfig("maxIter") = x; c})
      opt[Int]("iterationBatch")
        .text("batch size in iteration")
        .action((x, c) => {c.workerConfig("iterationBatch") = x; c})
      opt[Double]("moveRate")
        .text("moveRate")
        .action((x, c) => {c.masterConfig("moveRate") = x; c})
      opt[Boolean]("isLocalSpark")
        .text("run on a local spark instance")
        .action((x, c) => c.copy(isLocalSpark = x))
      opt[Int]("parallelism")
        .text("How many cores to run a model")
        .action((x, c) => c.copy(parallelism = x))
      opt[String]("distribute")
        .text("distribute type : parallel | serial | reweight")
        .action((x, c) => c.copy(distribute = x))
      opt[String]("folder")
        .text("train/test file folder")
        .action((x, c) => c.copy(folder = x))
      opt[Int]("workerNum")
        .text(s"worker number, default: ${defaultParams.workerNum}")
        .action((x, c) => c.copy(workerNum = x))
      opt[Int]("partitionNum")
        .text(s"partition number, default: ${defaultParams.partitionNum}")
        .action((x, c) => c.copy(partitionNum = x))
      opt[String]('f', "folder") action ((x, c) => c.copy(folder = x))
      opt[Int]("classNum")
        .text("class number")
        .action((x, c) => c.copy(classNum = x))
      opt[String]("dataType")
        .text("dataType : float | double")
        .action((x, c) => c.copy(dataType = x.toLowerCase))
      //add by zhangli
      opt[Int]("testInterval")
        .text("what interval to test model")
        .action((x, c) => c.copy(testInterval = x))
      opt[Int]("trainSize")
        .text("train Size")
        .action((x, c) => c.copy(trainSize = x))
      opt[Int]("testSize")
        .text("test Size")
        .action((x, c) => c.copy(testSize = x))
      opt[String]("labelsFile")
        .text("label file folder")
        .action((x, c) => c.copy(labelsFile = x))
      opt[String]("dataSetType")
        .text("dataSet Type")
        .action((x, c) => c.copy(dataSetType = x))
    }
  }

  def getOptimMethod(optM : String) : OptimMethod[Double] = {
    optM.toLowerCase match {
      case "lbfgs" => new LBFGS[Double]
      case "sgd" => new SGD
      case "adagrad" => new Adagrad
      case "null" => null
      case _ =>
        throw new UnsupportedOperationException
    }
  }

  def getOptimMethodFloat(optM : String) : OptimMethod[Float] = {
    optM.toLowerCase match {
      case "lbfgs" => new LBFGS[Float]
      case "sgd" => new SGD[Float]
      case "adagrad" => new Adagrad[Float]
      case "null" => null
      case _ =>
        throw new UnsupportedOperationException
    }
  }

}
