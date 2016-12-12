package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Paths

import com.intel.analytics.bigdl.models.lenet.{LeNet5, DataSet, Options}
import com.intel.analytics.bigdl.models.lenet.Options._
import com.intel.analytics.bigdl.nn.{Criterion, ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, Activities, T}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._


object Train {
  object Local {
    import Options._
    def main(args: Array[String]): Unit = {
      trainLocalParser.parse(args, new TrainLocalParams()).map(param => {
        val batchSize = 128
        val maxEpoch = 90

        val train = Paths.get(param.folder, "/train")
        val validation = Paths.get(param.folder, "/val")
        val trainDataSet = DataSet.localDataSet(train, true, batchSize)

        val model = if(param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          Vgg(classNum = 10)
        }
        val state = if(param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> 0.01,
            "weightDecay" -> 0.0005,
            "momentum" -> 0.9,
            "dampening" -> 0.0,
            "learningRateSchedule" -> SGD.EpochStep(25, 0.5)
          )
        }

        val optimizer = new LocalOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new ClassNLLCriterion[Float]().asInstanceOf[Criterion[Activities, Float]],
          coreNumber = param.coreNumber
        )

        if(param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.everyEpoch)
        }

        val validateDataSet = DataSet.localDataSet(validation, false, batchSize)
        optimizer
          .setValidation(Trigger.everyEpoch, validateDataSet, Array(new Top1Accuracy[Float]))
          .setState(state)
          .setEndWhen(Trigger.maxEpoch(maxEpoch))
          .optimize()
      })
    }
  }

  object Spark {
    import Options._
    def main(args: Array[String]): Unit = {
      trainSparkParser.parse(args, new TrainSparkParams()).map(param => {
        val batchSize = 128
        val maxEpoch = 90

        val train = Paths.get(param.folder, "/train")
        val validation = Paths.get(param.folder, "/val")

        val conf = Engine.sparkConf().setAppName("Train Vgg on Cifar10")
        val sc = new SparkContext(conf)

        val trainDataSet = DataSet.distributedDataSet(train, true, sc, param.nodesNumber, batchSize)

        val model = if(param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          Vgg(classNum = 10)
        }
        val state = if(param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> 0.01,
            "weightDecay" -> 0.0005,
            "momentum" -> 0.9,
            "dampening" -> 0.0,
            "learningRateSchedule" -> SGD.EpochStep(25, 0.5)
          )
        }

        val optimizer = new DistriOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new ClassNLLCriterion[Float]().asInstanceOf[Criterion[Activities, Float]],
          nodeNumber = param.nodesNumber,
          coresPerNode = param.coreNumberPerNode
        )
        val validateDataSet =
          DataSet.distributedDataSet(validation, false, sc, param.nodesNumber, batchSize)
        if(param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.everyEpoch)
        }
        optimizer
          .setValidation(Trigger.everyEpoch, validateDataSet, Array(new Top1Accuracy[Float]))
          .setState(state)
          .setEndWhen(Trigger.maxEpoch(maxEpoch))
          .optimize()
      })
    }
  }
}
