package com.intel.analytics.zoo.tutorial

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.spark.SparkConf
import Utils._
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.NNContext._
import com.intel.analytics.bigdl.dllib.optim.SGD
import com.intel.analytics.bigdl.dllib.utils.Shape

object SimpleMlp {

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {
      val dimInput = param.dimInput
      val nHidden = param.nHidden
      val recordSize = param.recordSize
      val maxEpoch = param.maxEpoch
      val batchSize = param.batchSize

      //init spark context
      val conf = new SparkConf()
        .setAppName(s"SampleMlp-$dimInput-$nHidden-$recordSize-$maxEpoch-$batchSize")
      val sc = initNNContext(conf)

      // make up some data
      val data = sc.range(0, recordSize, 1).map { _ =>
        val featureTensor = Tensor[Float](dimInput)
        featureTensor.apply1(_ => scala.util.Random.nextFloat())
        val labelTensor = Tensor[Float](1)
        labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat())
        Sample[Float](featureTensor, labelTensor)
      }

      val model = Sequential()
      model.add(Dense(nHidden, activation = "relu", inputShape = Shape(dimInput)).setName("fc_1"))
      model.add(Dense(nHidden, activation = "relu").setName("fc_2"))
      model.add(Dense(1).setName("fc_3"))

      println(model)

      model.compile(
        optimizer = new SGD(learningRate = 0.01),
        loss = MSECriterion()
      )
      model.fit(data, batchSize = param.batchSize, nbEpoch = param.maxEpoch)

    })
  }
}
