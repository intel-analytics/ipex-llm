package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.nn.{ClassNLLCriterion, LogSoftMax, Linear, Sequential}
import com.intel.analytics.dllib.lib.tensor.{T, torch}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{Matchers, FlatSpec}

class EvaluatorSpec extends FlatSpec with Matchers {
  "accuracy on 2d tensor" should "be correct" in {
    val output = torch.Tensor(torch.storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 1, 0, 0
    )), 1, Array(8, 4))

    val target = torch.Tensor(torch.storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2,
      2,
      4
    )))

    val (correct, count) = EvaluateMethods.calcAccuracy(output, target)
    correct should be (4)
    count should be (8)
  }

  "accuracy on 1d tensor" should "be correct" in {
    val output = torch.Tensor(torch.storage(Array[Double](
      0, 0, 0, 1
    )))

    val target1 = torch.Tensor(torch.storage(Array[Double](
      4
    )))

    val target2 = torch.Tensor(torch.storage(Array[Double](
      2
    )))

    val (correct1, count1) = EvaluateMethods.calcAccuracy(output, target1)
    correct1 should be (1)
    count1 should be (1)

    val (correct2, count2) = EvaluateMethods.calcAccuracy(output, target2)
    correct2 should be (0)
    count2 should be (1)
  }

  "Train with evaluation" should "be good" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[4]", "EpochOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(0 to 256, 4).map{ index =>
      if (index%2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    //Train the Model with toy data
    val communicator = new CompressedCommunicator[Double](data, data.partitions.length)
    val dataSet = new ShuffleFullBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
      val size = seq.size
      input.resize(Array(size, 4))
      target.resize(Array(size))
      var i = 0
      while (i < size) {
        target.setValue(i + 1, seq(i)._2 + 1)
        System.arraycopy(seq(i)._1, 0, input.storage().array(), input.storageOffset() - 1 + i * 4, 4)
        i += 1
      }
      (input, target)
    }, 4, 4, 1)
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new ClassNLLCriterion, new SGD, communicator, dataSet, T("learningRate" -> 0.01))
    optimizer.setMaxEpoch(200)
    optimizer.setEvaluation(EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(dataSet)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1.max(1)._2(Array(1)) should be (1.0)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2.max(1)._2(Array(1)) should be (2.0)
    sc.stop()
  }
}
