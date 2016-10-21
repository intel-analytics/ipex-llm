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

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.sparkdl.ps.OneReduceParameterManager
import com.intel.analytics.sparkdl.utils.T
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

class EvaluatorSpec extends FlatSpec with Matchers with BeforeAndAfter  {
  var sc: SparkContext = null

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "accuracy on 2d tensor" should "be correct" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 1, 0, 0
    )), 1, Array(8, 4))

    val target = Tensor(Storage(Array[Double](
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
    correct should be(4)
    count should be(8)
  }

  "accuracy on 1d tensor" should "be correct" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1
    )))

    val target1 = Tensor(Storage(Array[Double](
      4
    )))

    val target2 = Tensor(Storage(Array[Double](
      2
    )))

    val (correct1, count1) = EvaluateMethods.calcAccuracy(output, target1)
    correct1 should be(1)
    count1 should be(1)

    val (correct2, count2) = EvaluateMethods.calcAccuracy(output, target2)
    correct2 should be(0)
    count2 should be(1)
  }

  "top5 accuracy on 1d tensor" should "be correct" in {
    val output = Tensor(Storage(Array[Double](
      0.1, 0.2, 0.6, 0.01, 0.005, 0.005, 0.05, 0.03
    )))

    val target1 = Tensor(Storage(Array[Double](
      2
    )))

    val target2 = Tensor(Storage(Array[Double](
      5
    )))

    val target3 = Tensor(Storage(Array[Double](
      3
    )))

    val target4 = Tensor(Storage(Array[Double](
      7
    )))

    val (correct1, count1) = EvaluateMethods.calcTop5Accuracy(output, target1)
    correct1 should be(1)
    count1 should be(1)

    val (correct3, count3) = EvaluateMethods.calcTop5Accuracy(output, target3)
    correct3 should be(1)
    count3 should be(1)

    val (correct4, count4) = EvaluateMethods.calcTop5Accuracy(output, target4)
    correct4 should be(1)
    count4 should be(1)

    val (correct2, count2) = EvaluateMethods.calcTop5Accuracy(output, target2)
    correct2 should be(0)
    count2 should be(1)
  }

  "Top5 accuracy on 2d tensor" should "be correct" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 8, 1, 2, 0, 0, 0,
      0, 1, 0, 0, 2, 3, 4, 6,
      1, 0, 0, 0.6, 0.1, 0.2, 0.3, 0.4,
      0, 0, 1, 0, 0.5, 1.5, 2, 0,
      1, 0, 0, 6, 2, 3, 4, 5,
      0, 0, 1, 0, 1, 1, 1, 1,
      0, 0, 0, 1, 1, 2, 3, 4,
      0, 1, 0, 0, 2, 4, 3, 2
    )), 1, Array(8, 8))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2,
      2,
      4
    )))

    val (correct, count) = EvaluateMethods.calcTop5Accuracy(output, target)
    correct should be(4)
    count should be(8)
  }

  "Train with evaluation" should "be good" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[4]", "EpochOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to 256, 4).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleFullBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2 + 1)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 4, 4, 1)
    val pm = new OneReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 0.01))
    optimizer.setMaxEpoch(200)
    optimizer.addEvaluation("top1", EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(dataSet)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
    sc.stop()
  }
}
