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

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.ps.{AllReduceParameterManager, OneReduceParameterManager}
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.{Engine, RandomGenerator, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FlatSpec, Matchers}

object BetterOptimizerSpec {
  val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(0.0, 1.0, 0.0, 1.0)))
  val output1 = 0.0
  val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 0.0, 1.0, 0.0)))
  val output2 = 1.0
  var plusOne = 0.0
}

class BetterOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter
  with BeforeAndAfterAll {

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  import  EpochOptimizerSpec._
  var sc: SparkContext = null

  var dataSet : ShuffleBatchDataSet[(Int), Double] = null

  var dataSet2 : ShuffleBatchDataSet[(Int), Double] = null

  val (mseModule, mseWeight, mseGradient) = {
    val mlp = new Sequential[Tensor[Double], Tensor[Double], Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val (weight, grad) = mlp.getParameters()
    (mlp, weight, grad)
  }

  val (crnModule, crnWeight, crnGradient) = {
    val mlp = new Sequential[Tensor[Double], Tensor[Double], Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val (weight, grad) = mlp.getParameters()
    (mlp, weight, grad)
  }

  before {
    sc = new SparkContext("local[1]", "EpochOptimizerSpec")

    dataSet = new ShuffleBatchDataSet[(Int), Double](
      sc.makeRDD(0 to (256 * 8), 8),
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          if(i % 2 == 0) {
            target.setValue(i + 1, output1 + plusOne)
            input.select(1, i + 1).copy(input1)
          } else {
            target.setValue(i + 1, output2 + plusOne)
            input.select(1, i + 1).copy(input2)
          }
          i += 1
        }
        (input, target)
      }, 4, 4 * BetterGradAggEpochOptimizer.subModuleNumber, 1)


    dataSet2 = new ShuffleBatchDataSet[(Int), Double](
      sc.makeRDD(0 to (256 * 8), 8),
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          if(i % 2 == 0) {
            target.setValue(i + 1, output1 + plusOne)
            input.select(1, i + 1).copy(input1)
          } else {
            target.setValue(i + 1, output2 + plusOne)
            input.select(1, i + 1).copy(input2)
          }
          i += 1
        }
        (input, target)
      }, 4 * BetterGradAggEpochOptimizer.subModuleNumber,
      4 * BetterGradAggEpochOptimizer.subModuleNumber, 1)

    RandomGenerator.RNG.setSeed(1000)
    mseWeight.fill(0.125)
    crnWeight.fill(0.125)
    plusOne = 0.0
    Engine.setCoreNum(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "An Artificial Neural Network with MSE and LBFGS" should "be trained with good result" in {
    val pm = new AllReduceParameterManager[Double](mseWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    mseWeight.copy(pm.getParameter())
    val result1 = mseModule.forward(input1)
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = mseModule.forward(input2)
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  it should "be same with grad agg optimizer" in {
    val pm = new OneReduceParameterManager[Double](mseWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    val test = pm.getParameter().clone()
    mseWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](mseWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new LBFGS, pm2, dataSet2, new Metrics)
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    println(test)
    println(mseWeight)

    test.map(mseWeight, (a, b) => {
      a should be(b)
      a
    })
  }

  it should "be same with grad agg optimizer with multi stack dataset" in {
    val pm = new OneReduceParameterManager[Double](mseWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    val test = pm.getParameter().clone()
    mseWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](mseWeight, dataSet.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new LBFGS, pm2, dataSet, new Metrics)
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    println(test)
    println(mseWeight)

    test.map(mseWeight, (a, b) => {
      a should be(b)
      a
    })
  }

  "An Artificial Neural Network with MSE and SGD" should "be trained with good result" in {
    val pm = new AllReduceParameterManager[Double](mseWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    mseWeight.copy(pm.getParameter())
    val result1 = mseModule.forward(input1)
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = mseModule.forward(input2)
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  it should "be same with grad agg optimizer" in {
    val pm = new AllReduceParameterManager[Double](mseWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    val test = pm.getParameter()
    mseWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](mseWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new SGD, pm, dataSet2, new Metrics, T("learningRate" -> 20.0))
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    test.map(mseWeight, (a, b) => {
      a should be(b)
      a
    })
  }

  "An Artificial Neural Network with Cross Entropy and LBFGS" should
    "be trained with good result" in {
    plusOne = 1.0
    val pm = new AllReduceParameterManager[Double](crnWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(3)
    optimizer.optimize()

    crnWeight.copy(pm.getParameter())
    val result1 = crnModule.forward(input1)
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = crnModule.forward(input2)
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  it should "be same with grad agg optimizer" in {
    plusOne = 1.0
    val pm = new AllReduceParameterManager[Double](crnWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(3)
    optimizer.optimize()

    val test = pm.getParameter()
    crnWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](crnWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new LBFGS, pm, dataSet2, new Metrics)
    optimizer2.setMaxEpoch(3)
    optimizer2.optimize()

    test.map(crnWeight, (a, b) => {
      a should be(b)
      a
    })
  }

  "An Artificial Neural Network with Cross Entropy and SGD" should
    "be trained with good result" in {
    plusOne = 1.0
    val pm = new AllReduceParameterManager[Double](crnWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    crnWeight.copy(pm.getParameter())
    val result1 = crnModule.forward(input1)
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = crnModule.forward(input2)
    result2.max(1)._2(Array(1)) should be(2.0)
  }


  it should "be same with grad agg optimizer" in {
    plusOne = 1.0
    val pm = new AllReduceParameterManager[Double](crnWeight, dataSet.partitions())
    val optimizer = new BetterGradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(1)
    optimizer.optimize()

    val test = pm.getParameter()
    crnWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](crnWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new SGD, pm, dataSet2, new Metrics, T("learningRate" -> 20.0))
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    test.map(crnWeight, (a, b) => {
      a should be(b)
      a
    })
  }
}
