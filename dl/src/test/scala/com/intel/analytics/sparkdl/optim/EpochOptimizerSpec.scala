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
import com.intel.analytics.sparkdl.utils.{RandomGenerator, Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class EpochOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = null

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "An Artificial Neural Network with MSE and LBFGS" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    RandomGenerator.RNG.setSeed(1000)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 4, 4, 1)
    val pm = new OneReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new MSECriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  it should "be trained with good result for all reduce parameter manager" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    Engine.setCoreNum(1000)
    RandomGenerator.RNG.setSeed(1000)
    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 2).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 4, 4, 1)
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new MSECriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(3)
    optimizer.optimize()

    parameters.copy(pm.getParameter())
    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  "An Artificial Neural Network with MSE and SGD" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 4, 4, 1)
    val pm = new OneReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new MSECriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  it should "be trained with good result with all reduce" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 2).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 4, 4, 1)
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new MSECriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(3)
    optimizer.optimize()

    parameters.copy(pm.getParameter())
    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  "An Artificial Neural Network with Cross Entropy and LBFGS" should
    "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
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
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  it should "be trained with good result for all reduce" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 2).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
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
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new LBFGS, pm, dataSet, new Metrics)
    optimizer.setMaxEpoch(3)
    optimizer.optimize()

    parameters.copy(pm.getParameter())
    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  "An Artificial Neural Network with Cross Entropy and SGD" should
    "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
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
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  it should "be trained with good result for all reduce" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 3).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
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
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(3)
    optimizer.optimize()

    parameters.copy(pm.getParameter())

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  "An Artificial Neural Network with MSE and SGD in weight avg" should
    "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "DistributedOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256, 1).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 256, 256, 1)
    val pm = new OneReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new MSECriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2(Array(1)) should be(1.0 +- 5e-2)

    sc.stop()
  }

  it should "be trained with good result for all reduce" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "DistributedOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256, 1).map { index =>
      if (index % 2 == 0) {
        (input1, output1)
      } else {
        (input2, output2)
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    // Train the Model with toy data
    val parameters = mlp.getParameters()._1
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          target.setValue(i + 1, seq(i)._2)
          System.arraycopy(seq(i)._1, 0, input.storage().array(),
            input.storageOffset() - 1 + i * 4, 4)
          i += 1
        }
        (input, target)
      }, 256, 256, 1)
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new MSECriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    parameters.copy(pm.getParameter())

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2(Array(1)) should be(1.0 +- 5e-2)

    sc.stop()
  }

  "An Artificial Neural Network with NLL and SGD in weight avg" should
    "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[4]", "EpochOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256, 4).map { index =>
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
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
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
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(200)
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
    sc.stop()
  }

  it should "be trained with good result for all reduce" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    Engine.setCoreNum(1000)
    sc = new SparkContext("local[4]", "EpochOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1: Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2: Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256, 4).map { index =>
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
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data,
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
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(200)
    optimizer.optimize()

    parameters.copy(pm.getParameter())
    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
    sc.stop()
  }

  "An Artificial Neural Network with NLL and SGD in weight avg full partition" should
    "be trained with good result" in {
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
    optimizer.optimize()

    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
    sc.stop()
  }

  it should "be trained with good result for all reduce" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[4]", "EpochOptimizerSpec")

    Engine.setCoreNum(1000)
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
    val pm = new AllReduceParameterManager[Double](parameters, dataSet.partitions())
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new ClassNLLCriterion,
      new SGD, pm, dataSet, new Metrics, T("learningRate" -> 0.01))
    optimizer.setMaxEpoch(200)
    optimizer.optimize()

    parameters.copy(pm.getParameter())
    val result1 = mlp.forward(Tensor(Storage(input1)))
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = mlp.forward(Tensor(Storage(input2)))
    result2.max(1)._2(Array(1)) should be(2.0)
    sc.stop()
  }
}
