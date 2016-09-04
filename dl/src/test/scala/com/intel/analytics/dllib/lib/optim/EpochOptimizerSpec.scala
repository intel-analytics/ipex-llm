package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.nn._
import com.intel.analytics.dllib.lib.tensor.{T, torch}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, Matchers, FlatSpec}

class EpochOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter  {

  var sc : SparkContext = null

  after {
    if(sc != null) {
      sc.stop()
    }
  }

  "An Artificial Neural Network with MSE and LBFGS" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map{ index =>
      if (index%2 == 0) {
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

    //Train the Model with toy data
    val communicator = new CompressedCommunicator[Double](data, data.partitions.length)
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
      val size = seq.size
      input.resize(Array(size, 4))
      target.resize(Array(size))
      var i = 0
      while(i < size) {
        target.setValue(i + 1, seq(i)._2)
        System.arraycopy(seq(i)._1, 0, input.storage().array(), input.storageOffset() - 1 + i * 4, 4)
        i += 1
      }
      (input, target)
    }, 4, 4, 1)
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new MSECriterion, new LBFGS, communicator, dataSet)
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1(Array(1)) should be (0.0 +- 1e-2)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2(Array(1)) should be (1.0 +- 1e-2)
  }

  "An Artificial Neural Network with MSE and SGD" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map{ index =>
      if (index%2 == 0) {
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

    //Train the Model with toy data
    val communicator = new CompressedCommunicator[Double](data, data.partitions.length)
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
      val size = seq.size
      input.resize(Array(size, 4))
      target.resize(Array(size))
      var i = 0
      while(i < size) {
        target.setValue(i + 1, seq(i)._2)
        System.arraycopy(seq(i)._1, 0, input.storage().array(), input.storageOffset() - 1 + i * 4, 4)
        i += 1
      }
      (input, target)
    }, 4, 4, 1)
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new MSECriterion, new SGD, communicator, dataSet, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1(Array(1)) should be (0.0 +- 5e-2)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2(Array(1)) should be (1.0 +- 5e-2)
  }

  "An Artificial Neural Network with Cross Entropy and LBFGS" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map{ index =>
      if (index%2 == 0) {
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

    //Train the Model with toy data
    val communicator = new CompressedCommunicator[Double](data, data.partitions.length)
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
      val size = seq.size
      input.resize(Array(size, 4))
      target.resize(Array(size))
      var i = 0
      while(i < size) {
        target.setValue(i + 1, seq(i)._2 + 1)
        System.arraycopy(seq(i)._1, 0, input.storage().array(), input.storageOffset() - 1 + i * 4, 4)
        i += 1
      }
      (input, target)
    }, 4, 4, 1)
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new ClassNLLCriterion, new LBFGS, communicator, dataSet)
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1.max(1)._2(Array(1)) should be (1.0)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2.max(1)._2(Array(1)) should be (2.0)
  }

  "An Artificial Neural Network with Cross Entropy and SGD" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(0 to (256 * 8), 257).map{ index =>
      if (index%2 == 0) {
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

    //Train the Model with toy data
    val communicator = new CompressedCommunicator[Double](data, data.partitions.length)
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
      val size = seq.size
      input.resize(Array(size, 4))
      target.resize(Array(size))
      var i = 0
      while(i < size) {
        target.setValue(i + 1, seq(i)._2 + 1)
        System.arraycopy(seq(i)._1, 0, input.storage().array(), input.storageOffset() - 1 + i * 4, 4)
        i += 1
      }
      (input, target)
    }, 4, 4, 1)
    val optimizer = new GradAggEpochOptimizer[Double](mlp, new ClassNLLCriterion, new SGD, communicator, dataSet, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1.max(1)._2(Array(1)) should be (1.0)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2.max(1)._2(Array(1)) should be (2.0)
  }

  "An Artificial Neural Network with MSE and SGD in weight avg" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "DistributedOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(1 to 256, 1).map{ index =>
      if (index%2 == 0) {
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

    //Train the Model with toy data
    val communicator = new CompressedCommunicator[Double](data, data.partitions.length)
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
      val size = seq.size
      input.resize(Array(size, 4))
      target.resize(Array(size))
      var i = 0
      while (i < size) {
        target.setValue(i + 1, seq(i)._2)
        System.arraycopy(seq(i)._1, 0, input.storage().array(), input.storageOffset() - 1 + i * 4, 4)
        i += 1
      }
      (input, target)
    }, 256, 256, 1)
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new MSECriterion, new SGD, communicator, dataSet, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(100)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1(Array(1)) should be (0.0 +- 5e-2)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2(Array(1)) should be (1.0 +- 5e-2)

    sc.stop()
  }

  "An Artificial Neural Network with NLL and SGD in weight avg" should "be trained with good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[4]", "EpochOptimizerSpec")

    //Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    //Generate a toy training data
    val data = sc.makeRDD(1 to 256, 4).map{ index =>
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
    val dataSet = new ShuffleBatchDataSet[(Array[Double], Double), Double](data, (seq, input, target) => {
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
    val optimizer = new WeightAvgEpochOptimizer[Double](mlp, new ClassNLLCriterion, new SGD, communicator, dataSet, T("learningRate" -> 20.0))
    optimizer.setMaxEpoch(200)
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1.max(1)._2(Array(1)) should be (1.0)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2.max(1)._2(Array(1)) should be (2.0)
    sc.stop()
  }

  "An Artificial Neural Network with NLL and SGD in weight avg full partition" should "be trained with good result" in {
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
    optimizer.optimize()

    val result1 = mlp.forward(torch.Tensor(torch.storage(input1)))
    result1.max(1)._2(Array(1)) should be (1.0)

    val result2 = mlp.forward(torch.Tensor(torch.storage(input2)))
    result2.max(1)._2(Array(1)) should be (2.0)
    sc.stop()
  }
}
