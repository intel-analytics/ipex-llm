/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.spark.rdd.RDD

object TrainTest2 {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
//      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
val sc = Engine.init(4, 1, true)
        .map(conf => { conf.setAppName("Train Vgg on Cifar10")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
          new SparkContext(conf)
        })

      val partitionNum = 12
      Engine.setPartitionNumber(Some(partitionNum))
//      RandomGenerator.RNG.setSeed(10)
      val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(0.0, 1.0, 2.0, 3.0)))
//val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 1.0, 1.0, 1.0)))
      val output1 = 0.0
      val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(4.0, 5.0, 6.0, 7.0)))
//val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 1.0, 1.0, 1.0)))
      val output2 = 1.0
      var plusOne = 0.0
      val batchsize = param.batchSize / 4
      val prepareData: Int => (MiniBatch[Double]) = index => {
        val input = Tensor[Double]().resize(batchsize, 4)
        val target = Tensor[Double]().resize(batchsize)
        var i = 0
        while (i < batchsize) {
          if (i % 2 == 0) {
            target.setValue(i + 1, output1 + plusOne)
            input.select(1, i + 1).copy(input1)
          } else {
            target.setValue(i + 1, output2 + plusOne)
            input.select(1, i + 1).copy(input2)
          }
          i += 1
        }
        MiniBatch(input, target)
      }
//      val rdd = sc.get.parallelize(1 to 256 * param.nodeNumber,
val rdd = sc.get.parallelize(1 to 256 * 4,
        partitionNum).coalesce(4, true).repartition(partitionNum).map(prepareData).cache()
      rdd.count()

      val dataSet = new DistributedDataSet[MiniBatch[Double]] {
        override def originRDD(): RDD[_] = rdd

        override def data(train : Boolean): RDD[MiniBatch[Double]] = rdd

//        override def size(): Long = 256 * param.nodeNumber
override def size(): Long = 256 * 4

        override def shuffle(): Unit = {}
      }

      def mse: Module[Double] = {
        val mlp = new Sequential[Double]
        mlp.add(new Linear(4, 2))
        mlp.add(new Sigmoid)
        mlp.add(new Linear(2, 1))
        mlp.add(new Sigmoid)
        mlp
      }

      val mm = mse
      mm.getParameters()._1.fill(0.125)
      val optimizer = new DistriOptimizer[Double](
        mm,
        dataSet,
        new MSECriterion[Double]())
        .setOptimMethod(new SGD[Double]())
//        .setEndWhen(Trigger.maxIteration(2))
      val model = optimizer.optimize()

      println("model2: " + model.getParameters()._1)
    })
  }
}
