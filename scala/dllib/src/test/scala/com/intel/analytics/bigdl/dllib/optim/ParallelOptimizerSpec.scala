/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, MSECriterion}
import com.intel.analytics.bigdl.optim.DistriOptimizerSpecModel.mse
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class ParallelOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  private var sc: SparkContext = _

  before {
    val conf = Engine.createSparkConf()
      .setMaster("local[1]").setAppName("ParallelOptimizerSpec")
    sc = new SparkContext(conf)
    Engine.init
    Engine.setCoreNumber(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Train with parallel" should "work properly" in {
    val input = Tensor[Float](1, 10).fill(1.0f)
    val target = Tensor[Float](1).fill(1.0f)
    val miniBatch = MiniBatch(input, target)
    val model = Linear[Float](10, 2)
    model.getParameters()._1.fill(1.0f)
    val optimMethod = new SGD[Float]()

    val dataSet = DataSet.array(Array(miniBatch), sc)

    val optimizer = new DistriOptimizer[Float](model, dataSet, new ClassNLLCriterion[Float]())
      .setState(T("learningRate" -> 1.0))
      .setEndWhen(Trigger.maxIteration(10))

    optimizer.optimize()

  }

  "Train with parallel" should "have same results as DistriOptimizer" in {

    val input = Tensor[Float](1, 10).fill(1.0f)
    val target = Tensor[Float](1).fill(1.0f)
    val miniBatch = MiniBatch(input, target)
    val model1 = Linear[Float](10, 2)
    model1.getParameters()._1.fill(1.0f)

    val model2 = Linear[Float](10, 2)
    model2.getParameters()._1.fill(1.0f)

    val dataSet = DataSet.array(Array(miniBatch), sc)

    val parallelOptimizer = new DistriOptimizer[Float](model1,
      dataSet, new ClassNLLCriterion[Float]())
      .setState(T("learningRate" -> 1.0))
      .setEndWhen(Trigger.maxIteration(10))

    parallelOptimizer.optimize

    val distriOptimizer = new DistriOptimizer[Float](model2,
      dataSet, new ClassNLLCriterion[Float]())
      .setState(T("learningRate" -> 1.0))
      .setEndWhen(Trigger.maxIteration(10))

    distriOptimizer.optimize

    model1.getParameters()._1 should be (model2.getParameters()._1)

  }

}
