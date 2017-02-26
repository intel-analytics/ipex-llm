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

import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Trigger}
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToBatch}
import org.apache.log4j.{Level, Logger}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class SimpleMlpSpec extends FlatSpec with Matchers {
  "A SimpleMlpSpec" should "runs correctly" in {
    Logger.getLogger(getClass).setLevel(Level.INFO)
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"SampleMlp")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val dimInput = 70
    val nHidden = 100
    val maxEpoch = 10

    // make up some data
    val data = sc.range(0, 100000, 1, 1).map { _ =>
      val featureTensor = Tensor[Double](dimInput)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Double](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat())
      new Sample[Double](featureTensor, labelTensor)
    }

    val trainSet = DataSet.rdd(data).transform(SampleToBatch[Double](1000))

    val layer1 = Linear[Double](dimInput, nHidden)

    val layer2 = ReLU[Double]()
    val layer3 = Linear[Double](nHidden, nHidden)

    val layer4 = ReLU[Double]()
    val layer6 = Linear[Double](nHidden, nHidden)

    val layer5 = ReLU[Double]()
    val output = Linear[Double](nHidden, 1)

    val model = Sequential[Double]().
      add(Reshape(Array(dimInput))).
      add(layer1).
      add(layer2).
      add(layer3).
      add(layer4).
      add(layer5).
      add(layer6).
      add(output)

    val state =
      T(
        "learningRate" -> 0.01
      )
    val criterion = MSECriterion[Double]()

    val optimizer = Optimizer[Double, MiniBatch[Double]](model, trainSet, criterion)

    optimizer.
      setState(state).
      setEndWhen(Trigger.maxEpoch(maxEpoch)).
      setOptimMethod(new Adagrad[Double]()).
      optimize()
  }

}
