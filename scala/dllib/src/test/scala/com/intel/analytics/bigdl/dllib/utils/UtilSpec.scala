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
package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.DistriOptimizer.{Cache, CacheV1}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Util.{setExtraParametersFromModelRDD, cloneParameters}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.reflect.ClassTag

class UtilSpec extends FlatSpec with Matchers with BeforeAndAfter{

  private var sc: SparkContext = _

  before {
    val conf = Engine.createSparkConf().setAppName("Test")
      .setMaster("local[1]").set("spark.driver.maxResultSize", "2g")
      .set("spark.driver.memory", "10g")
    sc = SparkContext.getOrCreate(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "setExtraParametersFromModelRDD" should "be correct" in {
    val model = new CacheV1(localModels = Array[Module[Float]](new TestModule[Float]()),
      modelWeights = null,
      modelGradients = null,
      localCriterions = null,
      localStates = null,
      moduleTimeList = null,
      localMethods = null,
      optimMethods = null,
      parameterSynchronizer = null)

    val model2 = new TestModule[Float]()
    val modelRDD = sc.parallelize(Array(model)).map(_.asInstanceOf[Cache[Float]])
    setExtraParametersFromModelRDD[Float](modelRDD, model2, 500000)
  }

  "cloneDenseTensorParameters" should "be correct" in {
    val param = new Array[Tensor[Float]](3)
    param(0) = Tensor[Float](400, 2).fill(1f)
    param(1) = Tensor[Float](1).fill(2f)
    param(2) = Tensor[Float](200, 2).fill(3f)

    val params2 = cloneParameters(param)
    params2.length should be(3)
    params2(0).size should be(Array(400, 2))
    params2(1).size should be(Array(1))
    params2(2).size should be(Array(200, 2))

    params2(0).valueAt(1, 1) should be(1f)
    params2(1).valueAt(1) should be(2f)
    params2(2).valueAt(1, 1) should be(3f)

  }

  "cloneCompatParameters" should "be correct" in {
    val param = new Array[Tensor[Float]](3)
    param(0) = Tensor[Float](400, 2).fill(1f)
    param(1) = Tensor[Float](1).fill(2f)
    param(2) = Tensor[Float](200, 2).fill(3f)

    val compatTensor = Module.flatten(param)

    val params2 = cloneParameters(param)
    params2.length should be(3)
    params2(0).size should be(Array(400, 2))
    params2(1).size should be(Array(1))
    params2(2).size should be(Array(200, 2))

    params2(0).valueAt(1, 1) should be(1f)
    params2(1).valueAt(1) should be(2f)
    params2(2).valueAt(1, 1) should be(3f)

    params2(1).storage() should be(params2(0).storage())

  }


  "shift" should "be correct" in {
    Util.shift(Array(1, 2, 3, 4), 1, 1) should be(Array(1, 2, 3, 4))
    Util.shift(Array(1, 2, 3, 4), 1, 3) should be(Array(1, 3, 4, 2))
    Util.shift(Array(1, 2, 3, 4), 3, 1) should be(Array(1, 4, 2, 3))
  }
}

private class TestModule[T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[T], T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    input
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutput
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    val param = new Array[Tensor[T]](3)
    param(0) = Tensor[T](400000, 2)
    param(1) = Tensor[T](1)
    param(2) = Tensor[T](200000, 2)
    param
  }
}

