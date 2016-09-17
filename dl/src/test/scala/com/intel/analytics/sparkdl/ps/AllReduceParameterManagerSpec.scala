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

package com.intel.analytics.sparkdl.ps

import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.utils.{Engine, T}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.sparkdl.tensor.Tensor

class AllReduceParameterManagerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = null

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "sync parameter" should "broadcast parameter to given parameter rdd" in {
    sc = new SparkContext("local[4]", "AllReduceParameterManagerSpec")
    val param = Tensor[Double](5)
    param.setValue(1, 10.0)
    param.setValue(2, 10.0)
    param.setValue(3, 10.0)
    param.setValue(4, 10.0)
    param.setValue(5, 10.0)

    Engine.setCoreNum(1000)
    val parameters = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => Iterator.single(Tensor[Double](5).fill(10.0)))
    val pm = new AllReduceParameterManager[Double](param, parameters)
    pm.sync(parameters).collect().map(_ should be(param))
    pm.getParameter() should be(param)
  }

  it should "broadcast parameter to given parameter rdd for float tensor" in {
    sc = new SparkContext("local[4]", "AllReduceParameterManagerSpec")
    val param = Tensor[Float](5)
    param.setValue(1, 10.0f)
    param.setValue(2, 10.0f)
    param.setValue(3, 10.0f)
    param.setValue(4, 10.0f)
    param.setValue(5, 10.0f)

    Engine.setCoreNum(1000)
    val parameters = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => Iterator.single(Tensor[Float](5).fill(10.0f)))
    val pm = new AllReduceParameterManager[Float](param, parameters)
    pm.sync(parameters).collect().map(_ should be(param))
    pm.getParameter() should be(param)
  }

  "sum and update" should "be correct" in {
    sc = new SparkContext("local[4]", "AllReduceParameterManagerSpec")
    val parameters = Tensor[Double](9)
    parameters.setValue(1, 1.0)
    parameters.setValue(2, 2.0)
    parameters.setValue(3, 3.0)
    parameters.setValue(4, 4.0)
    parameters.setValue(5, 5.0)
    parameters.setValue(6, 6.0)
    parameters.setValue(7, 7.0)
    parameters.setValue(8, 8.0)
    parameters.setValue(9, 9.0)

    Engine.setCoreNum(1000)
    val gradients = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => {
        val result = Tensor[Double](9)
        result.setValue(1, 4.0)
        result.setValue(2, 3.0)
        result.setValue(3, 6.0)
        result.setValue(4, 7.0)
        result.setValue(5, 9.0)
        result.setValue(6, 1.0)
        result.setValue(7, 4.0)
        result.setValue(8, 8.0)
        result.setValue(9, 5.0)
        Iterator.single(result)
      })
    val pm = new AllReduceParameterManager[Double](parameters, gradients)
    pm.sumAndUpdate(gradients, (w, g, s) => {
      w.add(g)
    })

    val target = Tensor[Double](9)
    target.setValue(1, 21.0)
    target.setValue(2, 17.0)
    target.setValue(3, 33.0)
    target.setValue(4, 39.0)
    target.setValue(5, 50.0)
    target.setValue(6, 11.0)
    target.setValue(7, 27.0)
    target.setValue(8, 48.0)
    target.setValue(9, 34.0)
    pm.getParameter() should be(target)
  }

  it should "be correct for sgd" in {
    sc = new SparkContext("local[4]", "AllReduceParameterManagerSpec")
    val parameters = Tensor[Double](9)
    parameters.setValue(1, 1.0)
    parameters.setValue(2, 2.0)
    parameters.setValue(3, 3.0)
    parameters.setValue(4, 4.0)
    parameters.setValue(5, 5.0)
    parameters.setValue(6, 6.0)
    parameters.setValue(7, 7.0)
    parameters.setValue(8, 8.0)
    parameters.setValue(9, 9.0)

    val target = parameters.clone()

    val parameterRDD = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => {
        Iterator.single(Tensor[Double](9))
      })

    Engine.setCoreNum(1000)
    val gradients = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => {
        val result = Tensor[Double](9)
        result.setValue(1, 4.0)
        result.setValue(2, 3.0)
        result.setValue(3, 6.0)
        result.setValue(4, 7.0)
        result.setValue(5, 9.0)
        result.setValue(6, 1.0)
        result.setValue(7, 4.0)
        result.setValue(8, 8.0)
        result.setValue(9, 5.0)
        Iterator.single(result)
      })
    val gradient = gradients.first()
    val pm = new AllReduceParameterManager[Double](parameters, gradients)
    val optm = new SGD[Double]()
    pm.sumAndUpdate(gradients, (w, g, s) => {
      val avg = g.div(5)
      optm.optimize(_ => (0.1, avg), w, s)
    })
    pm.sync(parameterRDD).count()
    pm.sumAndUpdate(gradients, (w, g, s) => {
      val avg = g.div(5)
      optm.optimize(_ => (0.1, avg), w, s)
    })

    optm.optimize(_ => (0.1, gradient), target, T())
    val truncateParm = new FP16Parameter(target)
    truncateParm.copyTo(target)
    optm.optimize(_ => (0.1, gradient), target, T())
    truncateParm.copyFrom(target)
    truncateParm.copyTo(target)
    pm.getParameter() should be(target)

  }

  it should "be correct for float parameter" in {
    sc = new SparkContext("local[4]", "AllReduceParameterManagerSpec")

    Engine.setCoreNum(1000)
    val parameters = Tensor[Float](9)
    parameters.setValue(1, 1.0f)
    parameters.setValue(2, 2.0f)
    parameters.setValue(3, 3.0f)
    parameters.setValue(4, 4.0f)
    parameters.setValue(5, 5.0f)
    parameters.setValue(6, 6.0f)
    parameters.setValue(7, 7.0f)
    parameters.setValue(8, 8.0f)
    parameters.setValue(9, 9.0f)

    val gradients = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => {
        val result = Tensor[Float](9)
        result.setValue(1, 4.0f)
        result.setValue(2, 3.0f)
        result.setValue(3, 6.0f)
        result.setValue(4, 7.0f)
        result.setValue(5, 9.0f)
        result.setValue(6, 1.0f)
        result.setValue(7, 4.0f)
        result.setValue(8, 8.0f)
        result.setValue(9, 5.0f)
        Iterator.single(result)
      })
    val pm = new AllReduceParameterManager[Float](parameters, gradients)
    pm.sumAndUpdate(gradients, (w, g, s) => {
      w.add(g)
    })

    val target = Tensor[Float](9)
    target.setValue(1, 21.0f)
    target.setValue(2, 17.0f)
    target.setValue(3, 33.0f)
    target.setValue(4, 39.0f)
    target.setValue(5, 50.0f)
    target.setValue(6, 11.0f)
    target.setValue(7, 27.0f)
    target.setValue(8, 48.0f)
    target.setValue(9, 34.0f)
    pm.getParameter() should be(target)
  }

  it should "state should be saved" in {
    sc = new SparkContext("local[4]", "AllReduceParameterManagerSpec")

    Engine.setCoreNum(1000)
    val parameters = Tensor[Float](9)
    parameters.setValue(1, 1.0f)
    parameters.setValue(2, 2.0f)
    parameters.setValue(3, 3.0f)
    parameters.setValue(4, 4.0f)
    parameters.setValue(5, 5.0f)
    parameters.setValue(6, 6.0f)
    parameters.setValue(7, 7.0f)
    parameters.setValue(8, 8.0f)
    parameters.setValue(9, 9.0f)

    val gradients = sc.parallelize(Seq(1 to 5), 5)
      .mapPartitions(iter => {
        val result = Tensor[Float](9)
        result.setValue(1, 4.0f)
        result.setValue(2, 3.0f)
        result.setValue(3, 6.0f)
        result.setValue(4, 7.0f)
        result.setValue(5, 9.0f)
        result.setValue(6, 1.0f)
        result.setValue(7, 4.0f)
        result.setValue(8, 8.0f)
        result.setValue(9, 5.0f)
        Iterator.single(result)
      })
    val pm = new AllReduceParameterManager[Float](parameters, gradients)
    pm.sumAndUpdate(gradients, (w, g, s) => {
      w.add(g)
      s("test") = 1.0
    })
    pm.sumAndUpdate(gradients, (w, g, s) => {
      w.add(g)
      require(s[Double]("test") == 1.0)
    })

    val target = Tensor[Float](9)
    target.setValue(1, 41.0f)
    target.setValue(2, 32.0f)
    target.setValue(3, 63.0f)
    target.setValue(4, 74.0f)
    target.setValue(5, 95.0f)
    target.setValue(6, 16.0f)
    target.setValue(7, 47.0f)
    target.setValue(8, 88.0f)
    target.setValue(9, 59.0f)
    pm.getParameter() should be(target)
  }

}
