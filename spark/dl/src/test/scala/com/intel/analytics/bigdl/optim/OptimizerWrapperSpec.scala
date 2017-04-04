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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.mkl.MKL
import org.apache.spark.ml.{DLEstimator, Model}
import org.apache.spark.rdd.RDD

@com.intel.analytics.bigdl.tags.Parallel
class OptimizerWrapperSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  before {
    Engine.setNodeAndCore(1, 4)
  }
  "A wrapper" should "executor optimizer" in {
    val ds = new DistributedDataSet[Float] {
      override def originRDD(): RDD[_] = null
      override def data(train: Boolean): RDD[Float] = null
      override def size(): Long = 0
      override def shuffle(): Unit = {}
    }

    val model = Linear[Float](4, 3)
    val criterion = ClassNLLCriterion[Float]()

    val estimator = new DLEstimator(model, criterion, Array(0))


    val res = estimator.fit(null);

    res.isInstanceOf[Model[Float]] should be(true)

  }
  after {
    println("After")
  }
}
