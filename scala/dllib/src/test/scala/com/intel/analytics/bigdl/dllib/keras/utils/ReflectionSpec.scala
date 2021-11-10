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

package com.intel.analytics.bigdl.dllib.keras.utils

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.nn.keras.{Input, KerasLayer, Model}
import com.intel.analytics.bigdl.dllib.nn.{Graph, Linear}
import com.intel.analytics.bigdl.dllib.utils.{Engine, Shape}
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.ZooSpecHelper
import com.intel.analytics.bigdl.dllib.keras.layers.Dense
import com.intel.analytics.bigdl.dllib.keras.layers.utils.{AbstractModuleRef, EngineRef, GraphRef, KerasLayerRef}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

class KerasLayerRefSpec extends ZooSpecHelper {

  "invokeMethod excludeInvalidLayers" should "work properly" in {
    new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .excludeInvalidLayers(Seq(Dense[Float](2)))
  }

  "invokeMethod setInputShape" should "work properly" in {
    new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .setInputShape(Shape(3))
  }

  "invokeMethod setOutputShape" should "work properly" in {
    new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .setOutShape(Shape(3))
  }

  "invokeMethod checkWithCurrentInputShape" should "work properly" in {
    new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .checkWithCurrentInputShape(Shape(3))
  }

  "invokeMethod validateInput" should "work properly" in {
    new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .validateInput(Seq(Dense[Float](3)))
    intercept[Exception] {
      new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
        .validateInput(Seq(Linear[Float](2, 3)))
    }
  }

  "invokeMethod checkDuplicate" should "work properly" in {
    new KerasLayerRef[Float](Dense[Float](2).asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .checkDuplicate(mutable.HashSet(2))
  }
}

class AbstractModuleRefSpec extends ZooSpecHelper {

  "invokeMethod build" should "work properly" in {
    val outputShape = new AbstractModuleRef[Float](Dense[Float](2)
      .asInstanceOf[KerasLayer[Activity, Activity, Float]])
      .build(Shape(-1, 3))
    assert(outputShape == Shape(-1, 2))
  }
}

class GraphRefSpec extends ZooSpecHelper {

  "invokeMethod getOutputs" should "work properly" in {
    val input = Input[Float](inputShape = Shape(3))
    val model = Model(input, Dense[Float](2).inputs(input))
    val outputs = new GraphRef[Float](model.labor.asInstanceOf[Graph[Float]])
      .getOutputs()
    assert(outputs.length == 1)
  }
}

class EngineRefSpec extends ZooSpecHelper {

  private var sc: SparkContext = _

  override def doBefore(): Unit = {
    val conf = new SparkConf()
      .setMaster("local[4]")
    sc = NNContext.initNNContext(conf, appName = "TrainingSpec")
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "invokeMethod set and coreNumber" should "work properly" in {
    Engine.init
    val num = EngineRef.getCoreNumber()
    print(num)
  }
}
