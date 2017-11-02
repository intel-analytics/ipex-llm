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

package com.intel.analytics.bigdl.utils.keras

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.caffe.{CaffeLoader, Customizable}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

class Keras1LoaderSpec extends FlatSpec with Matchers {

  // just use to ensure we can load the model sucessfully
  // and then we can verify the result by kerasModel.predict against bigdlModel.forward
  "load simple module" should "ok" in {
    val module = new KerasLoader(
      "/Users/lizhichao/god/BigDL/spark/dl/src/test/resources/keras/mlp_functional.json")
      .loadModule()
    println(module.toString())
  }

  // TODO: Load module with share weights

  // TODO: Load module with multiple outputs

  // TODO: Load Module with multiple inputs

  // TODO: Load Module from Sequence model
}
