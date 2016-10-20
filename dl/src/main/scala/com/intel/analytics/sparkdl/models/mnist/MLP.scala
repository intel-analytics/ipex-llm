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

package com.intel.analytics.sparkdl.models.mnist

import com.intel.analytics.sparkdl.nn.{LogSoftMax, _}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object MLP {
  val rowN = 28
  val colN = 28
  val featureSize = rowN * colN
  val classNum = 10

  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val mlp = new Sequential[T]
    val nHidden = featureSize / 2
    mlp.add(new Reshape(Array(featureSize)))
    mlp.add(new Linear(featureSize, nHidden))
    mlp.add(new Tanh)
    mlp.add(new Linear(nHidden, classNum))
    mlp.add(new LogSoftMax)
    mlp
  }
}
