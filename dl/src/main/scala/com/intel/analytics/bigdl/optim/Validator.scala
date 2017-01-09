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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

abstract class Validator[T, D](
  model: Module[T],
  dataSet: DataSet[D]
) {
  def test(vMethods: Array[ValidationMethod[T]]): Array[(ValidationResult, ValidationMethod[T])]

  def assertEngineInited(): Unit = {
    require(Engine.isInitialized, s"you may forget to initialize Engine.")
  }
}

object Validator {
  def apply[T: ClassTag, D](model: Module[T], dataset: DataSet[D])
  (implicit ev: TensorNumeric[T]): Validator[T, D] = {
    dataset match {
      case d: DistributedDataSet[_] =>
        new DistriValidator[T](
          model = model,
          dataSet = d.asInstanceOf[DistributedDataSet[MiniBatch[T]]]
        ).asInstanceOf[Validator[T, D]]
      case d: LocalDataSet[_] =>
        new LocalValidator[T](
          model = model,
          dataSet = d.asInstanceOf[LocalDataSet[MiniBatch[T]]]
        ).asInstanceOf[Validator[T, D]]
      case _ =>
        throw new UnsupportedOperationException
    }
  }
}
