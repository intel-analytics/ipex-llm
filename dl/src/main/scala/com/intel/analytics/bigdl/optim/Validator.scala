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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, DistributedDataSet, LocalDataSet}

abstract class Validator[T, D](
  model: Module[T],
  dataSet: DataSet[D]
) {
  def test(vMethods: Array[ValidationMethod[T]]): Array[(ValidationResult, ValidationMethod[T])]
}

object Validator {
  def apply[T, D](model: Module[T], dataset: DataSet[D]): Validator[T, D] = {
    dataset match {
      case d: DistributedDataSet[MiniBatch[T]] =>
        new DistriValidator[T](
          model = model,
          dataSet = d
        ).asInstanceOf[Validator[T, D]]
      case d: LocalDataSet[MiniBatch[T]] =>
        new LocalValidator[T](
          model = model,
          dataSet = d
        ).asInstanceOf[Validator[T, D]]
      case _ =>
        throw new UnsupportedOperationException
    }
  }
}
