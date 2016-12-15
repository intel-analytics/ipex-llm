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

import com.intel.analytics.bigdl.dataset.{DataSet => DataSource, Batch}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The class is used as a reference optimizer in local optimizer unit test
 */
class RefLocalOptimizer[T: ClassTag](
  model: Module[T],
  dataset: DataSource[Iterator[Batch[T]]],
  criterion: Criterion[T]
)(implicit ev: TensorNumeric[T]) extends Optimizer[T, Iterator[Batch[T]],
  Iterator[Batch[T]]](model, dataset, criterion) {

  val (w, g) = model.getParameters()

  override def optimize(): Module[T] = {
    val data = dataset.data()
    var count = 0
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    while (!endWhen(state)) {
      val batch = data.next()
      val input = batch.data
      val target = batch.labels
      model.training()
      model.zeroGradParameters()
      val output = model.forward(input).asInstanceOf[Tensor[T]]
      val loss = criterion.forward(output, target)
      model.backward(input, criterion.backward(output, target))
      optimMethod.optimize(_ => (loss, g), w, state)
      count += input.size(1)
      state("neval") = state[Int]("neval") + 1
      println(s"loss is $loss")
      if (count >= dataset.size()) {
        state("epoch") = state[Int]("epoch") + 1
        count = 0
      }
    }

    model
  }
}
