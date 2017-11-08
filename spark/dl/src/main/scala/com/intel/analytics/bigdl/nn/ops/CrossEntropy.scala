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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Compute the cross entropy loss and the gradients.
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
class CrossEntropy[T: ClassTag](implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T] {

  private var buffer: Tensor[T] = _
  private var prob: Tensor[T] = _

  override def updateOutput(input: Table): Table = {
    val modelOutput = input[Tensor[T]](1)
    val label = input[Tensor[T]](2)

    require(modelOutput.nDimension() == 2, "CrossEntropy need a 2D input")
    require(modelOutput.isSameSizeAs(label), s"size not match output" +
      s"(${modelOutput.size().mkString("x")}) label(${label.size().mkString("x")})")
    val batch = modelOutput.size(1)
    if (!output.contains(1)) {
      output(1) = Tensor[T](batch)
      output(2) = Tensor[T]().resizeAs(modelOutput)
    }

    val loss = output[Tensor[T]](1)
    val grad = output[Tensor[T]](2)
    var i = 1
    while(i <= batch) {
      val (l, g) = xEntropy(modelOutput.select(1, i), label.select(1, i))
      loss.setValue(i, l)
      grad.select(1, i).copy(g)
      i += 1
    }

    output
  }

  private def xEntropy(logits: Tensor[T], label: Tensor[T]): (T, Tensor[T]) = {
    if (buffer == null) {
      buffer = Tensor[T]().resizeAs(logits)
      prob = Tensor[T]().resizeAs(logits)
    }

    // max_logits
    val max = logits.max()

    // logits - max_logits
    buffer.fill(ev.negative(max))
    buffer.add(logits)

    // exp(logits - max_logits)
    buffer.exp()
    prob.copy(buffer)

    // sum(exp(logits - max_logits))))
    val sum = buffer.sum()
    // log(sum(exp(logits - max_logits)))))
    val logSum = ev.log(sum)

    // (logits - max_logits)
    buffer.fill(ev.negative(max))
    buffer.add(logits)

    prob.div(sum)

    // (logits - max_logits) - log(sum(exp(logits - max_logits)))
    buffer.add(ev.negative(logSum))

    // sum(-labels *((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    (ev.negative(buffer.cmul(label).sum()), prob.add(ev.negative(ev.one), label))
  }
}

object CrossEntropy {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): CrossEntropy[T] =
    new CrossEntropy()
}
