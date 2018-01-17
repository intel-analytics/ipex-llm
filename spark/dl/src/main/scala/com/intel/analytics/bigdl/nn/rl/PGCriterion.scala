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

package com.intel.analytics.bigdl.nn.rl

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The Criterion to compute the negative policy gradient given a
 * multinomial distribution and the sampled action and reward.
 *
 * The input to this criterion should be a 2-D tensor representing
 * a batch of multinomial distribution, the target should also be
 * a 2-D tensor with the same size of input, representing the sampled
 * action and reward/advantage with the index of non-zero element in the vector
 * represents the sampled action and the non-zero element itself represents
 * the reward. If the action is space is large, you should consider using
 * SparseTensor for target.
 *
 * @param inputTransformer an optional transformer that can operate on the
 *                         input distribution, for example you can pass in
 *                         a NegativeEntropyPenalty to penalize the distribution
 *                         of with small entropy to improve exploration.
 */
@SerialVersionUID(- 76404060368920472L)
class PGCriterion[T: ClassTag](
  inputTransformer: Option[AbstractModule[Tensor[T], Tensor[T], T]] = None)
  (implicit ev: TensorNumeric[T])
  extends TensorCriterion[T] {
  private val criterion = {
    val inputTrans = Sequential[T]()
    if (inputTransformer.isDefined) inputTrans.add(inputTransformer.get)
    inputTrans.add(Log[T]())
    // to calculate the negative policy gradient, because we want maximize reward
    inputTrans.add(MulConstant(-1))

    TransformerCriterion[T](DotProductCriterion[T](), Some(inputTrans), None)
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    output = criterion.forward(input, target)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput = criterion.backward(input, target).asInstanceOf[Tensor[T]]
    gradInput
  }
}

object PGCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputTransformer: Option[AbstractModule[Tensor[T], Tensor[T], T]] = None)
    (implicit ev: TensorNumeric[T]): PGCriterion[T] = {
    new PGCriterion(inputTransformer)
  }
}
