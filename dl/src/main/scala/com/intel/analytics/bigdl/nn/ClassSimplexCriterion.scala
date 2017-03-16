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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import scala.reflect.ClassTag

/**
 * ClassSimplexCriterion implements a criterion for classification.
 * It learns an embedding per class, where each class' embedding is a
 * point on an (N-1)-dimensional simplex, where N is the number of classes.
 * @param nClasses
 */

@SerialVersionUID(- 8696382776046599502L)
class ClassSimplexCriterion[T: ClassTag](val nClasses: Int)
 (implicit ev: TensorNumeric[T]) extends MSECriterion[T] {

  require(nClasses > 1, "ClassSimplexCriterion: Required positive integer argument nClasses > 1," +
    s"but get nClasses $nClasses")

  private val simp = regsplex(nClasses - 1)
  private val simplex = Tensor[T](simp.size(1), nClasses)
  simplex.narrow(2, 1, simp.size(2)).copy(simp)

  @transient
  private var targetBuffer: Tensor[T] = null

  private def regsplex(n : Int): Tensor[T] = {
    val a = Tensor[T](n + 1, n)
    var k = 1
    val arr = new Array[Int](2)
    while (k <= n) {
      arr(0) = k
      arr(1) = k
      if (k == 1) a(arr) = ev.one
      if (k > 1) {
        val value1 = a.narrow(1, k, 1).narrow(2, 1, k - 1).norm(2)
        a(arr) = ev.sqrt(ev.minus(ev.one, ev.times(value1, value1)))
      }
      var c = ev.minus(ev.times(a(arr), a(arr)), ev.one)
      c = ev.divide(ev.minus(c, ev.fromType(1.0 / n)), a(arr))
      a.narrow(1, k + 1, n - k + 1).narrow(2, k, 1).fill(c)
      k += 1
    }
    a
  }

  private def transformTarget(target: Tensor[T]): Unit = {
    require(target.dim() == 1, s"ClassSimplexCriterion: target should be 1D tensors only!" +
      s"But get ${target.dim()}")
    if (null == targetBuffer) targetBuffer = Tensor[T](nClasses)

    targetBuffer.resize(target.size(1), nClasses)
    var i = 1
    while (i <= target.size(1)) {
      targetBuffer(i).copy(simplex(ev.toType[Int](target(Array(i)))))
      i += 1
    }
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    transformTarget(target)
    require(input.nElement() == targetBuffer.nElement(), "ClassSimplexCriterion: " +
      "element number wrong")
    output = super.updateOutput(input, targetBuffer)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == targetBuffer.nElement())

    gradInput = super.updateGradInput(input, targetBuffer)
    gradInput
  }

  override def toString(): String = {
    s"nn.ClassSimplexCriterion($nClasses)"
  }


  override def canEqual(other: Any): Boolean = other.isInstanceOf[ClassSimplexCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ClassSimplexCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        nClasses == that.nClasses
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), nClasses)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object ClassSimplexCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      nClasses: Int)(implicit ev: TensorNumeric[T]) : ClassSimplexCriterion[T] = {
    new ClassSimplexCriterion[T](nClasses)
  }
}
