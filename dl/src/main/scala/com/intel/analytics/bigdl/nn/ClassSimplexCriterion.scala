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
package com.intel.analytics.bigdl.nn

import scala.reflect.ClassTag

/**
 * ClassSimplexCriterion implements a criterion for classification.
 * It learns an embedding per class, where each class' embedding is a
 * point on an (N-1)-dimensional simplex, where N is the number of classes.
 * @param nClasses
 * @param sizeAverage
 */
class ClassSimplexCriterion[T: ClassTag](val nClasses: Int, val sizeAverage: Boolean = true)
 (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  require(nClasses > 1, "Required positive integer argument nClasses > 1")
  val gradInput: Tensor[T] = Tensor[T]()

  private val simp = regsplex(nClasses - 1)
  private val simplex = Tensor[T](simp.size(1), nClasses)
  simplex.narrow(2, 1, simp.size(2)).copy(simp)

  @transient
  private var _target: Tensor[T] = null

  private def regsplex(n : Int): Tensor[T] = {
    val a = Tensor[T](n + 1, n)
    var k = 1
    while (k <= n) {
      if (k == 1) a(Array(k, k)) = ev.fromType(1)
      if (k > 1) {
        val value1 = a.narrow(1, k, 1).narrow(2, 1, k - 1).norm(2)
        a(Array(k, k)) = ev.sqrt(ev.minus(ev.fromType(1), ev.times(value1, value1)))
      }
      var c = ev.minus(ev.times(a(Array(k, k)), a(Array(k, k))), ev.fromType(1))
      c = ev.divide(ev.minus(c, ev.fromType(1.0 / n)), a(Array(k, k)))
      a.narrow(1, k + 1, n - k + 1).narrow(2, k, 1).fill(c)
      k += 1
    }
    a
  }

  private def transformTarget(target: Tensor[T]): Unit = {
    require(target.dim() == 1, "1D tensors only!")
    if (null == _target) _target = Tensor[T](nClasses)

    _target.resize(target.size(1), nClasses)
    var i = 1
    while (i <= target.size(1)) {
      _target(i).copy(simplex(ev.toType[Int](target(Array(i)))))
      i += 1
    }
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    transformTarget(target)
    require(input.nElement() == _target.nElement(), "element wrong")

    output = ev.fromType(0)
    input.map(_target, (a, b) => {
      output = ev.plus(output, ev.times(ev.minus(a, b), ev.minus(a, b)))
      a
    })
    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))

    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == _target.nElement())

    gradInput.resizeAs(input).copy(input)
    val norm = if (sizeAverage) 2.0 / input.nElement() else 2
    gradInput.map(_target, (a, b) => ev.times(ev.fromType(norm), ev.minus(a, b)))
    gradInput
  }

  override def toString(): String = {
    s"nn.ClassSimplexCriterion($nClasses, $sizeAverage)"
  }


  override def canEqual(other: Any): Boolean = other.isInstanceOf[ClassSimplexCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ClassSimplexCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        nClasses == that.nClasses &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), nClasses, sizeAverage)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}
