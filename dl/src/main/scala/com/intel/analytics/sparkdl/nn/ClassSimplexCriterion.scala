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
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ClassSimplexCriterion[T: ClassTag](nClasses: Int)
 (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  var gradInput: Tensor[T] = Tensor[T]()
  var sizeAverage = true

  require(nClasses > 1, "Required positive integer argument nClasses > 1")
  val simp = regsplex(nClasses - 1)

  val tmp2 = Tensor[T](simp.size(1), nClasses -simp.size(2)).zero()
  val simplex = Tensor[T](simp.size(1), simp.size(2) + tmp2.size(2))
  simplex.narrow(2, 1, simp.size(2)).copy(simp)
  simplex.narrow(2, simp.size(2) + 1, tmp2.size(2)).copy(tmp2)
  val _target = Tensor[T](nClasses)

  private def regsplex(n : Int): Tensor[T] = {
    val a = Tensor[T](n+1, n).zero()
    val aArray = a.storage().array()
    var k = 1
    while(k < n){
      if (k == 1) a(k)(k) = ev.fromType(1)
      if (k > 1) {
        val tmp = ev.toType[Double](a.narrow(k, 1, k-1).norm())
        a(1)(1) =ev.fromType(math.sqrt(1 - tmp * tmp))
      }

      val tmp1 = a(k)(k).storage().array()
      val c= (ev.toType[Double](tmp1(0)) * ev.toType[Double](tmp1(0)) - 1 - 1/n) / ev.toType[Double](tmp1(0))
      a.narrow(k + 1, n+1, k).fill(ev.fromType(c))
    }
    a
  }

  private def transformTarget(target: Any): Unit ={
    if (target.isInstanceOf[Double]){ //T
      _target.resize(nClasses)
      _target.copy(simplex(target.asInstanceOf[Double].toInt))
    } else if(target.isInstanceOf[Tensor[Double]]){ //T
      require(target.asInstanceOf[Tensor[Double]].dim() == 1, "1D tensors only!")
      val nSamples = target.asInstanceOf[Tensor[Double]].size(1)
      _target.resize(nSamples, nClasses)
      var i = 1
      while (i < nSamples){
        _target(i).copy(simplex(target.asInstanceOf[Double].toInt))
        i += 1
      }
    }
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    transformTarget(target)
    require(input.nElement() == _target.nElement())

    output = ev.fromType[Int](0)
    input.map(_target, (a, b) => {
      output = ev.plus(output, ev.times(ev.minus(a, b), ev.minus(a, b)));
      a
    })
    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))

    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == _target.nElement())

    gradInput.resizeAs(input)
    var norm = ev.fromType[Int](2)
    if (sizeAverage) {
      norm = ev.fromType[Double](2.0 / input.nElement())
    }
    gradInput.copy(input)
    gradInput.map(_target, (a, b) => ev.times(norm, ev.minus(a, b)))
    gradInput
  }
}
