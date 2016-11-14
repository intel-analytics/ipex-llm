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
import com.intel.analytics.sparkdl.utils.{Activities, T, Table}

import scala.reflect.ClassTag

object Utils {

  /**
   * Resize table target as table src.
   * @param target
   * @param src
   */
  def recursiveResizeAs[T : ClassTag](target : Activities, src: Activities)(
    implicit ev: TensorNumeric[T]): Activities = {
    var result: Activities = null
    if (src.isInstanceOf[Table]) {
      val srcTable = src.toTable()
      result = if (target.isInstanceOf[Table]) {
        T(target)
      } else {
        target.toTable()
      }
      val resultTable = result.toTable()
      var i = 1
      while (i <= src.toTable().length()) {
        if (resultTable.contains(i)) {
          resultTable(i) = recursiveResizeAs(resultTable(i), srcTable(i))
        } else {
          resultTable(i) = recursiveResizeAs(null, srcTable(i))
        }
        i += 1
      }
      while (i <= resultTable.length()) {
        resultTable.remove(i)
        i += 1
      }
    } else if (src.isInstanceOf[Tensor[T]]) {
      result = if (target.isInstanceOf[Tensor[T]]) {
        target
      } else {
        Tensor[T]()
      }
      result.toTensor[T]().resizeAs(src.toTensor())
    }
    result
  }

  /**
   * Apply function 'func' on all tensor in the table.
   * @param x
   * @param func
   */
  def recursiveTensorApply1[T](x: Activities, func: Tensor[T] => Tensor[T])(
    implicit ev: TensorNumeric[T]): Unit = {
    require(x.isInstanceOf[Activities],
      s"expecting tensors or tables thereof. Got ${x} instead"
    )
    if (x.isInstanceOf[Table]) {
      var i = 1
      while (i <= x.toTable().length()) {
        recursiveTensorApply1(x.toTable()(i), func)
        i += 1
      }
    } else {
      func(x.toTensor[T]())
    }
  }

  /**
   * Apply function 'func' on each tensor in table x and table y recursively.
   *
   * Table x should have the same size with table y.
   *
   * @param x
   * @param y
   * @param func
   * @return
   */
  def recursiveTensorApply2[T](x: Activities, y: Activities,
    func: (Tensor[T], Tensor[T]) => Tensor[T])(implicit ev: TensorNumeric[T]): Activities = {
    if (y.isInstanceOf[Tensor[T]] && x.isInstanceOf[Tensor[T]]) {
      require(x.toTensor[T]().nElement() == y.toTensor[T]().nElement(),
        "x, y should have the same size")
      func(x.toTensor[T](), y.toTensor[T]())
    } else {
      require(x.isInstanceOf[Table] && y.isInstanceOf[Table], "x, y should have the same size")
      require(x.toTable().length() == y.toTable().length(), "x, y should have the same size")
      var i = 1
      while (i <= x.toTable().length()) {
        recursiveTensorApply2[T](x, y, func)
        i += 1
      }
    }
    x
  }

  /**
   * Apply a add operation on table x and table y one by one.
   * y := y + alpha * x
   *
   * Table x should have the same size with y.
   *
   * @param y
   * @param alpha
   * @param x
   * @tparam T: Float or Double
   * @return y
   */
  def recursiveAdd[T](y: Activities, alpha: Double = 1.0, x: Activities )(
    implicit ev: TensorNumeric[T]): Activities = {
    recursiveTensorApply2[T](y, x, (t1, t2) => t1.add(ev.fromType[Double](alpha), t2))
    y
  }

  /**
   * copy table x's tensor to table y.
   *
   * Table x should have the same size with y.
   *
   * @param y
   * @param x
   * @tparam T: Float or Double
   * @return y
   */
  def recursiveCopy[T](y: Activities, x: Activities )(
    implicit ev: TensorNumeric[T]): Activities = {
    recursiveTensorApply2[T](y, x, (t1, t2) => t1.copy(t2))
    y
  }

  /**
   * Fill the value to each Tensor in the table recursively
   * @param x
   * @param value
   */
  def recursiveFill[T](x: Activities, value : Double)(
    implicit ev: TensorNumeric[T]): Unit = {
    recursiveTensorApply1[T](x, t => t.fill(ev.fromType[Double](value)))
  }

}
