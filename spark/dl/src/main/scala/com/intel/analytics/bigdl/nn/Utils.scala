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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

object Utils {
  /**
   * This method recursively keep the shape of the source table `t2` and
   * set the elements of each tensor to zero, saving the result on the destination
   * table `t1`
   * Notice that `t1` and `t2` can only contain tables or tensors
   * @param t1 is the destination table
   * @param t2 is the source table
   * @return
   */
  def zeroTableCopy[T : ClassTag](t1: Table, t2: Table)(
    implicit ev: TensorNumeric[T]): Table = {
    t2.foreach { case ((k: Any, v: Any)) =>
      if (v.isInstanceOf[Table]) {
        t1.update(k, zeroTableCopy(if (t1.contains(k)) t1(k) else T(), t2(k)))
      } else {
        require(v.isInstanceOf[Tensor[_]], "Input can only consist of Tensor or Table")
        val tensorV = v.asInstanceOf[Tensor[T]]
        if (!t1.contains(k)) {
          t1.update(k, tensorV.clone().zero())
        } else {
          t1[Tensor[T]](k).resizeAs(tensorV)
          t1[Tensor[T]](k).zero()
        }
      }
    }
    t1.foreach { case ((k: Any, v: Any)) =>
      if (!t2.contains(k)) {
        t1.update(k, null)
      }
    }

    t1
  }

  /**
   * Resize table target as table src.
 *
   * @param target
   * @param src
   */
  def recursiveResizeAs[T : ClassTag](target : Activity, src: Activity)(
    implicit ev: TensorNumeric[T]): Activity = {
    var result: Activity = null
    if (src.isInstanceOf[Table]) {
      val srcTable = src.toTable
      result = if (null == target) {
        T()
      } else if (target.isInstanceOf[Tensor[_]]) {
        T(target)
      } else {
        target
      }

      val resultTable = result.toTable
      var i = 1
      while (i <= src.toTable.length()) {
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
    } else if (src.isInstanceOf[Tensor[_]]) {
      result = if (target.isInstanceOf[Tensor[_]]) {
        target
      } else {
        Tensor[T]()
      }
      result.toTensor[T].resizeAs(src.toTensor)
    }
    result
  }

  /**
   * Apply function 'func' on all tensor in the table.
 *
   * @param x
   * @param func
   */
  def recursiveTensorApply1[T](x: Activity, func: Tensor[T] => Tensor[T])(
    implicit ev: TensorNumeric[T]): Unit = {
    require(x.isInstanceOf[Activity],
      s"expecting tensors or tables thereof. Got ${x} instead"
    )
    if (x.isInstanceOf[Table]) {
      var i = 1
      while (i <= x.toTable.length()) {
        recursiveTensorApply1(x.toTable(i), func)
        i += 1
      }
    } else {
      func(x.toTensor[T])
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
  def recursiveTensorApply2[T](x: Activity, y: Activity,
    func: (Tensor[T], Tensor[T]) => Tensor[T])(implicit ev: TensorNumeric[T]): Activity = {
    if (y.isInstanceOf[Tensor[_]] && x.isInstanceOf[Tensor[_]]) {
      require(x.toTensor[T].nElement() == y.toTensor[T].nElement(),
        "x, y should have the same size")
      func(x.toTensor[T], y.toTensor[T])
    } else {
      require(x.isInstanceOf[Table] && y.isInstanceOf[Table], "x, y should have the same size")
      require(x.toTable.length() == y.toTable.length(), "x, y should have the same size")
      var i = 1
      while (i <= x.toTable.length()) {
        recursiveTensorApply2[T](x.toTable(i), y.toTable(i), func)
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
  def recursiveAdd[T](y: Activity, alpha: Double = 1.0, x: Activity )(
    implicit ev: TensorNumeric[T]): Activity = {
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
  def recursiveCopy[T](y: Activity, x: Activity )(
    implicit ev: TensorNumeric[T]): Activity = {
    recursiveTensorApply2[T](y, x, (t1, t2) => t1.copy(t2))
    y
  }

  /**
   * Fill the value to each Tensor in the table recursively
 *
   * @param x
   * @param value
   */
  def recursiveFill[T](x: Activity, value : Double)(
    implicit ev: TensorNumeric[T]): Unit = {
    recursiveTensorApply1[T](x, t => t.fill(ev.fromType[Double](value)))
  }

  /**
   * get all modules and map by name
   *
   * @param model
   * @tparam T
   * @return
   */
  def getNamedModules[T](model: Module[T]): Map[String, Module[T]] = {
    var namedModules: Map[String, Module[T]] = Map()
    def getModules(module: Module[T]): Unit = {
      module match {
        case m: Container[_, _, T] =>
          namedModules += (module.getName() -> module)
          for (m <- module.asInstanceOf[Container[_, _, T]].modules) getModules(m)
        case _ => namedModules += (module.getName() -> module)
      }
    }
    getModules(model)
    namedModules
  }

  /**
   * copy src's parameters and running status to dst
   * @param src source model
   * @param dst destination model
   */
  def copyModule[T](src: Module[T], dst: Module[T]): Module[T] = {
    // copy parameters
    val srcParameters = src.getParameters()._1
    val dstParameters = dst.getParameters()._1
    require(srcParameters.size(1) == dstParameters.size(1),
      s"$src and $dst is not the same type.")
    dstParameters.copy(srcParameters)
    // copy running status
    dst.copyStatus(src)
    dst
  }

  /**
   * get the inner loop size and outer loop size given a pivot dim
   * @param pivotDim is the dim whose value larger than 1
   * @return inner loop size and outer loop size
   */
  private[nn] def getInnerOuterNum[T](pivotDim: Int, data: Tensor[T]): (Int, Int) = {
    var k = 1
    var outerNum = 1
    while (k < pivotDim) {
      outerNum *= data.size(k)
      k += 1
    }
    var innerNum = 1
    k = pivotDim + 1
    while (k <= data.dim()) {
      innerNum *= data.size(k)
      k += 1
    }
    (innerNum, outerNum)
  }

  /**
   * if there is only one dim of size > 1, return this dim(count from 1)
   * else return -1
   * e.g. (1, 2, 1, 1) returns 1, (1, 2, 3, 1) returns -1, and (1, 1, 1, 1) returns -1
   * @param size size of tensor
   * @return (the only dim whose value > 1) else (-1)
   */
  private[nn] def getOnlyDimGtOne(size: Array[Int]): Int = {
    var i = 0
    var count = 0
    var pivot = 0
    while (i < size.length) {
      if (size(i) > 1) {
        count += 1
        pivot = i + 1
      }
      i += 1
    }
    if (count == 1) pivot else -1
  }
}
