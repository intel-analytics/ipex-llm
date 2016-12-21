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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable
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
    for ((k, v) <- t2.getState()) {
      if (v.isInstanceOf[Table]) {
        t1.update(k, zeroTableCopy(if (t1.contains(k)) t1(k) else T(), t2(k)))
      } else {
        require(v.isInstanceOf[Tensor[T]], "Input can only consist of Tensor or Table")
        val tensorV = v.asInstanceOf[Tensor[T]]
        if (!t1.contains(k)) {
          t1.update(k, tensorV.clone().zero())
        } else {
          t1[Tensor[T]](k).resizeAs(tensorV)
          t1[Tensor[T]](k).zero()
        }
      }
    }
    for ((k, v) <- t1.getState()) {
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
      } else if (target.isInstanceOf[Tensor[T]]) {
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
    } else if (src.isInstanceOf[Tensor[T]]) {
      result = if (target.isInstanceOf[Tensor[T]]) {
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
    if (y.isInstanceOf[Tensor[T]] && x.isInstanceOf[Tensor[T]]) {
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

  def shareGradInput[@specialized(Float, Double) T: ClassTag](model: Module[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    def sharingKey(m: Module[T]) = m.getClass.getName

    val cache = mutable.Map[Any, Storage[T]]()
    val packageName: String = model.getName().stripSuffix("Sequential")
    cache.put("fInput", Storage(Array(ev.fromType[Int](1))))
    cache.put("fGradInput", Storage(Array(ev.fromType[Int](1))))

    var index = 0
    def matchModels(model: Module[T]): Unit = {
      model match {
        case container: Container[Activity, Activity, T] =>
          container.modules.foreach( m => {
            if (m.gradInput.isInstanceOf[Tensor[T]] &&
              !m.getClass.getName.equals(packageName + "ConcatTable")) {
              val key = sharingKey(m)
              if (!cache.contains(key)) {
                cache.put(key, Storage(Array(ev.fromType[Int](1))))
              }
              m.gradInput = Tensor(cache.get(key).get, 1, Array(0))
            }
            matchModels(m)
          })
        case concatTable if (concatTable.isInstanceOf[ConcatTable[T]]) =>
          if (!cache.contains(index % 2)) {
            cache.put(index % 2, Storage(Array(ev.fromType[Int](1))))
          }
          concatTable.gradInput = Tensor[T](cache.get(index % 2).get, 1, Array(0))
          index = index + 1
        case spatialConvolution if (spatialConvolution.isInstanceOf[SpatialConvolution[T]]) =>
          val curModel = spatialConvolution.asInstanceOf[SpatialConvolution[T]]
          curModel.setSharedVar
          curModel.fInput = Tensor[T](cache.get("fInput").get)
          curModel.fGradInput = Tensor[T](cache.get("fGradInput").get)
        case _ => Unit
      }
    }
    matchModels(model)
  }

  def findModules[@specialized(Float, Double) T: ClassTag](model: Module[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    model match {
      case container: Container[Activity, Activity, T]
      => container.modules.foreach(m => findModules(m))
      case spatialConvolution if (spatialConvolution.isInstanceOf[SpatialConvolution[T]]) =>
        val curModel = spatialConvolution.asInstanceOf[SpatialConvolution[T]]
        val n: Float = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
        curModel.weight.apply1(_ => ev.fromType[Float](RNG.normal(0, Math.sqrt(2.0f / n)).toFloat))
        curModel.bias.apply1(_ => ev.fromType[Float](0))
      case spatialBatchNormalization
        if (spatialBatchNormalization.isInstanceOf[SpatialBatchNormalization[T]]) =>
        val curModel = spatialBatchNormalization.asInstanceOf[SpatialBatchNormalization[T]]
        curModel.weight.apply1(_ => ev.fromType[Float](1.0f))
        curModel.bias.apply1(_ => ev.fromType[Float](0.0f))
      case linear if (linear.isInstanceOf[Linear[T]]) =>
        linear.asInstanceOf[Linear[T]].bias.apply1(_ => ev.fromType[Float](0))
      case _ => Unit
    }
  }

}
