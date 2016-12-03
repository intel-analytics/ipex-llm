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

package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.mkl.{MklDnnDouble, MklDnnFloat}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class MklTensor[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends DenseTensor[T] {

  private[this] var _storageMkl: Storage[T] = new ArrayStorage[T](new Array[T](0))

  override def isMklTensor(): Boolean = true

  private[this] var _layoutUsr: Long = 0L // usr layout ptr
  private[this] var _layoutMkl: Long = 0L // mkl layout ptr
  private[this] var _convertToUsr: Long = 0L // convert mkl layout mem to scala layout mem
  private[this] var _convertToMkl: Long = 0L // convert scala layout mem to mkl layout mem

  def createUsrLayout(dimension: Int, size: Array[Long], strides: Array[Long]): Unit = {
    if (this.size().length > 0) {
      ev.getType() match {
        case "Double" => MklDnnDouble.layoutCreate()
        case "Float" =>
          if (layoutUsr == 0) {
            layoutUsr_=(MklDnnFloat.layoutCreate(dimension, size, strides))
          }
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }
    } else {
      0L
    }
  }

  def createMklLayout(primitive: Long, resType: Int): Unit = {
    if (primitive != 0) {
      ev.getType() match {
        case "Double" =>
          val ret = MklDnnDouble.layoutCreateFromPrimitive()
          storageMkl.resize(MklDnnDouble.layoutGetMemorySize())
          ret
        case "Float" =>
          if (layoutMkl == 0) {
            layoutMkl_=(MklDnnFloat.layoutCreateFromPrimitive(primitive, resType))
            storageMkl.resize(MklDnnFloat.layoutGetMemorySize(layoutMkl) / 4)
            println(MklDnnFloat.layoutGetMemorySize(layoutMkl) / 4)
          }
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }
    } else {
      0L
    }
  }

  def convert(toMkl: Boolean): Unit = {
    val isSame = ev.getType() match {
      case "Double" => MklDnnDouble.layoutCompare(layoutUsr, layoutMkl)
      case "Float" => MklDnnFloat.layoutCompare(layoutUsr, layoutMkl)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (layoutUsr != 0 && layoutMkl != 0 && isSame != 1) {
      import scala.language.implicitConversions
      implicit def bool2int(b: Boolean) = if (b) 1 else 0

      ev.getType() match {
        case "Double" =>
          convertToUsr_=(MklDnnDouble.conversionCreate())
          convertToMkl_=(MklDnnDouble.conversionCreate())
          MklDnnDouble.conversionExecute(this.storage().array().asInstanceOf[Array[Float]],
                                         storageMkl.array().asInstanceOf[Array[Float]],
                                         convertToMkl,
                                         toMkl)
        case "Float" =>
//          if (convertToUsr != 0) {
//            MklDnnFloat.deletePrimitive(convertToUsr)
//            convertToUsr_=(0)
//          }
//          if (convertToMkl != 0) {
//            MklDnnFloat.deletePrimitive(convertToMkl)
//            convertToMkl_=(0)
//          }
          if (convertToUsr == 0) {
            convertToUsr_=(MklDnnFloat.conversionCreate(layoutMkl, layoutUsr))
          }
          if (convertToMkl == 0) {
            convertToMkl_=(MklDnnFloat.conversionCreate(layoutUsr, layoutMkl))
          }

          require(convertToMkl != 0, "create mkl dnn conversion (usr -> mkl) failed.")
          require(convertToUsr != 0, "create mkl dnn conversion (mkl -> usr) failed.")

          if (toMkl) {
            MklDnnFloat.conversionExecuteToMkl(this.storage().array().asInstanceOf[Array[Float]],
                                               this.storageOffset() - 1,
                                               storageMkl.array().asInstanceOf[Array[Float]],
                                               convertToMkl)
            println("convert usr -> mkl")
          } else {
            MklDnnFloat.conversionExecuteToUsr(this.storage().array().asInstanceOf[Array[Float]],
                                               this.storageOffset() - 1,
                                               storageMkl.array().asInstanceOf[Array[Float]],
                                               convertToUsr)
          }
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }

    }

    if (isSame == 1) {
      if (toMkl) {
        storageMkl.copy(storage())
      } else {
        storage().copy(storageMkl)
      }
    }
  }

  // {{ getter && setter

  def convertToUsr: Long = _convertToUsr

  def convertToUsr_=(value: Long): Unit = {
    _convertToUsr = value
  }

  def convertToMkl: Long = _convertToMkl

  def convertToMkl_=(value: Long): Unit = {
    _convertToMkl = value
  }

  def storageMkl: Storage[T] = _storageMkl

  def storageMkl_=(value: Storage[T]): Unit = {
    _storageMkl = value
  }

  def layoutUsr: Long = _layoutUsr

  def layoutUsr_=(value: Long): Unit = {
    _layoutUsr = value
  }

  def layoutMkl: Long = _layoutMkl

  def layoutMkl_=(value: Long): Unit = {
    _layoutMkl = value
  }

  // }} ---------------------------------------------
}
