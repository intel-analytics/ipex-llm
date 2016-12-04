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

import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import com.intel.analytics.bigdl.mkl.{MklDnnDouble, MklDnnFloat}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class MklTensor[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends DenseTensor[T] {

  private[this] var _storageMkl: Storage[T] = new ArrayStorage[T](new Array[T](0))
  private[this] var _storage: Storage[T] = new ArrayStorage[T](new Array[T](0))

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

  // unsupport methods

  def returnInt(): Int = {
    require(false, "MklTensor unsupported method")
    0
  }

  def returnTensor(): Tensor[T] = {
    require(false, "MklTensor unsupported method")
    Tensor[T]()
  }

  def returnIntArray(): Array[Int] = {
    require(false, "MklTensor unsupported method")
    Array[Int]()
  }

  def returnT(): T = {
    require(false, "MklTensor unsupported method")
    this._storage(0)
  }

  def returnUnit() : Unit = {
    require(false, "MklTensor unsupported method")
  }

  def returnBoolean() : Boolean = {
    require(false, "MklTensor unsupported method")
    false
  }

  def returnThis() : this.type = {
    require(false, "MklTensor unsupported method")
    this
  }

  def returnString(): String = {
    require(false, "MklTensor unsupported method")
    ""
  }

  def returnChar(): Char = {
    require(false, "MklTensor unsupported method")
    'a'
  }

  def returnTuple(): (Tensor[T], Tensor[T]) = {
    require(false, "MklTensor unsupported method")
    (Tensor[T](), Tensor[T]())
  }

  override def storage(): Storage[T] = _storage

  override def storageOffset(): Int = returnInt()

  override def dim(): Int = returnInt()

  override def squeeze(): Tensor[T] = returnTensor()

  override def squeeze(dim: Int): Tensor[T] = returnTensor()

  override def size(): Array[Int] = returnIntArray()

  override def size(dim: Int): Int = returnInt()

  override def stride(): Array[Int] = returnIntArray()

  override def stride(dim: Int): Int = returnInt()

  override def resizeAs(src: Tensor[_]): Tensor[T] = returnTensor()

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = returnTensor()

  override def resize(size1: Int): Tensor[T] = returnTensor()

  override def resize(size1: Int, size2: Int): Tensor[T] = returnTensor()

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = returnTensor()

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = returnTensor()

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = returnTensor()

  override def view(sizes: Array[Int]): Tensor[T] = returnTensor()

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = returnTensor()

  override def fill(v: T): Tensor[T] = returnTensor()

  override def zero(): Tensor[T] = returnTensor()

  override def randn(): Tensor[T] = returnTensor()

  override def bernoulli(p: Double): Tensor[T] = returnTensor()

  override def rand(): Tensor[T] = returnTensor()

  override def set(other: Tensor[T]): Tensor[T] = returnTensor()

  override def set(storage: Storage[T], storageOffset: Int = 1, sizes: Array[Int] = null,
                   strides: Array[Int] = null): Tensor[T] = returnTensor()

  override def set(): Tensor[T] = returnTensor()

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = returnTensor()

  override def t(): Tensor[T] = returnTensor()

  override def select(dim: Int, index: Int): Tensor[T] = returnTensor()

  override def clone(): Tensor[T] = returnTensor()

  override def copy(other: Tensor[T]): Tensor[T] = returnTensor()

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = returnTensor()

  override def apply1(func: T => T): Tensor[T] = returnTensor()

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = returnTensor()

  override def apply(index: Int): Tensor[T] = returnTensor()

  override def apply(table: Table): Tensor[T] = returnTensor()

  override def update(table: Table, value: T): Unit = returnUnit()

  override def update(table: Table, src: Tensor[T]): Unit = returnUnit()

  override def update(index: Int, src: Tensor[T]): Unit = returnUnit()

  override def apply(indexes: Array[Int]): T = returnT()

  override def valueAt(d1: Int): T = returnT()

  override def valueAt(d1: Int, d2: Int): T = returnT()

  override def valueAt(d1: Int, d2: Int, d3: Int): T = returnT()

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = returnT()

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = returnT()

  private def getOffset(z: Int, dim: Int): Int = returnInt()

  override def update(index: Int, value: T): Unit = returnT()

  override def update(indexes: Array[Int], value: T): Unit = returnUnit()

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): this.type = returnThis()

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): this.type = returnThis()

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): this.type = returnThis()

  override def setValue(d1: Int, d2: Int, value: T): this.type = returnThis()

  override def setValue(d1: Int, value: T): this.type = returnThis()

  override def update(func: T => Boolean, value: T): Unit = returnUnit()

  override def isContiguous(): Boolean = returnBoolean()

  override def contiguous(): Tensor[T] = returnTensor()

  override def isSameSizeAs(other: Tensor[_]): Boolean = returnBoolean()

  override def split(size: Int, dim: Int): Array[Tensor[T]] = {
    require(false, "MklTensor unsupported method")
    Array[Tensor[T]]()
  }

  // scalastyle:off methodName
  override def +(s: T): Tensor[T] = returnTensor()

  override def +(t: Tensor[T]): Tensor[T] = returnTensor()

  override def -(s: T): Tensor[T] = returnTensor()

  override def -(t: Tensor[T]): Tensor[T] = returnTensor()

  override def unary_-(): Tensor[T] = returnTensor()

  override def /(s: T): Tensor[T] = returnTensor()

  override def /(t: Tensor[T]): Tensor[T] = returnTensor()

  override def *(s: T): Tensor[T] = returnTensor()

  override def *(t: Tensor[T]): Tensor[T] = returnTensor()

  // scalastyle:on methodName

  override def sum(): T = returnT()

  override def sum(dim: Int): Tensor[T] = returnTensor()

  override def sum(x: Tensor[T], dim: Int): Tensor[T] = returnTensor()

  override def mean(): T = returnT()

  override def mean(dim: Int): Tensor[T] = returnTensor()

  override def max(): T = returnT()

  override def max(dim: Int): (Tensor[T], Tensor[T]) = returnTuple()

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = returnTuple()

  override def min(): T = returnT()

  override def min(dim: Int): (Tensor[T], Tensor[T]) = returnTuple()

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = returnTuple()

  def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = returnTensor()

  override def add(value: T, y: Tensor[T]): Tensor[T] = returnTensor()

  override def add(x: Tensor[T]): Tensor[T] = returnTensor()

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  // Puts the result of x + value * y in current tensor
  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = returnTensor()

  override def add(value: T): Tensor[T] = returnTensor()

  override def sub(value: T, y: Tensor[T]): Tensor[T] = returnTensor()

  override def sub(x: Tensor[T]): Tensor[T] = returnTensor()

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()
  // Puts the result of x - value * y in current tensor
  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = returnTensor()

  override def sub(value: T): Tensor[T] = returnTensor()

  override def dot(y: Tensor[T]): T = returnT()

  override def cmax(value: T): Tensor[T] = returnTensor()

  override def dist(y: Tensor[T], norm: Int): T = returnT()

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = returnTensor()

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = returnTensor()

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = returnTensor()

  override def cmul(y: Tensor[T]): Tensor[T] = returnTensor()

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def cdiv(y: Tensor[T]): Tensor[T] = returnTensor()

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def cmax(y: Tensor[T]): Tensor[T] = returnTensor()

  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def mul(x: Tensor[T], value: T): Tensor[T] = returnTensor()

  override def mul(value: T): Tensor[T] = returnTensor()

  override def div(value: T): Tensor[T] = returnTensor()

  override def conv2(kernel: Tensor[T], vf: Char = 'V'): Tensor[T] = returnTensor()

  override def xcorr2(kernel: Tensor[T], vf: Char = 'V'): Tensor[T] = returnTensor()

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = returnTensor()

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = returnTensor()

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = returnTensor()

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = returnTensor()

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = returnTensor()

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = returnTensor()

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = returnTensor()

  override def addr(v2: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = returnTensor()

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = returnTensor()

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] = returnTensor()

  override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T],
                     vec2: Tensor[T]): Tensor[T] = returnTensor()

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = returnTensor()

  override def uniform(args: T*): T = returnT()

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = returnTensor()

  override def expandAs(template: Tensor[T]): Tensor[T] = returnTensor()

  override def expand(sizes: Array[Int]): Tensor[T] = returnTensor()

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = returnTensor()

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = returnTensor()

  override def baddbmm(beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T],
                       batch2: Tensor[T]): Tensor[T] = returnTensor()

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = returnTensor()

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = returnTensor()

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = returnTensor()

  override def abs(): Tensor[T] = returnTensor()

  override def toBreezeVector(): BrzDenseVector[T] = throw new UnsupportedOperationException

  override def getType(): TensorDataType = throw new IllegalArgumentException

  override def toMLlibMatrix(): Matrix = throw new UnsupportedOperationException

  override def toBreezeMatrix(): BrzDenseMatrix[T] = throw new UnsupportedOperationException

  override def toMLlibVector(): Vector = throw new UnsupportedOperationException

  override def equals(obj: Any): Boolean = returnBoolean()

  override def hashCode(): Int = returnInt()

  override def toString(): String = returnString()

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = returnBoolean()

  override def reshape(sizes: Array[Int]): Tensor[T] = returnTensor()

  override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T],
                    indices: Tensor[T]): (Tensor[T], Tensor[T]) = returnTuple()

  override def pow(x: Tensor[T], n: T): Tensor[T] = returnTensor()

  override def pow(n: T): Tensor[T] = returnTensor()

  override def log(x: Tensor[T]): Tensor[T] = returnTensor()

  override def log(): Tensor[T] = returnTensor()

  override def exp(x: Tensor[T]): Tensor[T] = returnTensor()

  override def exp(): Tensor[T] = returnTensor()

  override def sqrt(x: Tensor[T]): Tensor[T] = returnTensor()

  override def sqrt(): Tensor[T] = returnTensor()

  override def log1p(x: Tensor[T]): Tensor[T] = returnTensor()

  override def log1p(): Tensor[T] = returnTensor()

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] = returnTensor()

  override def abs(x: Tensor[T]): Tensor[T] = returnTensor()

  override def save(path: String, overWrite: Boolean): this.type = returnThis()

  override def maskedFill(mask: Tensor[T], value: T): Tensor[T] = returnTensor()

  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def maskedSelect(mask: Tensor[T], res: Tensor[T]): Tensor[T] = returnTensor()

  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def eq(x: Tensor[T], value: T): Tensor[T] = returnTensor()

  override def norm(value: Int): T = returnT()

  override def sign(): Tensor[T] = returnTensor()

  override def addSingletonDimension(t: Tensor[T], dim: Int = 1): Tensor[T] = returnTensor()

  override def ge(x: Tensor[T], value: Double): Tensor[T] = returnTensor()

  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()

  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = returnTensor()
}

object MklTensor {

  def convert[@specialized(Float, Double) T: ClassTag]
  (usrStorage: Storage[T],
   storageOffset: Int,
   mklStorage: Storage[T],
   primitive: Long,
   toMkl: Boolean)(implicit ev: TensorNumeric[T]): Unit = {
    import scala.language.implicitConversions
    implicit def bool2int(b: Boolean) = if (b) 1 else 0

    require(primitive == 0, "convert primitive doesn't exist")

    ev.getType() match {
      case "Double" =>
        MklDnnDouble.conversionExecute(
          usrStorage.array().asInstanceOf[Array[Float]],
          mklStorage.array().asInstanceOf[Array[Float]],
          primitive,
          false)
      case "Float" =>
        if (toMkl) {
          MklDnnFloat.conversionExecuteToMkl(
            usrStorage.array().asInstanceOf[Array[Float]],
            storageOffset,
            mklStorage.array().asInstanceOf[Array[Float]],
            primitive)
        } else {
          MklDnnFloat.conversionExecuteToUsr(
            usrStorage.array().asInstanceOf[Array[Float]],
            storageOffset,
            mklStorage.asInstanceOf[Array[Float]],
            primitive)
        }
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }
  }
}
