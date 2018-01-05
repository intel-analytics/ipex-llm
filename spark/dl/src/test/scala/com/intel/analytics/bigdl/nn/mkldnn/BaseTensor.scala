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

package com.intel.analytics.bigdl.nn.mkldnn

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor, TensorDataType, TensorType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Matrix
import org.apache.zookeeper.KeeperException.UnimplementedException

import scala.reflect.ClassTag

class BaseTensor[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Tensor[T] {
  val message = "Not supported in MklDnnTensor"

  override def isEmpty: Boolean = throw new UnsupportedOperationException(message)

  override def isScalar: Boolean =
    throw new UnsupportedOperationException(message)

  override def nDimension(): Int =
    throw new UnsupportedOperationException(message)

  override def dim(): Int =
    throw new UnsupportedOperationException(message)

  override def size(): Array[Int] =
    throw new UnsupportedOperationException(message)

  override def size(dim: Int): Int =
    throw new UnsupportedOperationException(message)

  override def stride(): Array[Int] =
    throw new UnsupportedOperationException(message)

  override def stride(dim: Int): Int =
    throw new UnsupportedOperationException(message)

  override def fill(v: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def forceFill(v: Any): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def zero(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def randn(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def randn(mean: Double, stdv: Double): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def rand(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def rand(lowerBound: Double, upperBound: Double): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def bernoulli(p: Double): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def transpose(dim1: Int, dim2: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def t(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def apply(index: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def apply(indexes: Array[Int]): T =
    throw new UnsupportedOperationException(message)

  override def value(): T =
    throw new UnsupportedOperationException(message)

  override def valueAt(d1: Int): T =
    throw new UnsupportedOperationException(message)

  override def valueAt(d1: Int, d2: Int): T =
    throw new UnsupportedOperationException(message)

  override def valueAt(d1: Int, d2: Int, d3: Int): T =
    throw new UnsupportedOperationException(message)

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T =
    throw new UnsupportedOperationException(message)

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T =
    throw new UnsupportedOperationException(message)

  override def apply(t: Table): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def update(index: Int, value: T): Unit =
    throw new UnsupportedOperationException(message)

  override def update(index: Int, src: Tensor[T]): Unit =
    throw new UnsupportedOperationException(message)

  override def update(indexes: Array[Int], value: T): Unit =
    throw new UnsupportedOperationException(message)

  override def setValue(value: T): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def setValue(d1: Int, value: T): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def setValue(d1: Int, d2: Int, value: T): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def update(t: Table, value: T): Unit =
    throw new UnsupportedOperationException(message)

  override def update(t: Table, src: Tensor[T]): Unit =
    throw new UnsupportedOperationException(message)

  override def update(filter: (T) => Boolean, value: T): Unit =
    throw new UnsupportedOperationException(message)

  override def isContiguous(): Boolean =
    throw new UnsupportedOperationException(message)

  override def contiguous(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def isSameSizeAs(other: Tensor[_]): Boolean =
    throw new UnsupportedOperationException(message)

  override def emptyInstance(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def resizeAs(src: Tensor[_]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cast[D: ClassTag](castTensor: Tensor[D])(implicit ev: TensorNumeric[D]): Tensor[D] =
    throw new UnsupportedOperationException(message)

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def resize(size1: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def resize(size1: Int, size2: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def nElement(): Int =
    throw new UnsupportedOperationException(message)

  override def select(dim: Int, index: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def storage(): Storage[T] =
    throw new UnsupportedOperationException(message)

  override def storageOffset(): Int =
    throw new UnsupportedOperationException(message)

  override def set(other: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def set(storage: Storage[T], storageOffset: Int, sizes: Array[Int], strides: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def set(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def copy(other: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def applyFun[A: ClassTag](t: Tensor[A], func: (A) => T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def apply1(func: (T) => T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def zipWith[A: ClassTag, B: ClassTag](t1: Tensor[A], t2: Tensor[B], func: (A, B) => T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def squeeze(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def squeeze(dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def squeezeNewTensor(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def view(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def repeatTensor(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def expandAs(template: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def expand(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def split(size: Int, dim: Int): Array[Tensor[T]] =
    throw new UnsupportedOperationException(message)

  override def split(dim: Int): Array[Tensor[T]] =
    throw new UnsupportedOperationException(message)

  override def toBreezeVector(): DenseVector[T] =
    throw new UnsupportedOperationException(message)

  override def toMLlibVector(): linalg.Vector =
    throw new UnsupportedOperationException(message)

  override def toBreezeMatrix(): DenseMatrix[T] =
    throw new UnsupportedOperationException(message)

  override def toMLlibMatrix(): Matrix =
    throw new UnsupportedOperationException(message)

  override def getType(): TensorDataType =
    throw new UnsupportedOperationException(message)

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean =
    throw new UnsupportedOperationException(message)

  override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def reshape(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def save(path: String, overWrite: Boolean): BaseTensor.this.type =
    throw new UnsupportedOperationException(message)

  override def getTensorNumeric(): TensorNumeric[T] =
    throw new UnsupportedOperationException(message)

  override def getTensorType: TensorType =
    throw new UnsupportedOperationException(message)

  override def toArray(): Array[T] =
    throw new UnsupportedOperationException(message)

  override def +(s: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def +(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def -(s: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def -(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def unary_-(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def /(s: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def /(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def *(s: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def *(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sum(): T =
    throw new UnsupportedOperationException(message)

  override def prod(): T =
    throw new UnsupportedOperationException(message)

  override def prod(x: Tensor[T], dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sum(dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sum(x: Tensor[T], dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def mean(): T =
    throw new UnsupportedOperationException(message)

  override def mean(dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def max(): T =
    throw new UnsupportedOperationException(message)

  override def max(dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(message)

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(message)

  override def min(): T =
    throw new UnsupportedOperationException(message)

  override def min(dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(message)

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(message)

  override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sqrt(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def tanh(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def abs(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def add(value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def add(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def add(value: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def dot(y: Tensor[T]): T =
    throw new UnsupportedOperationException(message)

  override def cmax(value: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def dist(y: Tensor[T], norm: Int): T =
    throw new UnsupportedOperationException(message)

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sub(value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sub(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sub(value: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cmul(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cdiv(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def mul(value: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def div(value: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def div(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def mul(x: Tensor[T], value: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def uniform(args: T*): T =
    throw new UnsupportedOperationException(message)

  override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def baddbmm(beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def pow(y: Tensor[T], n: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def pow(n: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def square(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def floor(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def floor(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def ceil(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def inv(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def erf(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def erfc(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def logGamma(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def digamma(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T], indices: Tensor[T], sortedResult: Boolean): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(message)

  override def log(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def exp(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sqrt(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def tanh(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def log1p(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def log(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def exp(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def log1p(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def abs(x: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def eq(x: Tensor[T], y: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def maskedFill(mask: Tensor[T], e: T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def norm(value: Int): T =
    throw new UnsupportedOperationException(message)

  override def sign(): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def ge(x: Tensor[T], value: Double): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cmax(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cmin(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def cmin(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def negative(x: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def reduce(dim: Int, result: Tensor[T], reducer: (T, T) => T): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def sumSquare(): T =
    throw new UnsupportedOperationException(message)

  override def clamp(min: Float, max: Float): Tensor[T] =
    throw new UnsupportedOperationException(message)

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] =
    throw new UnsupportedOperationException(message)
}

