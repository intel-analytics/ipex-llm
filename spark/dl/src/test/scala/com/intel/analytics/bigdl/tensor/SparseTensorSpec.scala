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
package com.intel.analytics.bigdl.tensor

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SparseTensorSpec  extends FlatSpec with Matchers {
  "dim, shape, nElement" should "return right result" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 4).range(1, 12, 1))
    sTensor1.dim() should be (2)
    sTensor1.nElement() should be (12)
    sTensor1.size() should be (Array(3, 4))

    val sTensor2 = Tensor.sparse(Array(Array(1, 2), Array(3, 5)), Array(1f, 2f), Array(3, 5))
    sTensor2.dim() should be (2)
    sTensor2.nElement() should be (2)
    sTensor2.size() should be (Array(3, 5))
  }

  "storageOffset" should "return right result" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 4).range(1, 12, 1))
    sTensor1.storageOffset() should be (1)
  }

  "narrow" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    val sTensor2 = sTensor.narrow(1, 2, 4)
    val sTensor3 = sTensor2.narrow(1, 2, 3)
    sTensor3.storageOffset() should be (11)
    sTensor3.asInstanceOf[SparseTensor[Float]]._indicesOffset should be (Array(2, 0))
  }

  "resize" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    sTensor.resize(Array(10, 10), 50)
    sTensor.size() should be (Array(10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (50)
  }

  "resize on empty tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    sTensor.set()
    sTensor.resize(Array(10, 10), 50)
    sTensor.size() should be (Array(10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (50)
  }

  "resize on narrowed tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1)).narrow(1, 2, 4)
    sTensor.resize(Array(10, 10), 50)
    sTensor.size() should be (Array(10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (55)
    sTensor.storageOffset() should be (6)
  }

  "resize 2D tensor to 3D tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1)).narrow(1, 2, 4)
    sTensor.resize(Array(10, 10, 10), 50)
    sTensor.size() should be (Array(10, 10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (55)
    sTensor.storageOffset() should be (6)
  }

  "resize 2D tensor to 1D tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1)).narrow(1, 2, 4)
    sTensor.resize(Array(10), 5)
    sTensor.size() should be (Array(10))
    sTensor.nElement() should be (5)
    sTensor.storage().array.length should be (30)
    sTensor.storageOffset() should be (6)
  }

  "resize 2D tensor to 1D tensor" should "return right result2" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    sTensor.resize(Array(30), 30)
    sTensor.size() should be (Array(30))
    sTensor.nElement() should be (30)
    sTensor.storage().array.length should be (30)
    sTensor.storageOffset() should be (1)
  }

  "resize tensor to higher dim when nElement < sum(size)" should "return right result" in {
    val indices = Array(Array(0, 4, 5, 7, 9))
    val values = Array.fill(5)(Random.nextFloat())
    val sTensor = Tensor.sparse(indices, values, Array(10))
    sTensor.resize(Array(1, 10), 5)
    Tensor.dense(sTensor).squeeze().toArray().sum should be (values.sum)
  }

  "resize narrowed tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(30).range(1, 30, 1)).narrow(1, 6, 18)
    sTensor.resize(Array(6, 3), 18)
    sTensor.size() should be (Array(6, 3))
    sTensor.nElement() should be (18)
    sTensor.storage().array.length should be (30)
    sTensor.storageOffset() should be (6)
  }

  "Tensor.dense narrowed tensor" should "return right result" in {
    val values = Array.fill(30)(Random.nextFloat())
    val sTensor = Tensor.sparse(Tensor(values, Array(6, 5)))
    val narrowed = sTensor.narrow(1, 2, 4)
    val narrowedSum = values.slice(5, 25).sum
    Tensor.dense(narrowed).resize(20).toArray().sum shouldEqual narrowedSum
  }

  "SparseTensor dot DenseTense" should "return right result" in {
    val rng = RandomGenerator.RNG
    rng.setSeed(10)
    val values = Array.fill(30)(rng.normal(1.0, 1.0).toFloat)
    val sTensor = Tensor.sparse(Tensor(values, Array(6, 5)))

    val dTensor = Tensor(Array(6, 5)).rand()

    val sparseResult = sTensor.dot(dTensor)
    val denseResult = dTensor.dot(Tensor.dense(sTensor))

    sparseResult should be (denseResult +- 1e-6f)
  }

  "Diagonal SparseTensor dot DenseTense" should "return right result" in {
    val sTensor = Tensor.sparse(
      indices = Array(Array(0, 1, 2, 3), Array(0, 1, 2, 3)),
      values = Array[Float](2f, 4f, 6f, 8f), shape = Array(4, 4))

    val dTensor = Tensor(Array(4, 4)).fill(1.0f)

    val sparseResult = sTensor.dot(dTensor)

    sparseResult should be (20f +- 1e-6f)
  }

  "SparseTensor.applyFun" should "work correctly" in {
    val func = (a: Float) => a.round * 2
    val srcTensor = Tensor.sparse(Tensor(3, 4).randn())
    val dstTensor = Tensor.sparse[Int](Array(3, 4), 12)
    dstTensor.applyFun(srcTensor, func)
    dstTensor.storage().array() shouldEqual
      srcTensor.storage().array().map(func)
  }

  "Tensor.cast" should "work on SparseTensor" in {
    val sTensor = Tensor.sparse(Tensor[Int](6, 5).rand())
    val sTensor2 = Tensor.sparse[Int](Tensor[Int](6, 5).rand())
    sTensor.cast(sTensor2)
    sTensor.storage().array() shouldEqual sTensor2.storage().array()

    val sTensor1 = sTensor.cast(sTensor.asInstanceOf[Tensor[Long]])
    sTensor1.storage() shouldEqual sTensor.storage()
  }

}
