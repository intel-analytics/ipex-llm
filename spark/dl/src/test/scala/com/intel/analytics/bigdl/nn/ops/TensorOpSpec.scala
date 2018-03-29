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

package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.Sigmoid
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class TensorOpSpec extends FlatSpec with Matchers {

  private val tt = Tensor[Float](2, 3).rand()
  private val copiedTT = Tensor[Float]().resizeAs(tt).copy(tt)

  private def ttCopy() = Tensor[Float]().resizeAs(copiedTT).copy(copiedTT)

  "Common TensorOps" should "work correctly" in {
    val rnd = Random.nextFloat()
    TensorOp.add[Float](rnd).forward(tt) shouldEqual ttCopy().add(rnd)
    TensorOp.sub[Float](rnd).forward(tt) shouldEqual ttCopy().sub(rnd)
    TensorOp.mul[Float](rnd).forward(tt) shouldEqual ttCopy().mul(rnd)
    TensorOp.div[Float](rnd).forward(tt) shouldEqual ttCopy().div(rnd)
    TensorOp.pow[Float](rnd).forward(tt) shouldEqual ttCopy().pow(rnd)
    TensorOp.ge[Float](rnd).forward(tt) shouldEqual ttCopy().ge(ttCopy(), rnd)
    TensorOp.eq[Float](rnd).forward(tt) shouldEqual ttCopy().eq(ttCopy(), rnd)

    val rndT = Tensor[Float](2, 3).rand()
    TensorOp.add[Float](rndT).forward(tt) shouldEqual ttCopy().add(rndT)
    TensorOp.sub[Float](rndT).forward(tt) shouldEqual ttCopy().sub(rndT)
    TensorOp.mul[Float](rndT).forward(tt) shouldEqual ttCopy().cmul(rndT)
    TensorOp.div[Float](rndT).forward(tt) shouldEqual ttCopy().div(rndT)

    TensorOp.sign[Float]().forward(tt) shouldEqual ttCopy().sign()
    TensorOp.sqrt[Float]().forward(tt) shouldEqual ttCopy().sqrt()
    TensorOp.square[Float]().forward(tt) shouldEqual ttCopy().square()
    TensorOp.t[Float]().forward(tt) shouldEqual ttCopy().t()
    TensorOp.exp[Float]().forward(tt) shouldEqual ttCopy().exp()
    TensorOp.abs[Float]().forward(tt) shouldEqual ttCopy().abs()
    TensorOp.log[Float]().forward(tt) shouldEqual ttCopy().log()
    TensorOp.log1p[Float]().forward(tt) shouldEqual ttCopy().log1p()
    TensorOp.floor[Float]().forward(tt) shouldEqual ttCopy().floor()
    TensorOp.ceil[Float]().forward(tt) shouldEqual ttCopy().ceil()
    TensorOp.inv[Float]().forward(tt) shouldEqual ttCopy().inv()
    TensorOp.negative[Float]().forward(tt) shouldEqual ttCopy().negative(ttCopy())
    TensorOp.tanh[Float]().forward(tt) shouldEqual ttCopy().tanh()
    TensorOp.sigmoid[Float]().forward(tt) shouldEqual Sigmoid[Float]().forward(ttCopy())
  }

  "Chaining user-defined TensorOps" should "work correctly" in {
    val transformer1: (Tensor[Float], TensorNumeric[Float]) => Tensor[Float] = {
      (t: Tensor[Float], _) => t.apply1((t: Float) => t * t)
    }
    val transformer2: (Tensor[Float], TensorNumeric[Float]) => Tensor[Float] = {
      (t: Tensor[Float], _) => t.apply1((t: Float) => t + 1)
    }
    val transformer3: (Tensor[Float], TensorNumeric[Float]) => Tensor[Float] = {
      (t: Tensor[Float], _) => t.apply1((t: Float) => math.sqrt(t).toFloat)
    }
    val op1 = TensorOp[Float](transformer1)
    val op2 = TensorOp[Float](transformer2)
    val op3 = TensorOp[Float](transformer3)

    val op = TensorOp[Float]((t: Tensor[Float], ev: TensorNumeric[Float]) => {
      transformer3(transformer2(transformer1(t, ev), ev), ev)
    })

    (op1 -> op2 -> op3).forward(tt) shouldEqual op.forward(tt)
  }

  "Chaining provided Common TensorOps" should "work correctly" in {
    var op = (TensorOp[Float]() * 2.3f + 1.23f) / 1.11f - 0.66f
    op.forward(tt) shouldEqual ttCopy().mul(2.3f).add(1.23f).div(1.11f).sub(0.66f)

    var cpy = ttCopy()
    op = TensorOp.negative()
    op.forward(tt) shouldEqual cpy.negative(cpy)
    op = op ** 3f
    op.forward(tt) shouldEqual cpy.pow(3f)
    op = op.abs.sqrt.log1p + 1.2f
    op.forward(tt) shouldEqual cpy.abs().sqrt().log1p().add(1.2f)

    cpy = ttCopy()
    op = ((TensorOp.square[Float]() + 1.0f) * 2.5f) >= 3.0
    op.forward(tt) shouldEqual {
      val x = cpy.square().add(1.0f).mul(2.5f)
      x.ge(x, 3.0)
    }

    val op1 = (op -> TensorOp.sigmoid[Float]()).inv.sqrt
    op1.forward(tt) shouldEqual Sigmoid[Float]().forward(cpy).inv().sqrt()
    val op2 = op -> TensorOp.sigmoid[Float]().inv.sqrt
    op2.forward(tt) shouldEqual Sigmoid[Float]().forward(cpy).inv().sqrt()
  }

}

class TensorOpSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val op = (((TensorOp[Float]() + 1.5f) ** 2) -> TensorOp.sigmoid()
      ).setName("TensorOP")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(op, input)
  }
}
