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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * [[TensorOp]] is an [[Operation]] with `Tensor[T]-formatted `input and output,
 * which provides shortcuts to build Operations for `tensor transformation` by closures.
 * <br></br>
 * [[TensorOp]] will make a deep copy of input Tensor before transformation,
 * so transformation will take no side effect. For now, `SparseTensors` are not supported.
 * <br></br>
 * Chained feature is supported in [[TensorOp]].
 * And common tensor actions are provided with a chained style.
 * <br></br>
 * For instance:
 * {{{
 * one case:
 *    val (transformer1, transformer2, transformer3) = ...
 *    val (op1, op2, op3) = (TensorOp[Float](transformer1), .., ..)
 *    val op = op1 -> op2 -> op3
 *      `equals`
 *    val op = TensorOp[Float]((t: Tensor[Float], ev: TensorNumeric[Float]) => {
 *      transformer3(transformer2(transformer1(t, ev), ev), ev)
 *     })
 *
 * another case:
 *    val op = (TensorOp[Float]() * 2.3f + 1.23f) / 1.11f - 0.66f
 *      `equals`
 *    val transformer = (t: Tensor[T], _) => t.mul(2.3f).add(1.23f).div(1.11f).sub(0.66f)
 *    val op = TensorOp[Float](transformer)
 * }}}
 *
 * @param transformer closure of tensor transformation
 * @tparam T Numeric type
 */
class TensorOp[T: ClassTag] private(
  private[bigdl] val transformer: (Tensor[T], TensorNumeric[T]) => Tensor[T]
)(implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], Tensor[T], T] {

  private lazy val buffer: Tensor[T] = Tensor[T]()

  // TODO: support SparseTensor
  final override def updateOutput(input: Tensor[T]): Tensor[T] = {
    buffer.resizeAs(input).copy(input)
    output = transformer(buffer, ev)
    output
  }

  // scalastyle:off
  final def ->(next: TensorOp[T]): TensorOp[T] = {
    val chained = (in: Tensor[T], ev: TensorNumeric[T]) => {
      next.transformer(transformer(in, ev), ev)
    }
    new TensorOp(chained)
  }

  /**
   * append additional TensorOp to do element-wise `f(x) = x + a`
   *
   * @param value T a
   * @return TensorOp[T]
   */
  final def +(value: T): TensorOp[T] = this -> TensorOp.add(value)

  /**
   * append additional TensorOp to do element-wise tensor addition
   *
   * @param tensor Tensor[T]
   * @return TensorOp[T]
   */
  final def +(tensor: Tensor[T]): TensorOp[T] = this -> TensorOp.add(tensor)

  /**
   * build a TensorOp to do element-wise `f(x) = x - a`
   *
   * @param value T a
   * @return TensorOp[T]
   */
  final def -(value: T): TensorOp[T] = this -> TensorOp.sub(value)

  /**
   * build a TensorOp to do element-wise tensor subtraction
   *
   * @param tensor Tensor[T]
   * @return TensorOp[T]
   */
  final def -(tensor: Tensor[T]): TensorOp[T] = this -> TensorOp.sub(tensor)

  /**
   * build a TensorOp to do element-wise `f(x) = a * x`
   *
   * @param value T a
   * @return TensorOp[T]
   */
  final def *(value: T): TensorOp[T] = this -> TensorOp.mul(value)

  /**
   * build a TensorOp to do element-wise multiplication
   *
   * @param tensor Tensor[T]
   * @return TensorOp[T]
   */
  final def *(tensor: Tensor[T]): TensorOp[T] = this -> TensorOp.mul(tensor)

  /**
   * build a TensorOp to do element-wise `f(x) = x / a`
   *
   * @param value T a
   * @return TensorOp[T]
   */
  final def /(value: T): TensorOp[T] = this -> TensorOp.div(value)

  /**
   * build a TensorOp to do element-wise division
   *
   * @param tensor Tensor[T]
   * @return TensorOp[T]
   */
  final def /(tensor: Tensor[T]): TensorOp[T] = this -> TensorOp.div(tensor)

  /**
   * build a TensorOp to do element-wise `f(x) = x ^ n`
   *
   * @param n the order of power
   * @return TensorOp[T]
   */
  final def **(n: T): TensorOp[T] = this -> TensorOp.pow(n)

  /**
   * build a TensorOp to do element-wise `f(x) = if (x>=a) 1; else 0`
   *
   * @param value Double a
   * @return TensorOp[T]
   */
  final def >=(value: Double): TensorOp[T] = this -> TensorOp.ge(value)

  /**
   * build a TensorOp to do element-wise `f(x) = if (x==a) 1; else 0`
   *
   * @param value T a
   * @return TensorOp[T]
   */
  final def ==(value: T): TensorOp[T] = this -> TensorOp.eq(value)
  // scalastyle:on

  /**
   * build a TensorOp to do matrix transposition for 2d Tensors
   *
   * @return TensorOp[T]
   */
  final def t: TensorOp[T] = this -> TensorOp.t()

  /**
   * build a TensorOp to do element-wise `f(x) = sqrt(x)`
   *
   * @return TensorOp[T]
   */
  final def sqrt: TensorOp[T] = this -> TensorOp.sqrt()

  /**
   * build a TensorOp to do element-wise `f(x) = log(x)`
   *
   * @return TensorOp[T]
   */
  final def log: TensorOp[T] = this -> TensorOp.log()

  /**
   * build a TensorOp to do element-wise `f(x) = log(x + 1)`
   *
   * @return TensorOp[T]
   */
  final def log1p: TensorOp[T] = this -> TensorOp.log1p()

  /**
   * build a TensorOp to do element-wise `f(x) = exp(x)`
   *
   * @return TensorOp[T]
   */
  final def exp: TensorOp[T] = this -> TensorOp.exp()

  /**
   * build a TensorOp to do element-wise `floor`
   *
   * @return TensorOp[T]
   */
  final def floor: TensorOp[T] = this -> TensorOp.floor()

  /**
   * build a TensorOp to do element-wise `ceil`
   *
   * @return TensorOp[T]
   */
  final def ceil: TensorOp[T] = this -> TensorOp.ceil()

  /**
   * build a TensorOp to do element-wise `f(x) = 1 / x`
   *
   * @return TensorOp[T]
   */
  final def inv: TensorOp[T] = this -> TensorOp.inv()

  /**
   * build a TensorOp to do element-wise `f(x) = -x`
   *
   * @return TensorOp[T]
   */
  final def neg: TensorOp[T] = this -> TensorOp.negative()

  /**
   * build a TensorOp to do element-wise `f(x) = |x|`
   *
   * @return TensorOp[T]
   */
  final def abs: TensorOp[T] = this -> TensorOp.abs()

  /**
   * build a TensorOp to do element-wise `f(x) = tanh(x)`
   *
   * @return TensorOp[T]
   */
  final def tanh: TensorOp[T] = this -> TensorOp.tanh()

  /**
   * build a TensorOp to do element-wise `f(x) = if (x>0) 1; if (x=0) 0; else -1`
   *
   * @return TensorOp[T]
   */
  final def sign: TensorOp[T] = this -> TensorOp.sign()

  /**
   * build a TensorOp to do element-wise `f(x) = 1 / (1 + exp(-x))`
   *
   * @return TensorOp[T]
   */
  final def sigmoid: TensorOp[T] = this -> TensorOp.sigmoid()

}

object TensorOp {

  /**
   * build a TensorOp with user-defined transformer
   *
   * @param transformer user-defined tensor transformer
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def apply[T: ClassTag](transformer: (Tensor[T], TensorNumeric[T]) => Tensor[T]
  )(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp(transformer)
  }

  /**
   * build a TensorOp with identity transformer
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t)
  }

  /**
   * build a TensorOp to do element-wise `f(x) = x + a`
   *
   * @param value T a
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def add[T: ClassTag](value: T)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp[T]((t: Tensor[T], _) => t.add(value))
  }

  /**
   * build a TensorOp to do element-wise tensor addition
   *
   * @param tensor Tensor[T]
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def add[T: ClassTag](tensor: Tensor[T])(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp[T]((t: Tensor[T], _) => t.add(tensor))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = x - a`
   *
   * @param value T a
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def sub[T: ClassTag](value: T)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp[T]((t: Tensor[T], _) => t.sub(value))
  }

  /**
   * build a TensorOp to do element-wise tensor subtraction
   *
   * @param tensor Tensor[T]
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def sub[T: ClassTag](tensor: Tensor[T])(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp[T]((t: Tensor[T], _) => t.sub(tensor))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = a * x`
   *
   * @param value T a
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def mul[T: ClassTag](value: T)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp[T]((t: Tensor[T], _) => t.mul(value))
  }

  /**
   * build a TensorOp to do element-wise multiplication
   *
   * @param tensor Tensor[T]
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def mul[T: ClassTag](tensor: Tensor[T])(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp[T]((t: Tensor[T], _) => t.cmul(tensor))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = x / a`
   *
   * @param value T a
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def div[T: ClassTag](value: T)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.div(value))
  }

  /**
   * build a TensorOp to do element-wise division
   *
   * @param tensor Tensor[T]
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def div[T: ClassTag](tensor: Tensor[T])(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.div(tensor))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = if (x>=a) 1; else 0`
   *
   * @param value Double a
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def ge[T: ClassTag](value: Double)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.ge(t, value))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = if (x==a) 1; else 0`
   *
   * @param value T a
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def eq[T: ClassTag](value: T)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.eq(t, value))
  }

  /**
   * build a TensorOp to do matrix transposition for 2d Tensors
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def t[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.t())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = sqrt(x)`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def sqrt[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.sqrt())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = log(x)`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def log[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.log())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = log(x + 1)`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def log1p[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.log1p())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = exp(x)`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def exp[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.exp())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = x ^ n`
   *
   * @param n the order of power
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def pow[T: ClassTag](n: T)(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.pow(n))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = x ^ 2`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def square[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.square())
  }

  /**
   * build a TensorOp to do element-wise `floor`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def floor[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.floor())
  }

  /**
   * build a TensorOp to do element-wise `ceil`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def ceil[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.ceil())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = 1 / x`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def inv[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.inv())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = -x`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def negative[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.negative(t))
  }

  /**
   * build a TensorOp to do element-wise `f(x) = |x|`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def abs[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.abs())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = tanh(x)`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def tanh[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.tanh())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = if (x>0) 1; if (x=0) 0; else -1`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def sign[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], _) => t.sign())
  }

  /**
   * build a TensorOp to do element-wise `f(x) = 1 / (1 + exp(-x))`
   *
   * @tparam T type param of TensorOp
   * @return TensorOp[T]
   */
  def sigmoid[T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorOp[T] = {
    new TensorOp((t: Tensor[T], ev: TensorNumeric[T]) => {
      t.negative(t).exp()
        .add(ev.one)
        .inv()
    })
  }

}


/**
 * Select and copy a Tensor from a [[Table]] with [[key]].
 * And do tensor transformation if [[transformer]] is defined.
 *
 * @param key the key of selected tensor
 * @param transformer user-defined transformer
 * @tparam T Numeric type
 */
class SelectTensor[T: ClassTag](
  private val key: scala.Any,
  private val transformer: TensorOp[T] = null
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    val selected = input[Tensor[T]](key)
    if (transformer != null) {
      output = transformer.updateOutput(selected)
    } else {
      // TODO: support SparseTensor.copy
      output.resizeAs(selected).copy(selected)
    }

    output
  }

}

object SelectTensor {

  def apply[T: ClassTag](
    key: scala.Any,
    transformer: TensorOp[T] = null
  )(implicit ev: TensorNumeric[T]): SelectTensor[T] = {
    new SelectTensor[T](key, transformer)
  }

}
