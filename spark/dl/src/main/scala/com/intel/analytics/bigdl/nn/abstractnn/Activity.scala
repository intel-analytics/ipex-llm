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

package com.intel.analytics.bigdl.nn.abstractnn

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect._

/**
 * [[Activity]] is a trait which represents
 * the concept of neural input within neural
 * networks. For now, two type of input are
 * supported and extending this trait, which
 * are [[Tensor]] and [[Table]].
 */
trait Activity {
  def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D]

  def toTable: Table

  def isTensor: Boolean

  def isTable: Boolean
}

/**
 * Sometimes a module may not have gradInput in the backward(e.g. some operation layer or
 * stopGradient in a Graph). This is allowed when the gradInput is not used anywhere.
 *
 * In such case, the gradInput of the module should be marked as EmptyGradInput. This class make
 * sure an error will happen when user try to use such gradInput.
 */
class EmptyGradInput private[abstractnn](moduleName: String) extends Activity with Serializable {

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] =
    throw new UnsupportedOperationException(s"The gradInput of $moduleName is empty. You should" +
      s"not use it anywhere")

  override def toTable: Table =
    throw new UnsupportedOperationException(s"The gradInput of $moduleName is empty. You should" +
      s"not use it anywhere")

  override def isTensor: Boolean =
    throw new UnsupportedOperationException(s"The gradInput of $moduleName is empty. You should" +
      s"not use it anywhere")

  override def isTable: Boolean =
    throw new UnsupportedOperationException(s"The gradInput of $moduleName is empty. You should" +
      s"not use it anywhere")
}

object Activity {
  /**
   * Allocate a data instance by given type D and numeric type T
   * @tparam D Data type
   * @tparam T numeric type
   * @return
   */
  def allocate[D <: Activity: ClassTag, T : ClassTag](): D = {
    val buffer = if (classTag[D] == classTag[Table]) {
      T()
    } else if (classTag[D] == classTag[Tensor[_]]) {
      if (classTag[Boolean] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericBoolean
        Tensor[Boolean]()
      } else if (classTag[Char] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericChar
        Tensor[Char]()
      } else if (classTag[Short] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericShort
        Tensor[Short]()
      } else if (classTag[Int] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericInt
        Tensor[Int]()
      } else if (classTag[Long] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericLong
        Tensor[Long]()
      } else if (classTag[Float] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericFloat
        Tensor[Float]()
      } else if (classTag[Double] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericDouble
        Tensor[Double]()
      } else if (classTag[String] == classTag[T]) {
        import com.intel.analytics.bigdl.numeric.NumericString
        Tensor[String]()
      } else if (classTag[ByteString] == classTag[T]) {
        import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
        Tensor[ByteString]()
      } else {
        throw new IllegalArgumentException("Type T activity is not supported")
      }
    } else {
      null
    }
    buffer.asInstanceOf[D]
  }

  def emptyGradInput(name: String): EmptyGradInput = new EmptyGradInput(name)
}
