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
package com.intel.analytics.bigdl.ppml.fl.nn.ckks

import com.intel.analytics.bigdl.ckks.CKKS
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.{ClassTag, classTag}

class Encryptor[T: ClassTag](val ckksEncryptorPtr: Long)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[Byte], T] {
  val ckks = new CKKS()

  val floatInput = Activity.allocate[Tensor[Float], Float]()

  override def updateOutput(input: Tensor[T]): Tensor[Byte] = {
    val floatInput = if (classTag[T] == classTag[Float]) {
      input.toTensor[Float]
    } else {
      input.cast[Float](this.floatInput)
    }
    val enInput = ckks.ckksEncrypt(ckksEncryptorPtr, floatInput.storage().array())
    output = Tensor[Byte](Storage[Byte](enInput)).resize(input.size())
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[Byte]): Tensor[T] = {
    val deGradOutput = ckks.ckksDecrypt(ckksEncryptorPtr, gradOutput.storage().array())
    val floatGradInput = Tensor[Float](Storage[Float](deGradOutput)).resize(input.size())
    gradInput = if (classTag[T] == classTag[Float]) {
      floatGradInput.toTensor[T]
    } else {
      floatGradInput.cast[T](gradInput)
    }
    gradInput
  }

}

object Encryptor {
  def apply[T: ClassTag](
        ckksEncryptorPtr: Long)(implicit ev: TensorNumeric[T]): Encryptor[T] = {
    new Encryptor[T](ckksEncryptorPtr)
  }
}
