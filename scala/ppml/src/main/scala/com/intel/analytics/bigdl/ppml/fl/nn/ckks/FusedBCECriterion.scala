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
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}

class FusedBCECriterion(val ckksCommonPtr: Long) {
  val ckks = new CKKS()
  var ckksOutput : Array[Array[Byte]] = null

  def forward(input: Array[Byte], target: Array[Byte]): Array[Byte] = {
//  Log4Error.invalidInputError(input.size().sameElements(target.size()),
//  s"input size should be equal to target size, but got input size: ${input.size().toList}," +
//  s" target size: ${target.size().toList}")
    ckksOutput = ckks.train(ckksCommonPtr, input, target)
    ckksOutput(0)
  }

  def forward(input: Tensor[Byte], target: Tensor[Byte]): Tensor[Byte] = {
    //  Log4Error.invalidInputError(input.size().sameElements(target.size()),
    //  s"input size should be equal to target size, but got input size: ${input.size().toList}," +
    //  s" target size: ${target.size().toList}")
    val loss = forward(input.storage.array(), target.storage.array())
    Tensor[Byte](Storage[Byte](loss)).resize(input.size())
  }

  def backward(input: Array[Byte], target: Array[Byte]): Array[Byte] = {
    ckksOutput(1)
  }

  def backward(input: Tensor[Byte], target: Tensor[Byte]): Tensor[Byte] = {
    Tensor[Byte](Storage(ckksOutput(1))).resize(input.size())
  }
}


object FusedBCECriterion {
  def apply(ckksCommonPtr: Long): FusedBCECriterion = {
    new FusedBCECriterion(ckksCommonPtr)
  }
}

