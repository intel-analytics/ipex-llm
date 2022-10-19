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

class CAddTable(val ckksCommonPtr: Long) {
  val ckks = new CKKS()

  def updateOutput(input: Array[Byte]*): Array[Byte] = {
    //  Log4Error.invalidInputError(input.size().sameElements(target.size()),
    //  s"input size should be equal to target size, but got input size: ${input.size().toList}," +
    //  s" target size: ${target.size().toList}")
    var ckksOutput = input(0)
    if (input.size > 1) {
      (1 until input.size).foreach{i =>
        ckksOutput = ckks.cadd(ckksCommonPtr, ckksOutput, input(i))
      }
    }
    ckksOutput
  }

  def updateGradInput(input: Array[Byte]*): Array[Byte] = {
    input.last
  }
}


object CAddTable {
  def apply(ckksCommonPtr: Long): CAddTable = {
    new CAddTable(ckksCommonPtr)
  }
}


