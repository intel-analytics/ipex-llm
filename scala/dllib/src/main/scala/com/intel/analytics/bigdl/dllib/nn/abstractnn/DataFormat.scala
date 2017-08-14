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

/**
 * DataFormat are used to specify the data format of the input and output data when data is
 * 2-D images.
 */
sealed trait DataFormat {
  def getHWCDims(inputDims: Int): (Int, Int, Int)

  val value: String
}

object DataFormat {
  def apply(formatString: String): DataFormat = {
    formatString.toUpperCase match {
      case "NHWC" => NHWC
      case "NCHW" => NCHW
    }
  }

  /**
   * Specify the input/output data format when data is stored in the order of
   * [batch, channels, height, width]
   */
  case object NCHW extends DataFormat {
    def getHWCDims(inputDims: Int): (Int, Int, Int) = {
      if (inputDims == 3) (2, 3, 1) else (3, 4, 2)
    }

    val value = "NCHW"
  }

  /**
   * Specify the input/output data format when data is stored in the order of
   * [batch, height, width, channels]
   */
  case object NHWC extends DataFormat {
    def getHWCDims(inputDims: Int): (Int, Int, Int) = {
      if (inputDims == 3) (1, 2, 3) else (2, 3, 4)
    }

    val value = "NHWC"
  }
}
