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

package com.intel.analytics.bigdl.dllib.common

import com.intel.analytics.bigdl.mkl.{MKL => BMKL}
import com.intel.analytics.bigdl.dllib.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import org.apache.logging.log4j.LogManager
// import com.intel.analytics.zoo.mkl.MKL.{vdErf, vsErf}

import scala.reflect.ClassTag

private[bigdl] object zooMKLBlas {
  private val logger = LogManager.getLogger(getClass)

  def erf[T: ClassTag](tensor: Tensor[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    /* uncomment below lines after zoo core merged into bigdl core
    if (BMKL.isMKLLoaded && tensor.isContiguous()) {
      ev.getType() match {
        case FloatType =>
          val value = tensor.storage().array().asInstanceOf[Array[Float]]
          vsErf(tensor.nElement(), value, tensor.storageOffset() - 1,
            value, tensor.storageOffset() - 1)
        case DoubleType =>
          val value = tensor.storage().array().asInstanceOf[Array[Double]]
          vdErf(tensor.nElement(), value, tensor.storageOffset() - 1,
            value, tensor.storageOffset() - 1)
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    } else {
      logger.warn("MKL is not used for erf, with mkl the performance will be much better")
      tensor.erf()
    }
    */
    tensor.erf()
  }

}
