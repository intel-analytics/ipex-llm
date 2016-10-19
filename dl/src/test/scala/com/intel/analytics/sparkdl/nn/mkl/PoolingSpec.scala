/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.nn
import com.intel.analytics.sparkdl.nn.{Constant, Default, SpatialMaxPooling, Xavier}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag
class PoolingSpec extends FlatSpec with Matchers {
  "SpatialMaxPooling ceil mode" should "generate correct output and gradient input" in {
    def test[T : ClassTag]()(implicit ev : TensorNumeric[T]): Unit = {
      val maxPoolDnn  = new    SpatialMaxPooling[T](3, 3, 2, 2).ceil()
      val maxPoolBlas = new nn.SpatialMaxPooling[T](3, 3, 2, 2).ceil()

      for (i <- 0 until 5) {
        val input = Tensor[T](32, 64, 112, 112).rand()

        val outputDnn  = maxPoolDnn.forward(input)
        val outputBlas = maxPoolBlas.forward(input)

        Tools.AverageError(outputDnn, outputBlas, "output") should be (0.0 +- 1e-6)

        val gradOutput = Tensor[T]().resizeAs(outputDnn).rand()

        val gradInputDnn  = maxPoolDnn.backward(input, gradOutput)
        val gradInputBlas = maxPoolBlas.backward(input, gradOutput)

        Tools.CumulativeError(gradInputDnn, gradInputBlas, "gradOutput")
        Tools.AverageError(gradInputDnn, gradInputBlas, "gradOutput") should be (0.0 +- 1e-6)
      }
    }

    for (i <- 0 until Tools.GetRandTimes()) {
      test[Float]()
    }
  }

  "SpatialAvergePooling ceil mode" should "generate correct output and gradient input" in {
    def test[T : ClassTag]()(implicit ev : TensorNumeric[T]): Unit = {
      val maxPoolDnn  = new    SpatialAveragePooling[T](5, 5, 3, 3).ceil()
      val maxPoolBlas = new nn.SpatialAveragePooling[T](5, 5, 3, 3).ceil()

      for (i <- 0 until 5) {
        val input = Tensor[T](8, 64, 112, 112).rand()

        val outputDnn  = maxPoolDnn.forward(input)
        val outputBlas = maxPoolBlas.forward(input)

        Tools.AverageError(outputDnn, outputBlas, "output") should be (0.0 +- 1e-6)

        val gradOutput = Tensor[T]().resizeAs(outputDnn).rand()

        val gradInputDnn  = maxPoolDnn.backward(input, gradOutput)
        val gradInputBlas = maxPoolBlas.backward(input, gradOutput)

        Tools.CumulativeError(gradInputDnn, gradInputBlas, "gradOutput")
        Tools.AverageError(gradInputDnn, gradInputBlas, "gradOutput") should be (0.0 +- 1e-6)
      }
    }

    for (i <- 0 until Tools.GetRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }
  "SpatialAvergePooling ceil mode 7 7 1 1" should "generate correct output and gradient input" in {
    def test[T : ClassTag]()(implicit ev : TensorNumeric[T]): Unit = {
      val maxPoolDnn  = new    SpatialAveragePooling[T](7, 7, 1, 1).ceil()
      val maxPoolBlas = new nn.SpatialAveragePooling[T](7, 7, 1, 1).ceil()

      for (i <- 0 until 5) {
        val input = Tensor[T](8, 1024, 7, 7).rand()

        val outputDnn  = maxPoolDnn.forward(input)
        val outputBlas = maxPoolBlas.forward(input)

        Tools.AverageError(outputDnn, outputBlas, "output") should be (0.0 +- 1e-6)

        val gradOutput = Tensor[T]().resizeAs(outputDnn).rand()

        val gradInputDnn  = maxPoolDnn.backward(input, gradOutput)
        val gradInputBlas = maxPoolBlas.backward(input, gradOutput)

        Tools.CumulativeError(gradInputDnn, gradInputBlas, "gradInput")
        Tools.AverageError(gradInputDnn, gradInputBlas, "gradOutput") should be (0.0 +- 1e-6)
      }
    }

    for (i <- 0 until Tools.GetRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }
}
