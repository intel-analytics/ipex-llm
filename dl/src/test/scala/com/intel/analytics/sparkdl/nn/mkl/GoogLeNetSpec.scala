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

import com.intel.analytics.sparkdl.models._
import org.scalatest.{FlatSpec, Matchers}

class GoogLeNetSpec extends FlatSpec with Matchers{
  // "GoogLeNet V1 with mkl dnn" should "ends with no segment fault" in {
  //   Perf.performance[Float](new Params(batchSize = 32, module = "googlenet_v2"))
  // }

  "GoogLeNet V1 with mkl dnn" should "ends with the same result" in {
    import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, Module}
    import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
    import com.intel.analytics.sparkdl.tensor.Tensor
    import scala.reflect.ClassTag

    def testModel[T : ClassTag]()(implicit tn : TensorNumeric[T]) : Unit = {
      val modelMkl = GoogleNet_v1[T](1000)
      val modelNN  = GoogleNetNN_v1[T](1000)

      val input = Tensor[T](32, 3, 224, 224)
      input.rand()
      println(modelMkl)
      println(modelNN)

      val criterion = new ClassNLLCriterion[T]()

      val labelsMkl = Tensor[T](32).fill(tn.fromType(1))
      val outputMkl = modelMkl.forward(input)
      criterion.forward(outputMkl, labelsMkl)
      val gradOutputMkl = criterion.backward(outputMkl, labelsMkl)
      val resultMkl = modelMkl.backward(input, gradOutputMkl)

      val labelNN = Tensor[T](32).fill(tn.fromType(1))
      val outputNN = modelNN.forward(input)
      criterion.forward(outputNN, labelNN)
      val gradOutputNN = criterion.backward(outputNN, labelNN)
      val resultNN = modelNN.backward(input, gradOutputNN)

      println(labelsMkl)
      println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      println(labelNN)

      println(outputMkl)
      println("==================================================================================")
      println(outputNN)

      outputMkl should be equals outputNN
      gradOutputMkl should be equals gradOutputNN
      resultMkl should be equals resultNN
      outputMkl should be equals input

      println(outputMkl.storage().array().length)
      println(input.storage().array().length)
    }

    testModel[Float]()
  }
}
