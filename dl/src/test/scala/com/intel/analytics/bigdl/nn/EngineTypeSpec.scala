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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.models.imagenet.{GoogleNet_v1, GoogleNet_v2}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, Mixed, MklBlas, MklDnn}
import org.scalatest.{FlatSpec, Matchers}

class EngineTypeSpec extends FlatSpec with Matchers {
  "Module.getEngineType" should "return right type" in {
    Engine.setEngineType(MklBlas)
    val model = GoogleNet_v2[Float](1000)
    model.getEngineType() should be (MklBlas)

    Engine.setEngineType(MklDnn)
    val model2 = GoogleNet_v1[Float](1000)
    model2.getEngineType() should be (MklDnn)

    val model3 = Sequential[Tensor[Float], Tensor[Float], Float]
    model3.add(model)
    model3.add(model2)
    model3.getEngineType() should be (Mixed)
    println(model3.getEngineType())
  }
}
