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

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}

class UtilsSpec extends FlatSpec with Matchers {

  "getNamedModules" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    val model = Sequential().setName("model")
      .add(Identity().setName("id"))
      .add(ReLU().setName("relu"))

    val namedMudules = Utils.getNamedModules(model)

    namedMudules("model").isInstanceOf[Sequential[Float]] should be (true)
    namedMudules("id").isInstanceOf[Identity[Float]] should be (true)
    namedMudules("relu").isInstanceOf[ReLU[Float]] should be (true)
  }

  "isLayerwised" should "work properly" in {
    val model = Sequential[Double]().add(Identity()).add(ReLU())
    Utils.isLayerwiseScaled(model) should be (false)
    model.setScaleB(2.0)
    Utils.isLayerwiseScaled(model) should be (true)

    val model2 = Sequential[Double]().add(SpatialConvolution[Double](2, 2, 2, 2)
      .setScaleW(3.0)).add(ReLU())
    Utils.isLayerwiseScaled(model2) should be (true)

    val model3 = Sequential[Double]().add(SpatialConvolution[Double](2, 2, 2, 2)
      .setScaleB(2.0)).add(ReLU())
    Utils.isLayerwiseScaled(model3) should be (true)

  }

}
