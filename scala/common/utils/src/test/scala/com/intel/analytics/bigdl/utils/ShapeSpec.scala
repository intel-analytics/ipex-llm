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

package com.intel.analytics.bigdl.utils

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ShapeSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "update of SingleShape" should "be test" in {
    assert(Shape(1, 2, 3).copyAndUpdate(-1, 20) == Shape(1, 2, 20))
  }

  "update of MultiShape" should "be test" in {
    val multiShape = Shape(List(Shape(1, 2, 3), Shape(4, 5, 6)))
    assert(multiShape.copyAndUpdate(-1, Shape(5, 5, 5)) ==
      Shape(List(Shape(1, 2, 3), Shape(5, 5, 5))))
  }

  "multiShape not equal" should "be test" in {
    intercept[RuntimeException] {
      assert(Shape(List(Shape(1, 2, 3), Shape(5, 5, 5))) ==
        Shape(List(Shape(1, 2, 3), Shape(5, 6, 5))))
    }}

  "singleShape not equal" should "be test" in {
    intercept[RuntimeException] {
      assert(Shape(1, 2, 3) == Shape(1, 2, 4))
    }}
}
