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
import com.intel.analytics.bigdl.numeric.NumericFloat

class AbstractModuleSpec extends FlatSpec with Matchers {
  "Get name" should "find the module if it exists" in {
    val m = Linear(4, 3).setName("module")
    m("module").get should be(m)
  }

  "Get name" should "find the module if it exists in container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.add(m)

    s("module").get should be(m)
  }

  "Get name" should "find the module if it exists in deeper container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.add(m)
    val k = Sequential()
    k.add(s)

    k("module").get should be(m)
  }

  "Get name" should "get the container if it is the container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.setName("container")
    s.add(m)

    s("container").get should be(s)
  }

  "Get name" should "not find if there is no such module" in {
    val m = Linear(4, 3)
    m("module") should be(None)
    val s = Sequential()
    s.add(m)
    s("container") should be(None)
  }

  "Get name" should "throw exception if there are two modules with same name" in {
    val m1 = Linear(4, 3)
    val m2 = Linear(4, 3)
    m1.setName("module")
    m2.setName("module")
    val s = Sequential()
    s.add(m1).add(m2)

    intercept[IllegalArgumentException] {
      s("module").get
    }
  }
}
