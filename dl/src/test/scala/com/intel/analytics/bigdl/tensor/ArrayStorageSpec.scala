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

package com.intel.analytics.bigdl.tensor

import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class ArrayStorageSpec extends FlatSpec with Matchers {

  "update value" should "be test" in {
    var i = 0
    val values = Array(1.0, 2.0, 3.0)
    val storage = Storage(values)
    val time = System.nanoTime()
    while (i < 1e8) {
      storage(0) = 2.0
      i += 1
    }

    println(s"spend time is ${(System.nanoTime() - time) / 1e6}ms")
  }

  "basic test" should "pass" in {
    val values = Array(1.0, 2.0, 3.0)
    val storage = Storage(values)
    storage(0) should be(1.0)
    storage.length() should be(3)
    storage(1) = 4.0
    storage.array()(1) should be(4.0)

    val iterator = storage.iterator
    var i = 0
    while (iterator.hasNext) {
      val v = iterator.next()
      v should be(values(i))
      i += 1
    }

    storage.resize(10)
    storage.length() should be(10)
    storage.fill(10.0, 1, 10)
    storage(9) should be(10.0)

  }

  "copy from double to double" should "pass" in {
    val values1 = Array(1.0, 2.0, 3.0)
    val values2 = Array(4.0, 5.0, 6.0)
    val storage1 = Storage(values1)
    val storage2 = Storage(values2)
    storage1.copy(storage2)
    storage2(0) = 8.0
    storage1(0) should be(4.0)
  }
}
