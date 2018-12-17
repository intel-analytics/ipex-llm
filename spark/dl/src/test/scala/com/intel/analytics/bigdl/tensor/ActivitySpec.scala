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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils.Table
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class ActivitySpec extends FlatSpec with Matchers {
  "Activity.allocate" should "be able to allocate table" in {
    val r = Activity.allocate[Table, Any]()
    r.isInstanceOf[Table] should be(true)
  }

  "Activity.allocate" should "be able to allocate Tensor[Boolean]" in {
    val r = Activity.allocate[Tensor[_], Boolean]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(BooleanType)
  }

  "Activity.allocate" should "be able to allocate Tensor[Char]" in {
    val r = Activity.allocate[Tensor[_], Char]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(CharType)
  }

  "Activity.allocate" should "be able to allocate Tensor[Short]" in {
    val r = Activity.allocate[Tensor[_], Short]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(ShortType)
  }

  "Activity.allocate" should "be able to allocate Tensor[Int]" in {
    val r = Activity.allocate[Tensor[_], Int]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(IntType)
  }

  "Activity.allocate" should "be able to allocate Tensor[Long]" in {
    val r = Activity.allocate[Tensor[_], Long]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(LongType)
  }

  "Activity.allocate" should "be able to allocate Tensor[Float]" in {
    val r = Activity.allocate[Tensor[_], Float]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(FloatType)
  }

  "Activity.allocate" should "be able to allocate Tensor[Double]" in {
    val r = Activity.allocate[Tensor[_], Double]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(DoubleType)
  }

  "Activity.allocate" should "be able to allocate Tensor[String]" in {
    val r = Activity.allocate[Tensor[_], String]()
    r.isInstanceOf[Tensor[_]] should be(true)
    r.asInstanceOf[Tensor[_]].getType() should be(StringType)
  }

  "Activity.allocate" should "be able to allocate nothing for Activity" in {
    val r = Activity.allocate[Activity, Any]()
    r should be(null)
  }
}
