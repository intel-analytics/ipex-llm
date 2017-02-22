/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.models.fasterrcnn.utils

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class AnchorSpec extends FlatSpec with Matchers {
  val anchorParam = AnchorParam(_scales = Array[Float](3, 6, 9, 16, 32),
    _ratios = Array(0.5f, 0.667f, 1.0f, 1.5f, 2.0f))
  val anchorTool = new Anchor(anchorParam)

  "generateBasicAnchors with 3 * 3 params" should "work well" in {
    val anchors = anchorTool.generateBasicAnchors(Tensor(Storage(Array[Float](0.5f, 1, 2))),
      Tensor(Storage(Array[Float](8, 16, 32))))
    val expected = Tensor(Storage(Array(
      -84.0, -176.0, -360.0, -56.0, -120.0, -248.0, -36.0, -80.0, -168.0,
      -40.0, -88.0, -184.0, -56.0, -120.0, -248.0, -80.0, -168.0, -344.0,
      99.0, 191.0, 375.0, 71.0, 135.0, 263.0, 51.0, 95.0, 183.0,
      55.0, 103.0, 199.0, 71.0, 135.0, 263.0, 95.0, 183.0, 359.0).map(x => x.toFloat)))
    expected.resize(4, 9)
    anchors should be(expected)
  }
  "generateBasicAnchors with 5 * 5 params" should "work well" in {
    val out = anchorTool.generateBasicAnchors(Tensor(Storage(Array[Float](0.5f, 1, 2, 4, 8))),
      Tensor(Storage(Array[Float](8, 16, 32, 64, 128))))
    val expected = Tensor(Storage(Array(
      -84.0, -176.0, -360.0, -728.0, -1464.0,
      -56.0, -120.0, -248.0, -504.0, -1016.0,
      -36.0, -80.0, -168.0, -344.0, -696.0,
      -24.0, -56.0, -120.0, -248.0, -504.0,
      -16.0, -40.0, -88.0, -184.0, -376.0,

      -40.0, -88.0, -184.0, -376.0, -760.0,
      -56.0, -120.0, -248.0, -504.0, -1016.0,
      -80.0, -168.0, -344.0, -696.0, -1400.0,
      -120.0, -248.0, -504.0, -1016.0, -2040.0,
      -184.0, -376.0, -760.0, -1528.0, -3064.0,

      99.0, 191.0, 375.0, 743.0, 1479.0,
      71.0, 135.0, 263.0, 519.0, 1031.0,
      51.0, 95.0, 183.0, 359.0, 711.0,
      39.0, 71.0, 135.0, 263.0, 519.0,
      31.0, 55.0, 103.0, 199.0, 391.0,

      55.0, 103.0, 199.0, 391.0, 775.0,
      71.0, 135.0, 263.0, 519.0, 1031.0,
      95.0, 183.0, 359.0, 711.0, 1415.0,
      135.0, 263.0, 519.0, 1031.0, 2055.0,
      199.0, 391.0, 775.0, 1543.0, 3079.0).map(x => x.toFloat)))
    expected.resize(4, 25)
    out should be(expected)
  }

  "generate shifts with tensor " should "work properly" in {
    val shifts2 = anchorTool.generateShifts(2, 3, 2)
    val expected = Tensor(Storage(Array(
      0.0, 2.0, 0.0, 2.0, 0.0, 2.0,
      0.0, 0.0, 2.0, 2.0, 4.0, 4.0).map(x => x.toFloat))).resize(2, 6)
    shifts2._1 should be(expected(1).resize(6, 1))
    shifts2._2 should be(expected(2).resize(6, 1))
  }

  "getAllAnchors" should "work properly" in {
    val anchorTool = new Anchor(AnchorParam(Array(3f, 2f), Array(4f, 5f)))
    val all = anchorTool.generateAnchors(3, 2, 3)

    val expected = Tensor(Storage(Array(
      -10.0, -14.5, -14.0, -19.5, -7.0,
      -11.5, -11.0, -16.5, -4.0, -8.5,
      -8.0, -13.5, -10.0, -14.5, -14.0,
      -19.5, -7.0, -11.5, -11.0, -16.5,
      -4.0, -8.5, -8.0, -13.5, -46.0,
      -59.5, -36.0, -47.0, -46.0, -59.5,
      -36.0, -47.0, -46.0, -59.5, -36.0,
      -47.0, -43.0, -56.5, -33.0, -44.0,
      -43.0, -56.5, -33.0, -44.0, -43.0,
      -56.5, -33.0, -44.0, 25.0, 29.5,
      29.0, 34.5, 28.0, 32.5, 32.0,
      37.5, 31.0, 35.5, 35.0, 40.5,
      25.0, 29.5, 29.0, 34.5, 28.0,
      32.5, 32.0, 37.5, 31.0, 35.5,
      35.0, 40.5, 61.0, 74.5, 51.0,
      62.0, 61.0, 74.5, 51.0, 62.0,
      61.0, 74.5, 51.0, 62.0, 64.0,
      77.5, 54.0, 65.0, 64.0, 77.5,
      54.0, 65.0, 64.0, 77.5, 54.0,
      65.0
    ).map(x => x.toFloat))).resize(4, 24)

    all should be(expected)
  }
}
