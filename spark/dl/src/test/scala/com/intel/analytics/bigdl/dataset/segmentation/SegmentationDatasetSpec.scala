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

package com.intel.analytics.bigdl.dataset.segmentation

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class SegmentationDatasetSpec extends FlatSpec with Matchers with BeforeAndAfter {

  val compressed1 = "Q:XX24Sm0e:jQ1]EajKSV69iZJV9T_OdF[>^NmB`2Z`0Y?a^OmR5lc6Zj2m[IckG0ZEXdl9l0j" +
    "[SFT9_\\e1Z:XgZNh[OPM:d\\4O"
  val compressed2 = "iaj0]TWZ2j9XZleMYG_oLf9U`]7mQbWg0b[gHlV[RYOQmZQ2TTU1oj]PNeVbEl[VNnZ" +
    "]OkYLgfMja01fgW=\\1TofBY6c:Sheb0`n1Q[dol1PXc`0YQh]RNi^Z_OZeMOb?30nbR1^P^g0ShmNfPkYO" +
    "^LWkkNXW3]_m0gUQ[2kdb?ZePeMhPZB^[NaQQMgZLkVlU54aUSjJ32"

  val arr1 = Array(321, 2312, 4, 3243, 345, 4325, 6, 54, 6345, 63, 546, 357, 6, 57,
    465, 7,
    46, 87, 568, 576, 9, 5789, 6789, 8679, 2, 346, 2, 4, 324234, 32, 4, 324, 54675, 654, 123,
    6, 27, 16, 4527, 15)

  val arr2 = Array(27193, 2432141, 314, 3541, 35, 452, 345, 243657, 24365462, 5435,
    325234, 2146524, 363254, 63547, 21451, 4535, 2345, 754, 0, 1324, 1, 435234, 45, 6, 246,
    345, 612345, 2345, 64563546, 546345, 2435, 2, 45, 1, 543, 4, 543, 35426, 768557, 357,
    42563, 243, 5546, 3547, 35735, 2462354, 546354, 5436, 97866, 3754, 635, 1, 5436246,
    5, 7, 8, 9)
  "string2RLE" should "run well" in {

    val result = MaskUtils.string2RLE(compressed1, 100, 200)
    result.counts should be(arr1)
    result.height should be (100)
    result.width should be (200)

    val result2 = MaskUtils.string2RLE(compressed2, 100, 200)
    result2.counts should be (arr2)
    result2.height should be (100)
    result2.width should be (200)
  }

  "RLE2String" should "run well" in {
    MaskUtils.RLE2String(RLEMasks(arr2, 100, 200)) should be (compressed2)
    MaskUtils.RLE2String(RLEMasks(arr1, 100, 200)) should be (compressed1)
  }

  // real data in instances_val2014.json
  // annId = 455475
  val poly1 = Array(426.91, 58.24, 434.49, 77.74, 467.0, 80.99, 485.42, 86.41, 493.0, 129.75,
    521.17, 128.67, 532.01, 144.92, 545.01, 164.42, 552.6, 170.93, 588.35, 178.51, 629.53,
    165.51, 629.53, 177.43, 578.6, 214.27, 558.01, 241.35, 526.59, 329.12, 512.51, 370.29,
    502.75, 415.8, 418.24, 409.3, 399.82, 414.72, 388.98, 420.14, 382.48, 424.47, 391.15, 430.97,
    414.99, 425.55, 447.49, 427.72, 449.66, 435.3, 431.24, 438.56, 421.49, 452.64, 422.57,
    456.98, 432.33, 464.56, 439.91, 458.06, 481.08, 465.64, 502.75, 464.56, 507.09, 473.23,
    639.28, 474.31, 639.28, 1.9, 431.24, 0.0
  ).map(_.toFloat)

  // annId = 692513
  val poly2 = Array(
    416.41, 449.28, 253.36, 422.87, 234.06, 412.2, 277.23, 406.61, 343.77, 411.69, 379.84,
    414.23, 384.41, 424.9, 397.11, 427.95, 410.31, 427.95, 445.36, 429.98, 454.0, 438.61, 431.65,
    438.61, 423.01, 449.28
  ).map(_.toFloat)

  "poly2RLE" should "run well" in {
    val rle = MaskUtils.poly2RLE(PolyMasks(Array(poly1), 480, 640), 480, 640)
    val targetRle = MaskUtils.string2RLE(
      "Xnc51n>2N2O0O2N2O1N101N10O0100O100O01000O10O10O100000O010000O01000O1000000O1001O00ZBAk" +
        "<?TCFh<:WCHh<8VCJj<6UCLj<4TCOd1Ih88cE1V18T9HcE2k0g0_9WOeE4>T1k9hNfE64_1S:[NhE84`1Q:X" +
        "NjE95a1P:UNkE:5d1m9RNnE96e1l9RNnE87f1k9RNnE78g1j9RNmE7:g1i9RNmE6;h1h9RNmE5<i1g9RNmE5" +
        "<i1g9RNmE5<j1f9QNnE5<j1f9RNlE6=h1g9RNlE6=h1g9RNlE6=h1h9QNkE7=h1h9QNkE7=h1h9TNhE5?g1i" +
        "9QOWFo0i9QOWFo0i9QOWFP1h9POXFP1h9POXFP1h9QOWFo0i9QOWFo0i9QOWFo0i9QOWFo0i9QOWFo0j9QOU" +
        "Fo0k9QOUFo0k9QOUFP1j9POVFP1j9POVFP1j9QOUFo0k9QOUFo0k9QOUFo0k9QOUFo0k9QOUFP1j9POVFP1j" +
        "9QOUFo0k9QOUFP1k9oNUFQ1k9oNUFQ1k9oNUFQ1k9POTFQ1k9oNUFQ1k9oNUFQ1k9oNUFR1j9nNVFR1j9nNV" +
        "FR1j9oNUFR1j9nNWFQ1i9nNXFR1h9nNXFS1g9mNYFY1b9fN^F_1]9aNcFe1W9[NiFk1Q9UNoFP2l8PNTGV2f" +
        "8jMZG\\2`8dM`G_2]8aMcG_2]8aMcG_2]8aMcG_2]8aMcG_2]8aMcG_2]8aMcG_2^8`MbG_2_8aMaG_2_8aM" +
        "aG_2_8aMaG_2[8fMdGZ2X8lMfGT2U8SNiGm1R8ZNlGf1P8_NoGa1l7dNTH\\1h7hNXHX1c7mN]HS1^7RObHn" +
        "0Z7VOfHj0V7ZOjHf0T7\\OlHd0Q7_OoHa0n6BRI>k6EUI;h6HXI8e6K[I5b6N^I2_61aI0[63eIOV64jINQ6" +
        "5oILm57SJKh58XJIe59[JI`5:`JG\\5<dJFW5=iJDS5?mJCo4?QKBk4a0UKAf4b0ZK_Ob4d0^K^O]4e0cK\\" +
        "OZ4f0fK\\OU4g0kKZOQ4i0oKYOl3k0SLVOi3m0WLUOe3m0[LTOa3o0_LSO\\3P1dLQOX3R1hLoNT3T1lLlNR" +
        "3V1nLkNn2X1RMiNj2Z1VMgNf2\\1ZMeNb2^1^McN_2_1aMaN\\2b1dM^NY2e1gM\\NU2g1kMYNR2j1nMVNP2" +
        "l1PNTNn1n1RNRNl1P2TNQNj1P2VNPNi1Q2WNoMg1S2YNmMf1T2ZNmMd1T2\\NlMb1V2^NjMa1W2_NiM`1X2`" +
        "NhM^1Z2bNgM\\1Z2dNfM[1[2eNeMY1]2gNcMX1^2hNbMW1_2iNbMU1_2kNaMS1a2mN_MR1b2nN^MQ1c2oN]M" +
        "o0e2QO\\Mm0e2SO[Mm0e2SO[Ml0f2TOZMk0g2UOZMj0f2VOZMi0g2WOYMh0h2XOXMg0i2ZOVMf0j2ZOWMd0j" +
        "2\\OUMd0l2\\OTMd0l2\\OTMc0m2]ORMc0o2]OQMb0P3^OPMb0P3^OoLb0R3^OnLa0S3_OmL`0T3@kLa0U3_" +
        "OkL`0V3@jL?W3AhL`0X3@hL?Y3AgL>Z3BeL>\\3BdL>\\3BdL=]3CcL<^3DaL=_3CaL<`3D`L;a3E^L;c3E]" +
        "L;c3E]L:d3F[L:f3FZL:f3FZL9g3GXL9i3GWL8j3HVL8j3HUL8l3HTL7m3ISL6n3JQL7o3IQL6P4JPL5Q4Ko" +
        "K5Q4KnK5S4KmK4T4LlK3U4M_50000000000000000n>", 480, 640
    )
    rle(0).counts.length should be (targetRle.counts.length)
    rle(0).counts.zip(targetRle.counts).foreach{case (rleCount, targetCount) =>
      rleCount should be (targetCount +- 1)
    }

    val rle2 = MaskUtils.poly2RLE(PolyMasks(Array(poly2), 480, 640), 480, 640)
    MaskUtils.RLE2String(rle2(0)) should be(
      "la^31o>1O001N101O001O001O001N2O001O001O001O0O1000001O00000O10001O000000000O2O0000000000" +
        "1O0000000000001O00000000010O00000000001O00000000001O01O00000001O00000000001O0001O0000" +
        "0001O00000000001O0001O000001O00000000001O00000001O01O00000000001O0000000001O01O000000" +
        "00001O000000000010O0000000001O00000002N2O1N3M2N000010O000001O0000010O0000000001O00000" +
        "000001O00000000001O000000000001O0000000O1O1O1O1O1N2O1O1O10000000001O00000000000000001" +
        "O1O1O1O1O1O1O1OZhf2")

  }

  "mergeRLEs" should "run well" in {
    val rle1 = MaskUtils.poly2RLE(PolyMasks(Array(poly1), 480, 640), 480, 640)(0)
    val rle2 = MaskUtils.poly2RLE(PolyMasks(Array(poly2), 480, 640), 480, 640)(0)
    val merged = MaskUtils.mergeRLEs(Array(rle1, rle2), false)
    val targetRle = MaskUtils.string2RLE(
      "la^31o>1O001N101O001O001O001N2O001O001O001O0O1000001O00000O10001O000000000O2O0000000000" +
        "1O0000000000001O00000000010O00000000001O00000000001O01O00000001O00000000001O0001O0000" +
        "0001O00000000001O0001O000001O00000000001O00000001O01O00000000001O0000000001O01O000000" +
        "00001O000000000010O0000000001O00000002N2O1N3M00O100O2N100O100O100O2O0O100O10001N10000" +
        "O10001O[OTB6k=KUB5k=KUB5k=KUB5j=KWB5i=KWB6h=JXB6g=KYB5g=KYB5g=KYB5g=KYB5f=LZB4f=L[B3f" +
        "=LZB4f=LZB40]Ob=?^B4OB_=:bB3OE^=8cB2NH_=6cB1NK^=4dB?T2YOh88TE`0e1IT9HVE?X1:_9WOYE`0j0" +
        "h0k9hN[Ea0?T1S:[N^Eb0>V1Q:XNaEb0>X1P:UNbEc0>[1m9RNeEb0?\\1l9RNeEa0`0]1k9RNeE`0a0^1j9R" +
        "NfE>a0`1i9RNfE=b0a1h9RNfE<c0b1g9RNfE<c0b1g9RNfE<c0c1f9QNgE<c0c1f9RNfE<c0b1g9RNfE<c0b1" +
        "g9RNfE<c0b1h9QNfE<b0c1h9QNgE;a0d1h9TNeE8b0d1i9QOWFo0i9QOWFo0i9QOWFP1h9POXFP1h9POXFP1h" +
        "9QOWFo0i9QOWFo0i9QOWFo0i9QOWFo0i9QOWFo0j9QOUFo0k9QOUFo0k9QOUFP1j9POVFP1j9POVFP1j9QOUF" +
        "o0k9QOUFo0k9QOUFo0k9QOUFo0k9QOUFP1j9POVFP1j9QOUFo0k9QOUFP1k9oNUFQ1k9oNUFQ1k9oNUFQ1k9P" +
        "OTFQ1k9oNUFQ1k9oNUFQ1k9oNUFR1j9nNVFR1j9nNVFR1j9oNUFR1j9nNWFQ1i9nNXFR1h9nNXFS1g9mNYFY1" +
        "b9fN^F_1]9aNcFe1W9[NiFk1Q9UNoFP2l8PNTGV2f8jMZG\\2`8dM`G_2]8aMcG_2]8aMcG_2]8aMcG_2]8aM" +
        "cG_2]8aMcG_2]8aMcG_2^8`MbG_2_8aMaG_2_8aMaG_2_8aMaG_2[8fMdGZ2X8lMfGT2U8SNiGm1R8ZNlGf1P" +
        "8_NoGa1l7dNTH\\1h7hNXHX1c7mN]HS1^7RObHn0Z7VOfHj0V7ZOjHf0T7\\OlHd0Q7_OoHa0n6BRI>k6EUI;" +
        "h6HXI8e6K[I5b6N^I2_61aI0[63eIOV64jINQ65oILm57SJKh58XJIe59[JI`5:`JG\\5<dJFW5=iJDS5?mJC" +
        "o4?QKBk4a0UKAf4b0ZK_Ob4d0^K^O]4e0cK\\OZ4f0fK\\OU4g0kKZOQ4i0oKYOl3k0SLVOi3m0WLUOe3m0[L" +
        "TOa3o0_LSO\\3P1dLQOX3R1hLoNT3T1lLlNR3V1nLkNn2X1RMiNj2Z1VMgNf2\\1ZMeNb2^1^McN_2_1aMaN\\" +
        "2b1dM^NY2e1gM\\NU2g1kMYNR2j1nMVNP2l1PNTNn1n1RNRNl1P2TNQNj1P2VNPNi1Q2WNoMg1S2YNmMf1T2Z" +
        "NmMd1T2\\NlMb1V2^NjMa1W2_NiM`1X2`NhM^1Z2bNgM\\1Z2dNfM[1[2eNeMY1]2gNcMX1^2hNbMW1_2iNbM" +
        "U1_2kNaMS1a2mN_MR1b2nN^MQ1c2oN]Mo0e2QO\\Mm0e2SO[Mm0e2SO[Ml0f2TOZMk0g2UOZMj0f2VOZMi0g2" +
        "WOYMh0h2XOXMg0i2ZOVMf0j2ZOWMd0j2\\OUMd0l2\\OTMd0l2\\OTMc0m2]ORMc0o2]OQMb0P3^OPMb0P3^O" +
        "oLb0R3^OnLa0S3_OmL`0T3@kLa0U3_OkL`0V3@jL?W3AhL`0X3@hL?Y3AgL>Z3BeL>\\3BdL>\\3BdL=]3CcL" +
        "<^3DaL=_3CaL<`3D`L;a3E^L;c3E]L;c3E]L:d3F[L:f3FZL:f3FZL9g3GXL9i3GWL8j3HVL8j3HUL8l3HTL7" +
        "m3ISL6n3JQL7o3IQL6P4JPL5Q4KoK5Q4KnK5S4KmK4T4LlK3U4M_50000000000000000n>", 480, 640
    )
    merged.counts.length should be (targetRle.counts.length)
    merged.counts.zip(targetRle.counts).foreach{case (rleCount, targetCount) =>
      rleCount should be (targetCount +- 1)
    }
  }

}

