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

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest._
import flatspec._
import matchers._

class SegmentationDatasetSpec extends AnyFlatSpec with should.Matchers with BeforeAndAfter {

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

  // scalastyle:off
  // annId 902000091136
  val _rle1 = Array[Int](68483, 6, 473, 8, 471, 10, 469, 12, 467, 13, 456, 6, 5, 13, 455, 8, 1, 16, 454, 26, 453, 27, 453, 26, 454, 26, 453, 26, 455, 24, 456, 21, 459, 20, 461, 18, 463, 17, 465, 14, 467, 13, 467, 17, 462, 19, 459, 22, 457, 24, 455, 25, 454, 26, 454, 26, 454, 26, 453, 28, 453, 28, 452, 28, 452, 28, 453, 27, 454, 26, 456, 4, 3, 17, 463, 16, 464, 16, 464, 19, 460, 21, 459, 22, 458, 23, 456, 24, 456, 24, 457, 23, 457, 23, 457, 24, 456, 24, 456, 24, 456, 24, 455, 25, 456, 24, 456, 23, 457, 11, 2, 10, 458, 10, 3, 8, 474, 3, 17764, 6, 473, 8, 471, 10, 469, 12, 468, 19, 461, 20, 459, 22, 459, 22, 458, 22, 458, 22, 459, 21, 460, 20, 459, 21, 459, 20, 459, 21, 458, 21, 458, 19, 461, 20, 460, 21, 458, 22, 459, 21, 459, 21, 459, 21, 460, 20, 461, 18, 464, 3, 3, 10, 471, 8, 474, 3, 1897, 6, 473, 8, 471, 10, 469, 12, 468, 17, 463, 18, 461, 20, 461, 20, 460, 20, 460, 20, 461, 19, 462, 18, 464, 16, 465, 14, 466, 14, 465, 14, 467, 12, 468, 12, 468, 11, 470, 10, 471, 8, 474, 3, 7647, 4, 475, 6, 473, 7, 473, 7, 468, 12, 467, 13, 466, 14, 465, 15, 465, 15, 465, 14, 465, 15, 466, 14, 466, 14, 466, 14, 455, 4, 8, 13, 455, 5, 7, 13, 454, 7, 6, 13, 453, 9, 6, 12, 447, 1, 3, 11, 7, 10, 447, 2, 3, 11, 7, 10, 446, 3, 3, 11, 7, 9, 446, 5, 3, 10, 7, 10, 445, 5, 3, 10, 7, 10, 445, 5, 3, 9, 9, 9, 444, 6, 3, 9, 12, 6, 445, 5, 3, 8, 13, 7, 444, 6, 2, 5, 17, 6, 444, 6, 2, 3, 19, 6, 445, 6, 1, 3, 19, 7, 445, 8, 20, 7, 473, 7, 473, 7, 473, 8, 472, 8, 472, 8, 3, 2, 467, 8, 2, 5, 465, 8, 2, 8, 462, 8, 1, 11, 460, 8, 1, 13, 458, 24, 458, 23, 458, 22, 459, 21, 459, 21, 459, 20, 461, 19, 461, 19, 461, 18, 463, 17, 464, 16, 434, 1, 29, 16, 431, 4, 30, 14, 427, 9, 31, 13, 427, 9, 31, 13, 427, 9, 32, 11, 427, 11, 32, 10, 427, 12, 31, 10, 427, 13, 30, 9, 427, 14, 31, 8, 428, 14, 31, 6, 428, 16, 32, 4, 429, 16, 464, 16, 464, 16, 465, 15, 466, 14, 468, 12, 467, 12, 467, 13, 466, 13, 466, 12, 468, 12, 468, 12, 467, 13, 468, 13, 467, 14, 465, 15, 464, 16, 463, 17, 462, 18, 462, 18, 462, 17, 462, 18, 463, 18, 462, 18, 462, 18, 463, 17, 464, 16, 466, 14, 466, 13, 466, 14, 467, 12, 468, 12, 468, 11, 470, 10, 471, 8, 474, 3, 1917, 6, 473, 8, 471, 10, 469, 12, 468, 12, 468, 12, 467, 13, 467, 13, 468, 12, 468, 12, 468, 11, 468, 12, 468, 12, 468, 12, 467, 13, 468, 12, 468, 12, 468, 11, 467, 16, 463, 18, 461, 20, 459, 22, 458, 22, 458, 22, 457, 23, 457, 23, 456, 24, 456, 23, 457, 23, 456, 23, 458, 20, 460, 21, 459, 22, 459, 22, 459, 21, 460, 20, 460, 20, 459, 21, 459, 21, 459, 21, 458, 21, 460, 20, 460, 19, 461, 18, 463, 10, 1, 3, 467, 8, 474, 3, 24001, 4, 475, 6, 474, 6, 469, 11, 468, 12, 467, 13, 466, 14, 466, 14, 466, 14, 465, 18, 463, 18, 462, 19, 461, 20, 461, 19, 462, 18, 463, 17, 462, 18, 461, 20, 460, 21, 459, 22, 457, 23, 458, 22, 458, 22, 458, 23, 458, 23, 458, 23, 459, 3, 4, 14, 467, 13, 467, 13, 467, 13, 468, 12, 467, 13, 467, 12, 468, 12, 467, 13, 468, 12, 468, 12, 468, 11, 470, 10, 471, 8, 474, 3, 22057, 7, 1, 6, 465, 16, 463, 18, 461, 20, 460, 20, 460, 20, 459, 21, 460, 20, 460, 20, 460, 19, 462, 18, 463, 17, 462, 18, 461, 19, 461, 18, 462, 18, 461, 18, 462, 15, 465, 14, 465, 15, 465, 14, 467, 13, 467, 13, 467, 13, 4, 6, 452, 18, 3, 8, 450, 19, 2, 10, 448, 20, 1, 12, 446, 21, 1, 12, 446, 21, 1, 12, 446, 34, 445, 35, 445, 35, 446, 35, 445, 36, 444, 37, 444, 37, 444, 36, 445, 35, 445, 35, 444, 36, 445, 12, 4, 19, 445, 12, 4, 18, 446, 11, 5, 18, 447, 10, 6, 16, 449, 8, 8, 12, 454, 3, 13, 4, 5967)
  // annId 900100218891
  val _rle2 = Array[Int](636, 27, 6, 31, 416, 30, 2, 35, 413, 68, 412, 70, 410, 71, 409, 72, 408, 73, 407, 74, 406, 74, 406, 75, 405, 75, 406, 75, 405, 75, 406, 74, 406, 74, 407, 74, 407, 73, 407, 74, 406, 74, 406, 74, 406, 74, 406, 74, 406, 74, 406, 74, 407, 73, 407, 73, 407, 73, 406, 74, 406, 74, 406, 74, 406, 74, 406, 75, 405, 75, 405, 75, 405, 75, 405, 75, 405, 76, 405, 75, 405, 76, 404, 76, 404, 79, 400, 83, 397, 84, 396, 86, 394, 87, 393, 88, 392, 89, 391, 90, 390, 90, 390, 91, 389, 91, 389, 92, 388, 92, 388, 92, 388, 92, 389, 91, 389, 91, 389, 91, 389, 91, 389, 91, 389, 91, 389, 91, 389, 91, 389, 90, 390, 90, 390, 90, 390, 89, 391, 98, 382, 101, 379, 102, 378, 104, 376, 105, 375, 106, 374, 108, 372, 111, 369, 112, 368, 115, 365, 116, 364, 118, 362, 125, 355, 128, 352, 129, 351, 131, 349, 132, 348, 133, 347, 138, 342, 141, 339, 142, 338, 144, 336, 145, 336, 145, 335, 146, 335, 146, 334, 61, 2, 83, 335, 60, 2, 84, 335, 58, 5, 82, 11, 14, 310, 26, 1, 30, 7, 82, 7, 20, 309, 23, 2, 29, 13, 77, 5, 25, 307, 20, 6, 27, 15, 75, 3, 28, 307, 18, 8, 23, 19, 74, 1, 32, 307, 14, 11, 19, 23, 107, 310, 6, 17, 14, 27, 107, 335, 9, 29, 108, 338, 3, 31, 109, 371, 119, 361, 122, 360, 123, 361, 10, 3, 107, 375, 107, 372, 109, 371, 110, 369, 112, 367, 114, 366, 115, 364, 117, 363, 118, 362, 118, 361, 120, 360, 120, 360, 121, 277, 8, 74, 121, 274, 12, 73, 121, 273, 13, 73, 122, 273, 12, 73, 122, 274, 11, 73, 122, 274, 11, 23, 1, 3, 2, 44, 123, 274, 10, 29, 1, 43, 123, 275, 9, 31, 1, 41, 123, 265, 1, 11, 7, 33, 1, 39, 124, 263, 2, 12, 6, 73, 124, 263, 3, 10, 7, 73, 124, 262, 4, 9, 14, 33, 1, 33, 124, 262, 5, 6, 22, 13, 17, 32, 123, 262, 36, 5, 22, 32, 123, 262, 38, 2, 24, 32, 122, 262, 64, 32, 122, 262, 65, 32, 122, 261, 65, 32, 122, 261, 65, 33, 122, 260, 65, 34, 121, 260, 65, 34, 2, 2, 117, 260, 65, 39, 117, 259, 65, 39, 117, 259, 65, 40, 116, 260, 64, 40, 116, 260, 64, 41, 116, 260, 63, 41, 116, 260, 63, 43, 115, 260, 61, 47, 112, 261, 60, 51, 108, 261, 60, 61, 99, 262, 11, 10, 24, 76, 97, 263, 7, 14, 19, 80, 97, 264, 5, 16, 15, 83, 97, 266, 2, 17, 12, 86, 98, 285, 8, 89, 98, 285, 6, 91, 99, 381, 99, 381, 99, 381, 100, 381, 100, 380, 100, 381, 100, 380, 100, 381, 100, 380, 100, 381, 99, 382, 99, 381, 99, 257, 10, 115, 98, 254, 23, 5, 4, 96, 98, 253, 42, 87, 98, 251, 45, 86, 98, 250, 48, 84, 98, 249, 50, 83, 98, 248, 52, 82, 98, 248, 53, 81, 98, 247, 55, 80, 98, 246, 56, 80, 98, 246, 57, 79, 98, 245, 58, 79, 98, 245, 59, 78, 98, 245, 59, 78, 97, 246, 59, 78, 97, 246, 59, 78, 96, 247, 59, 79, 95, 247, 59, 79, 95, 247, 59, 80, 93, 248, 59, 80, 92, 249, 59, 81, 91, 249, 59, 81, 90, 250, 58, 81, 90, 252, 53, 74, 100, 253, 49, 75, 102, 255, 44, 77, 103, 256, 42, 70, 110, 259, 40, 68, 85, 1, 26, 261, 37, 68, 86, 4, 22, 263, 21, 8, 5, 69, 88, 7, 16, 268, 17, 82, 89, 12, 8, 273, 13, 84, 90, 294, 12, 84, 90, 296, 10, 84, 89, 301, 5, 85, 88, 300, 7, 85, 88, 298, 9, 85, 87, 297, 11, 85, 86, 297, 12, 84, 86, 297, 13, 84, 85, 297, 14, 84, 84, 297, 15, 83, 83, 298, 17, 82, 82, 298, 18, 81, 81, 300, 18, 80, 82, 299, 19, 74, 88, 298, 22, 68, 92, 298, 24, 63, 95, 297, 27, 59, 97, 297, 34, 50, 98, 298, 39, 45, 98, 297, 42, 43, 97, 298, 43, 42, 97, 298, 44, 41, 97, 298, 45, 40, 96, 299, 52, 33, 95, 300, 57, 28, 95, 300, 58, 27, 94, 301, 58, 27, 93, 302, 59, 26, 92, 303, 59, 26, 91, 304, 59, 26, 90, 305, 60, 25, 88, 307, 60, 25, 87, 308, 60, 26, 84, 310, 60, 26, 81, 313, 60, 27, 76, 317, 60, 25, 69, 327, 59, 24, 69, 328, 59, 23, 68, 331, 58, 22, 68, 332, 58, 22, 66, 335, 57, 21, 64, 338, 57, 20, 61, 343, 56, 20, 55, 350, 55, 19, 56, 350, 55, 19, 56, 351, 53, 20, 56, 352, 52, 20, 56, 353, 41, 3, 6, 21, 56, 354, 36, 10, 3, 21, 56, 356, 34, 34, 56, 357, 20, 8, 4, 44, 47, 358, 19, 56, 47, 357, 20, 56, 47, 356, 21, 56, 47, 355, 22, 56, 46, 355, 23, 56, 46, 354, 24, 56, 46, 354, 25, 10, 2, 38, 50, 354, 26, 8, 4, 36, 51, 354, 39, 15, 4, 18, 49, 355, 39, 10, 15, 13, 48, 354, 66, 12, 47, 355, 66, 14, 43, 357, 67, 14, 20, 3, 18, 357, 68, 15, 18, 6, 14, 359, 68, 17, 14, 12, 6, 363, 69, 20, 6, 385, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 69, 411, 68, 412, 68, 412, 67, 413, 67, 413, 67, 413, 66, 414, 65, 415, 65, 415, 64, 416, 63, 417, 43, 3, 16, 418, 42, 6, 13, 419, 40, 10, 10, 421, 38, 12, 7, 423, 36, 14, 6, 425, 34, 15, 4, 426, 34, 446, 33, 446, 34, 446, 33, 447, 31, 448, 26, 454, 21, 459, 20, 460, 20, 460, 20, 460, 20, 460, 21, 459, 21, 459, 21, 459, 21, 459, 21, 459, 21, 459, 21, 459, 21, 459, 25, 455, 29, 451, 31, 450, 33, 447, 35, 446, 31, 1, 4, 444, 29, 7, 5, 440, 27, 9, 4, 440, 26, 11, 2, 442, 24, 457, 22, 458, 21, 460, 20, 461, 19, 462, 18, 463, 19, 463, 19, 462, 20, 462, 22, 461, 16, 468, 8, 2410, 3, 474, 6, 12, 1, 459, 8, 9, 7, 454, 16, 1, 10, 451, 31, 447, 34, 444, 37, 4, 12, 426, 57, 422, 61, 418, 63, 416, 66, 413, 68, 411, 70, 410, 71, 408, 73, 406, 75, 405, 76, 403, 78, 402, 78, 402, 79, 400, 80, 400, 81, 399, 81, 399, 81, 399, 82, 398, 82, 398, 82, 398, 82, 398, 82, 398, 82, 398, 82, 398, 82, 396, 84, 394, 86, 392, 88, 390, 90, 389, 91, 388, 92, 387, 93, 386, 94, 385, 94, 385, 95, 385, 94, 385, 95, 384, 96, 384, 97, 382, 98, 382, 98, 382, 99, 5, 13, 362, 100, 2, 19, 359, 124, 356, 125, 355, 127, 353, 128, 352, 129, 351, 130, 350, 131, 349, 132, 348, 133, 347, 134, 346, 134, 346, 135, 345, 135, 345, 136, 344, 136, 345, 135, 345, 136, 345, 46, 1, 88, 345, 135, 346, 134, 346, 134, 347, 133, 348, 132, 348, 132, 347, 133, 347, 133, 346, 134, 346, 134, 346, 134, 345, 135, 345, 135, 345, 134, 346, 134, 346, 133, 347, 133, 347, 133, 347, 132, 348, 131, 349, 131, 349, 130, 350, 129, 351, 128, 352, 127, 353, 124, 357, 119, 361, 114, 367, 110, 370, 108, 372, 102, 1, 2, 375, 100, 380, 97, 384, 92, 388, 92, 389, 91, 389, 92, 389, 91, 389, 91, 389, 91, 389, 91, 389, 91, 389, 91, 390, 90, 390, 90, 391, 89, 391, 89, 392, 88, 392, 88, 393, 87, 393, 87, 394, 85, 396, 84, 396, 83, 398, 82, 399, 81, 399, 80, 401, 78, 401, 79, 400, 79, 401, 78, 401, 78, 401, 79, 401, 80, 399, 88, 392, 89, 391, 89, 45, 3, 342, 90, 44, 11, 17, 2, 316, 91, 41, 20, 9, 4, 315, 94, 36, 29, 2, 4, 315, 98, 31, 36, 315, 102, 25, 38, 315, 105, 21, 39, 315, 109, 12, 44, 315, 164, 316, 162, 318, 161, 318, 152, 328, 137, 343, 133, 347, 130, 350, 129, 351, 129, 351, 128, 353, 126, 355, 125, 356, 123, 357, 121, 360, 119, 361, 117, 364, 114, 366, 113, 367, 112, 368, 112, 368, 111, 369, 111, 369, 111, 369, 110, 370, 110, 370, 110, 370, 110, 370, 109, 371, 109, 371, 109, 372, 108, 372, 108, 373, 107, 373, 108, 373, 107, 373, 48, 2, 58, 373, 47, 5, 56, 373, 46, 11, 51, 372, 45, 12, 51, 373, 43, 13, 52, 373, 42, 13, 19, 1, 31, 375, 40, 14, 17, 5, 29, 376, 38, 16, 14, 8, 27, 378, 36, 17, 12, 11, 26, 379, 34, 19, 10, 14, 23, 381, 30, 22, 8, 17, 24, 381, 22, 32, 3, 17, 28, 379, 18, 54, 33, 27, 2, 348, 14, 56, 33, 18, 14, 348, 4, 62, 35, 7, 26, 412, 70, 409, 72, 408, 73, 331, 6, 69, 75, 329, 8, 68, 76, 327, 10, 66, 78, 326, 11, 65, 79, 324, 12, 64, 81, 321, 14, 64, 81, 319, 15, 64, 83, 316, 17, 64, 83, 119, 15, 181, 18, 63, 85, 118, 18, 177, 19, 63, 85, 119, 20, 173, 20, 62, 86, 119, 23, 169, 21, 62, 87, 118, 24, 167, 22, 61, 88, 118, 26, 164, 24, 60, 88, 118, 26, 164, 25, 16, 2, 40, 89, 118, 26, 163, 27, 14, 3, 35, 94, 118, 26, 162, 31, 10, 5, 34, 94, 118, 26, 162, 34, 7, 5, 34, 94, 118, 26, 161, 48, 33, 97, 115, 26, 161, 48, 33, 100, 112, 26, 161, 48, 32, 104, 79, 6, 24, 26, 160, 50, 31, 105, 77, 21, 10, 26, 160, 50, 31, 107, 74, 27, 5, 26, 160, 50, 30, 109, 70, 61, 160, 50, 30, 110, 65, 65, 160, 50, 30, 111, 61, 68, 160, 50, 30, 112, 57, 71, 160, 50, 21, 122, 53, 74, 160, 50, 15, 129, 50, 76, 160, 50, 12, 133, 47, 78, 160, 50, 10, 135, 45, 80, 160, 50, 8, 138, 43, 81, 160, 50, 6, 140, 42, 82, 160, 35, 3, 12, 5, 142, 40, 83, 160, 22, 3, 7, 10, 7, 4, 144, 39, 84, 160, 14, 1, 6, 24, 1, 6, 145, 38, 85, 161, 10, 7, 1, 33, 146, 36, 86, 161, 11, 39, 147, 35, 87, 162, 11, 37, 148, 34, 88, 162, 11, 36, 149, 34, 88, 163, 10, 34, 151, 33, 89, 163, 10, 32, 153, 32, 90, 164, 9, 30, 155, 32, 90, 165, 8, 28, 157, 31, 91, 165, 9, 25, 159, 31, 91, 166, 9, 4, 6, 12, 161, 31, 91, 167, 22, 4, 165, 30, 92, 168, 190, 30, 92, 169, 189, 30, 92, 171, 187, 30, 92, 172, 186, 30, 92, 174, 183, 31, 92, 177, 180, 31, 92, 181, 8, 1, 166, 32, 92, 190, 166, 32, 92, 190, 166, 32, 92, 190, 165, 33, 92, 190, 164, 34, 92, 190, 164, 34, 92, 190, 163, 35, 92, 190, 162, 36, 92, 190, 161, 37, 92, 190, 160, 38, 92, 190, 159, 40, 91, 190, 157, 42, 91, 190, 156, 44, 90, 190, 154, 46, 90, 190, 153, 48, 89, 191, 152, 48, 89, 191, 152, 49, 88, 192, 152, 49, 87, 192, 152, 41, 95, 193, 151, 24, 112, 193, 151, 21, 115, 194, 150, 19, 117, 195, 149, 17, 119, 195, 149, 15, 121, 195, 149, 14, 122, 195, 149, 13, 123, 195, 149, 12, 124, 195, 149, 11, 125, 195, 149, 10, 126, 195, 149, 9, 127, 195, 149, 9, 127, 195, 149, 8, 128, 195, 148, 8, 129, 195, 148, 8, 129, 195, 147, 8, 130, 195, 147, 8, 130, 195, 147, 8, 130, 195, 146, 8, 131, 195, 145, 9, 131, 195, 145, 9, 131, 195, 144, 10, 131, 196, 142, 11, 131, 196, 141, 12, 131, 197, 139, 13, 131, 197, 138, 14, 130, 199, 135, 16, 129, 200, 134, 17, 128, 202, 131, 19, 126, 205, 110, 1, 16, 22, 125, 6)
  // scalastyle:on

  val rle1 = RLEMasks(_rle1, 480, 640)
  val rle2 = RLEMasks(_rle2, 480, 640)

  "rleToOneBbox" should "run well" in {
    rle1.bbox should be (142f, 245f, 486f + 141, 111f + 244)
    rle2.bbox should be (1f, 155f, 639f, 325f + 154)
  }

  "bboxIOU" should "run well" in {
    MaskUtils.bboxIOU((142f, 245f, 486f + 141, 111f + 244), (1f, 155f, 639f, 325f + 154),
      false) should be (0.25976165f)
  }

  "rleArea" should "run well" in {
    MaskUtils.rleArea(rle1) should be (5976)
    MaskUtils.rleArea(rle2) should be (77429)
  }

  "rleIOU" should "run well" in {
    MaskUtils.rleIOU(rle1, rle2, true) should be (0.58199465f)
    MaskUtils.rleIOU(rle1, rle2, false) should be (0.04351471f +- 0.0000001f)
    MaskUtils.rleIOU(rle2, rle1, false) should be (0.04351471f +- 0.0000001f)
    MaskUtils.rleIOU(rle2, rle1, true) should be (0.04491857f)
  }
}

