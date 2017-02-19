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

class BboxSpec extends FlatSpec with Matchers {
  val boxTool = new Bbox
  "bboxVote" should "work properly" in {
    val detsNMS = Tensor(Storage(Array(124.67757, 198.44446, 499.0, 319.46954, 0.44293448,
      342.84973, 159.25734, 499.0, 306.46826, 0.105595924,
      71.944336, 42.300797, 435.08072, 323.51843, 0.095248096).map(x => x.toFloat)))
    detsNMS.resize(Array(3, 5))
    val detsAll = Tensor(Storage(Array(19.352219, 163.53407, 499.0, 318.2702, 0.1923532,
      124.67757, 198.44446, 499.0, 319.46954, 0.44293448,
      0.0, 191.88805, 499.0, 321.89288, 0.118584715,
      0.0, 99.202065, 493.6775, 321.6583, 0.1829617,
      0.0, 190.09538, 391.3026, 328.06378, 0.053920552,
      239.84499, 103.70652, 499.0, 321.04962, 0.100431755,
      100.49286, 91.12562, 499.0, 323.65686, 0.18068084,
      178.0736, 175.94618, 499.0, 321.1881, 0.37770998,
      0.0, 118.12887, 386.87604, 319.87268, 0.084006935,
      175.99504, 35.884323, 498.94446, 318.77704, 0.07011555,
      121.67645, 182.41452, 478.1596, 320.16412, 0.38305077,
      0.0, 30.538345, 492.15747, 312.7992, 0.07436083,
      77.29045, 125.990456, 478.4983, 320.66547, 0.10816171,
      123.633224, 205.55194, 447.4248, 319.41675, 0.11607359,
      178.96523, 206.04062, 499.0, 317.30673, 0.07201823,
      155.32773, 207.17589, 499.0, 316.96683, 0.07939793,
      287.53674, 118.663925, 499.0, 309.00146, 0.10261241,
      57.928955, 130.33197, 408.22736, 317.33112, 0.25868088,
      163.74406, 111.216034, 493.4549, 318.0517, 0.07715336,
      92.417786, 190.0836, 426.7011, 319.14508, 0.053326167,
      342.84973, 159.25734, 499.0, 306.46826, 0.105595924,
      71.944336, 42.300797, 435.08072, 323.51843, 0.095248096).map(x => x.toFloat)))
    detsAll.resize(Array(22, 5))
    val target = boxTool.bboxVote(detsNMS.select(2, 5),
      detsNMS.narrow(2, 1, 4),
      detsAll.select(2, 5),
      detsAll.narrow(2, 1, 4))
    val expectedBoxes = Tensor(Storage(Array(117.93, 181.086, 488.87, 319.773,
      315.59, 139.251, 499.0, 307.717,
      57.5732, 95.5176, 458.503, 320.046).map(x => x.toFloat)))
    val expectedScores = Tensor(Storage(Array(0.442934, 0.105596, 0.0952481).map(x => x.toFloat)))
    expectedBoxes.resize(3, 4)
    target.bboxes should be(expectedBoxes)
    target.classes should be(expectedScores)
  }


  "bboxTransformInv" should "work properly" in {

    val boxes = Tensor(Storage(Array(0.54340494, 0.0047188564, 0.13670659, 0.18532822, 0.8116832,
      0.2783694, 0.12156912, 0.5750933, 0.10837689, 0.17194101,
      0.4245176, 0.67074907, 0.89132196, 0.21969749, 0.81622475,
      0.84477615, 0.82585275, 0.20920213, 0.9786238, 0.27407375).map(x => x.toFloat)))
    boxes.resize(4, 5)



    val deltas = Tensor(Storage(Array(
      0.431704183663, 0.795662508473, 0.980920857012, 0.210026577673, 0.359507843937,
      0.940029819622, 0.0152549712463, 0.059941988818, 0.544684878179, 0.598858945876,
      0.817649378777, 0.598843376928, 0.890545944729, 0.769115171106, 0.354795611657,
      0.336111950121, 0.603804539043, 0.5769014994, 0.250695229138, 0.340190215371,
      0.175410453742, 0.105147685412, 0.742479689098, 0.285895690407, 0.178080989506,
      0.37283204629, 0.381943444943, 0.630183936475, 0.852395087841, 0.237694208624,
      0.00568850735257, 0.0364760565926, 0.581842192399, 0.975006493607, 0.0448622824608,
      0.252426353445, 0.890411563442, 0.0204391320269, 0.884853293491, 0.505431429636)
      .map(x => x.toFloat)))
    deltas.resize(8, 5)

    val expectedResults = Tensor(Storage(Array(
      0.36640674522, 0.64723382891, 0.597628476893, -0.196252115502, 0.958912029942,
      1.43795206519, -0.558912463015, 0.365638008002, 0.860638198136, 0.608662780166,
      2.36227582099, 3.67942969367, 4.8726776815, 2.03576790462, 2.39127703713,
      3.63013155633, 2.55833193458, 1.49467692786, 3.26375288055, 2.15739607458,
      0.69544806191, 0.148952007861, 0.746986104878, -0.372917280555, 0.967526811475,
      0.637483321098, -0.451279106784, 0.968151508352, 0.372232468489, 0.0714749295084,
      1.58158720973, 1.87687437777, 3.88657497995, 2.36938642765, 2.01816061047,
      2.6536754621, 3.70058090758, 1.61535429345, 4.90314673809, 1.89848096643)
      .map(x => x.toFloat)))
    expectedResults.resize(8, 5)

    val res = boxTool.bboxTransformInv(boxes, deltas)
    res should be(expectedResults)
  }

  "clipBoxes" should "work properly" in {
    val boxes = Tensor(Storage(Array(
      43.1704183663, 79.5662508473, 98.0920857012, 21.0026577673, 35.9507843937,
      94.0029819622, 1.52549712463, 5.9941988818, 54.4684878179, 59.8858945876,
      81.7649378777, 59.8843376928, 89.0545944729, 76.9115171106, 35.4795611657,
      33.6111950121, 60.3804539043, 57.69014994, 25.0695229138, 34.0190215371,
      17.5410453742, 10.5147685412, 74.2479689098, 28.5895690407, 17.8080989506,
      37.283204629, 38.1943444943, 63.0183936475, 85.2395087841, 23.7694208624,
      0.568850735257, 3.64760565926, 58.1842192399, 97.5006493607, 4.48622824608,
      25.2426353445, 89.0411563442, 2.04391320269, 88.4853293491, 50.5431429636)
      .map(x => x.toFloat)))
    boxes.resize(8, 5)

    val expectedResults = Tensor(Storage(Array(
      19.0, 19.0, 19.0, 19.0, 19.0,
      9.0, 1.52549712463, 5.9941988818, 9.0, 9.0,
      19.0, 19.0, 19.0, 19.0, 19.0,
      9.0, 9.0, 9.0, 9.0, 9.0,
      17.5410453742, 10.5147685412, 19.0, 19.0, 17.8080989506,
      9.0, 9.0, 9.0, 9.0, 9.0,
      0.568850735257, 3.64760565926, 19.0, 19.0, 4.48622824608,
      9.0, 9.0, 2.04391320269, 9.0, 9.0)
      .map(x => x.toFloat)))
    boxes.resize(8, 5)

    boxTool.clipBoxes(boxes, 10, 20)
    boxes.map(expectedResults, (a, b) => {
      assert(Math.abs(a - b) < 1e-6)
      a
    })
  }
}
