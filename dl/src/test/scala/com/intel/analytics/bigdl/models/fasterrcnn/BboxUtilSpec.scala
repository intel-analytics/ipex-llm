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

package com.intel.analytics.bigdl.models.fasterrcnn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class BboxUtilSpec extends FlatSpec with Matchers {
  "bboxTransformInv" should "work properly" in {

    val boxes = Tensor(Storage(Array(0.54340494, 0.2783694, 0.4245176, 0.84477615,
      0.0047188564, 0.12156912, 0.67074907, 0.82585275,
      0.13670659, 0.5750933, 0.89132196, 0.20920213,
      0.18532822, 0.10837689, 0.21969749, 0.9786238,
      0.8116832, 0.17194101, 0.81622475, 0.27407375).map(x => x.toFloat))).resize(5, 4)



    val deltas = Tensor(Storage(Array(
      0.4317042, 0.9400298, 0.81764936, 0.33611196,
      0.7956625, 0.015254972, 0.5988434, 0.6038045,
      0.98092085, 0.05994199, 0.89054596, 0.5769015,
      0.21002658, 0.5446849, 0.76911515, 0.25069523,
      0.35950786, 0.59885895, 0.3547956, 0.3401902)
      .map(x => x.toFloat)))
    deltas.resize(5, 4)

    val expectedResults = Tensor(Storage(Array(
      0.36640674, 1.437952, 2.3622758, 3.6301315,
      0.64723384, -0.55891246, 3.6794298, 2.558332,
      0.5976285, 0.36563802, 4.872678, 1.494677,
      -0.19625212, 0.8606382, 2.0357678, 3.263753,
      0.958912, 0.6086628, 2.391277, 2.157396)
      .map(x => x.toFloat)))
    expectedResults.resize(5, 4)

    val res = BboxUtil.bboxTransformInv(boxes, deltas)
    res should be(expectedResults)
  }

  "bboxTransformInv with deltas 5 * 8" should "work properly" in {

    val boxes = Tensor(Storage(Array(0.54340494, 0.0047188564, 0.13670659, 0.18532822, 0.8116832,
      0.2783694, 0.12156912, 0.5750933, 0.10837689, 0.17194101,
      0.4245176, 0.67074907, 0.89132196, 0.21969749, 0.81622475,
      0.84477615, 0.82585275, 0.20920213, 0.9786238, 0.27407375).map(x => x.toFloat)))
      .resize(4, 5).t().contiguous()



    val deltas = Tensor(Storage(Array(
      0.431704183663, 0.795662508473, 0.980920857012, 0.210026577673, 0.359507843937,
      0.940029819622, 0.0152549712463, 0.059941988818, 0.544684878179, 0.598858945876,
      0.817649378777, 0.598843376928, 0.890545944729, 0.769115171106, 0.354795611657,
      0.336111950121, 0.603804539043, 0.5769014994, 0.250695229138, 0.340190215371,
      0.175410453742, 0.105147685412, 0.742479689098, 0.285895690407, 0.178080989506,
      0.37283204629, 0.381943444943, 0.630183936475, 0.852395087841, 0.237694208624,
      0.00568850735257, 0.0364760565926, 0.581842192399, 0.975006493607, 0.0448622824608,
      0.252426353445, 0.890411563442, 0.0204391320269, 0.884853293491, 0.505431429636)
      .map(x => x.toFloat))).resize(8, 5).t().contiguous()

    val expectedResults = Tensor(Storage(Array(
      0.36640674522, 0.64723382891, 0.597628476893, -0.196252115502, 0.958912029942,
      1.43795206519, -0.558912463015, 0.365638008002, 0.860638198136, 0.608662780166,
      2.36227582099, 3.67942969367, 4.8726776815, 2.03576790462, 2.39127703713,
      3.63013155633, 2.55833193458, 1.49467692786, 3.26375288055, 2.15739607458,
      0.69544806191, 0.148952007861, 0.746986104878, -0.372917280555, 0.967526811475,
      0.637483321098, -0.451279106784, 0.968151508352, 0.372232468489, 0.0714749295084,
      1.58158720973, 1.87687437777, 3.88657497995, 2.36938642765, 2.01816061047,
      2.6536754621, 3.70058090758, 1.61535429345, 4.90314673809, 1.89848096643)
      .map(x => x.toFloat))).resize(8, 5).t().contiguous()

    val res = BboxUtil.bboxTransformInv(boxes, deltas)
    res should be(expectedResults)
  }

  "clipBoxes" should "work properly" in {
    val boxes = Tensor(Storage(Array(
      43.170418, 94.00298, 81.76494, 33.611195,
      79.56625, 1.5254971, 59.88434, 60.380455,
      98.09209, 5.994199, 89.054596, 57.69015,
      21.002657, 35.468487, 76.911514, 55.069523,
      35.950783, 34.885895, 35.47956, 70.01902)
      .map(x => x.toFloat))).resize(5, 4)

    val scores = Tensor(Storage(Array(0.999516, 0.9487129, 0.9859998, 0.9985473, 0.9780578)
      .map(x => x.toFloat)))

    val expectedResults = Tensor(Storage(Array(
      43.170418, 59.0, 49.0, 33.611195,
      49.0, 1.5254971, 49.0, 59.0,
      49.0, 5.994199, 49.0, 57.69015,
      21.002657, 35.468487, 49.0, 55.069523,
      35.950783, 34.885895, 35.47956, 59.0)
      .map(x => x.toFloat))).resize(5, 4)

    val expectedScores = Tensor(Storage(Array(0, 0, 0, 0.9985473, 0).map(x => x.toFloat)))

    BboxUtil.clipBoxes(boxes, 60, 50, 4, 5, scores)
    scores should be(expectedScores)
    boxes should be(expectedResults)
  }

  "clipBoxes with size 5 * 8" should "work properly" in {
    val boxes = Tensor(Storage(Array(
      43.1704183663, 79.5662508473, 98.0920857012, 21.0026577673, 35.9507843937,
      94.0029819622, 1.52549712463, 5.9941988818, 54.4684878179, 59.8858945876,
      81.7649378777, 59.8843376928, 89.0545944729, 76.9115171106, 35.4795611657,
      33.6111950121, 60.3804539043, 57.69014994, 25.0695229138, 34.0190215371,
      17.5410453742, 10.5147685412, 74.2479689098, 28.5895690407, 17.8080989506,
      37.283204629, 38.1943444943, 63.0183936475, 85.2395087841, 23.7694208624,
      0.568850735257, 3.64760565926, 58.1842192399, 97.5006493607, 4.48622824608,
      25.2426353445, 89.0411563442, 2.04391320269, 88.4853293491, 50.5431429636)
      .map(x => x.toFloat))).resize(8, 5).t().contiguous()

    val expectedResults = Tensor(Storage(Array(
      19.0, 19.0, 19.0, 19.0, 19.0,
      9.0, 1.52549712463, 5.9941988818, 9.0, 9.0,
      19.0, 19.0, 19.0, 19.0, 19.0,
      9.0, 9.0, 9.0, 9.0, 9.0,
      17.5410453742, 10.5147685412, 19.0, 19.0, 17.8080989506,
      9.0, 9.0, 9.0, 9.0, 9.0,
      0.568850735257, 3.64760565926, 19.0, 19.0, 4.48622824608,
      9.0, 9.0, 2.04391320269, 9.0, 9.0)
      .map(x => x.toFloat))).resize(8, 5).t().contiguous()

    BboxUtil.clipBoxes(boxes, 10, 20)
    boxes.map(expectedResults, (a, b) => {
      assert(Math.abs(a - b) < 1e-6)
      a
    })
  }
}
