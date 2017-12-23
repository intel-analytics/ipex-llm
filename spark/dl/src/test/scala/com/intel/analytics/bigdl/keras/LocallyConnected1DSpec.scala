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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn.LocallyConnected1D
import com.intel.analytics.bigdl.tensor.Tensor


class LocallyConnected1DSpec extends KerasBaseSpec {

  "LocallyConnected1D" should "be ok" in {
    ifskipTest()
    val kerasCode =
      /*
      """
        |input_tensor = Input(shape=[100,10])
        |input = input = np.random.rand(1,100,10)
        |output_tensor=LocallyConnected1D(2,3,subsample_length=3,input_shape=(100,10))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
      */
//    val locallyConnected1d = LocallyConnected1D[Float](100, 10, 2, 3, 3)

      """
        |input_tensor = Input(shape=[6,2])
        |input = np.array([[[1,2], [2,3], [3,4],[4,5],[5,6],[6,7]]])
        |weights = [np.array([[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]],[[9],[10]]]),np.zeros([5,1])]
        |output_tensor = LocallyConnected1D(2,3,subsample_length=1,input_shape=(6,2))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val locallyConnected1d = LocallyConnected1D[Float](6, 2, outputFrameSize = 2,
      kernelW = 3, strideW = 1)


    val wc = (data: Array[Tensor[Float]]) => {

      val out = new Array[Tensor[Float]](data.length)
      val d1l: Int = data(0).size(1)
      val d2l: Int = data(0).size(2)
      val d3l: Int = data(0).size(3)

      out(0) = Tensor(d1l, d3l, d2l)

      val page: Int = d2l * d3l
      for (i <- 0 to d1l * d2l * d3l - 1) {
        val d1 = i / page + 1
        val d2 = (i % page) / (d3l) + 1
        val d3 = (i % page) % d3l + 1
        val v = data(0).valueAt(d1, d2, d3)
        out(0).setValue(d1, d3, d2, v)
      }

      if (data.length > 1) {
        out(1) = data(1)
      }
      out
    }

    checkOutputAndGrad(locallyConnected1d, kerasCode, wc)

  }


  "LocallyConnected1D reshape" should "be ok" in {

    val locallyConnected1d = LocallyConnected1D[Float](6, 2, outputFrameSize = 2,
      kernelW = 3, strideW = 1)

    val d1l = 2
    val d2l = 3

    val _input = Tensor[Float](d1l, d2l)
    val _output = Tensor[Float](1, d1l, d2l)

    for (i <- 0 to d1l * d2l - 1) {
      val d1 = i / d2l + 1
      val d2 = i % d2l + 1
      _input.setValue(d1, d2, i)
      _output.setValue(1, d1, d2, i)
    }


    val input = locallyConnected1d.reshapeInput(_input)
    val output = locallyConnected1d.reshapeOutput(_input, _output)

    val size = Array(1, d1l, d2l)

    input.size() should be(size)
    _input.storage().map(x => x.toInt).toArray should be(input.storage().map(x => x.toInt).toArray)

    output.size() should be(Array(d1l, d2l))
    _output.storage().map(x => x.toInt).toArray should be(
      output.storage().map(x => x.toInt).toArray)
  }

}
