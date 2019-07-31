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

import com.intel.analytics.bigdl.nn.mkldnn.{Equivalent, Tools}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.sun.deploy.ref.AppModel.ExtensionResource
import org.dmg.pmml.True
import org.scalatest.{FlatSpec, Matchers}

class PriorBoxSpec extends FlatSpec with Matchers {
  "Priorbox" should "work" in {
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    val minSizes = Array(460.8f)
    val maxSizes = Array(537.6f)
    val aspectRatios = Array(2f)
//    val param = ComponetParam(256, 4, minSizes = Array(460.8f),
//      maxSizes = Array(537.6f), aspectRatios = Array(2), isFlip, isClip, variances, 512)
    val layer = PriorBox[Float](minSizes = minSizes, maxSizes = maxSizes,
      _aspectRatios = aspectRatios, isFlip = isFlip, isClip = isClip,
      variances = variances, step = 0, offset = 0.5f, imgH = 512, imgW = 512)
    val input = Tensor[Float](8, 256, 1, 1)

    val out = layer.forward(input)

    val expectedStr = "0.0507812\n0.0507812\n0.949219\n0.949219\n0.0146376\n" +
      "0.0146376\n0.985362\n0.985362\n-0.135291\n0.182354\n1.13529\n0.817646\n" +
      "0.182354\n-0.135291\n0.817646\n1.13529\n0.1\n0.1\n0.2\n0.2\n0.1\n0.1\n0.2\n" +
      "0.2\n0.1\n0.1\n0.2\n0.2\n0.1\n0.1\n0.2\n0.2"

    val expected = Tensor(Storage(expectedStr.split("\n").map(_.toFloat))).resize(1, 2, 16)

    out.map(expected, (a, b) => {
      assert((a - b).abs < 1e-5);
      a
    })
  }

  "Priorbox 111 " should "work" in {
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    val minSizes = Array(460.8f)
    val maxSizes = Array(537.6f)
    val aspectRatios = Array(2f)
    //    val param = ComponetParam(256, 4, minSizes = Array(460.8f),
    //      maxSizes = Array(537.6f), aspectRatios = Array(2), isFlip, isClip, variances, 512)
    val layer = PriorBox[Float](minSizes = minSizes, maxSizes = maxSizes,
      _aspectRatios = aspectRatios, isFlip = isFlip, isClip = isClip,
      variances = variances, step = 0, offset = 0.5f, imgH = 512, imgW = 512)
    val input = Tensor[Float](8, 256, 1, 1)

    val out = layer.forward(input)

    val expectedStr = "0.0507812\n0.0507812\n0.949219\n0.949219\n0.0146376\n" +
      "0.0146376\n0.985362\n0.985362\n-0.135291\n0.182354\n1.13529\n0.817646\n" +
      "0.182354\n-0.135291\n0.817646\n1.13529\n0.1\n0.1\n0.2\n0.2\n0.1\n0.1\n0.2\n" +
      "0.2\n0.1\n0.1\n0.2\n0.2\n0.1\n0.1\n0.2\n0.2"

    val expected = Tensor(Storage(expectedStr.split("\n").map(_.toFloat))).resize(1, 2, 16)

    out.map(expected, (a, b) => {
      assert((a - b).abs < 1e-5);
      a
    })
  }

  "Anchor" should "be ok" in {
    val ratios = Array[Float](0.5f, 1.0f, 2.0f)
    val scales = Array[Float](32f, 64f, 128f, 256f, 512f)
    val strides = Array[Float](4f, 8f, 16f, 32f, 64f)
    val scalesForStride = Array[Float](8f)
    val cls = Anchor(ratios, scalesForStride)
    val res1 = cls.generateAnchors(34, 25, 4f).clone()
    val res2 = cls.generateAnchors(136, 100, 8f).clone()
    val res3 = cls.generateAnchors(68, 50, 16f).clone()
    val res4 = cls.generateAnchors(17, 13, 32f).clone()
    val res5 = cls.generateAnchors(272, 200, 64f).clone()

    require(res1.size(1) == 2550 && res1.size(2) == 4)
    require(res2.size(1) == 40800 && res2.size(2) == 4)
    require(res2.size(1) == 10200 && res2.size(2) == 4)
    require(res2.size(1) == 663 && res2.size(2) == 4)
    require(res2.size(1) == 163200 && res2.size(2) == 4)

    println("done")

  }

  "Proposal" should "be ok" in {
    val preNmsTopN: Int = 1000
    val postNmsTopN: Int = 1000
    val ratios: Array[Float] = Array[Float](0.1f, 0.2f, 0.3f)
    val scales: Array[Float] = Array[Float](4, 5, 6)
    val rpnPreNmsTopNTrain: Int = 1000
    val rpnPostNmsTopNTrain: Int = 2000


    val proposal = new Proposal(preNmsTopN, postNmsTopN,
      ratios, scales, rpnPreNmsTopNTrain, rpnPostNmsTopNTrain, min_size = 0)

    val score = Tensor[Float](1, 18, 20, 30).randn()
    val boxes = Tensor[Float](1, 36, 20, 30).randn()
    val imInfo = Tensor[Float](T(300, 300, 1, 1)).resize(1, 4)
    val input = T(score, boxes, imInfo)

    proposal.forward(input)

    println("done")
  }

  "boxlist" should "be ok" in {
    val nms = new Nms()

    val scores = Tensor[Float](
      T(0.5144, 0.5111, 0.5096, 0.5089, 0.5072, 0.5070, 0.5066, 0.5066, 0.5063,
      0.5054, 0.5049, 0.5041, 0.5041, 0.5039, 0.5038, 0.5024, 0.5022, 0.5015,
      0.5015, 0.5003, 0.5002, 0.4998, 0.4997, 0.4997, 0.4996, 0.4996, 0.4994,
      0.4991, 0.4989, 0.4987, 0.4985, 0.4982, 0.4978, 0.4974, 0.4964, 0.4963,
      0.4962, 0.4961, 0.4961, 0.4959, 0.4958, 0.4952, 0.4944, 0.4943, 0.4941,
      0.4935, 0.4934, 0.4934, 0.4924, 0.4921, 0.4921, 0.4919, 0.4916, 0.4914,
      0.4914, 0.4910, 0.4903, 0.4900, 0.4896, 0.4893, 0.4890, 0.4886, 0.4821))

    val bbox = Tensor[Float](
      T(T( 3.5514,  0.0000, 33.7322, 19.0000),
        T( 0.0000,  0.0000, 17.2042, 19.0000),
        T(10.0755,  0.0000, 37.0000, 19.0000),
        T( 0.0000,  0.0000, 24.6860, 19.0000),
        T( 6.4633,  0.0000, 36.8832, 19.0000),
        T( 0.0000,  0.0000, 21.8702, 19.0000),
        T( 0.0000,  0.0000, 29.2260, 19.0000),
        T( 9.4366,  0.0000, 37.0000, 19.0000),
        T( 0.0000,  0.0000, 24.8995, 17.0870),
        T( 0.0000,  0.0000, 24.3542, 19.0000),
        T( 0.0000,  0.0000, 17.4157, 16.6371),
        T( 1.7347,  0.0000, 32.6897, 19.0000),
        T( 0.0000,  0.0000, 17.9271, 19.0000),
        T(14.6003,  0.0000, 37.0000, 19.0000),
        T(10.9857,  0.0000, 33.2061, 19.0000),
        T( 8.8203,  0.0000, 37.0000, 17.2047),
        T( 0.0000,  0.0000, 13.6237, 19.0000),
        T( 0.0000,  0.0000, 29.5733, 19.0000),
        T( 6.6627,  0.0000, 36.9200, 19.0000),
        T( 3.0336,  0.0000, 25.1643, 19.0000),
        T( 0.0000,  0.0000, 17.3206, 19.0000),
        T(10.4114,  0.0000, 32.8115, 19.0000),
        T( 0.0000,  0.0000, 13.0061, 19.0000),
        T( 0.0000,  0.0000, 20.5060, 15.6632),
        T( 5.7826,  0.0000, 37.0000, 17.1439),
        T( 2.1855,  0.0000, 24.8192, 19.0000),
        T( 0.0000,  0.0000, 21.5678, 19.0000),
        T( 0.0000,  0.0000, 16.7051, 19.0000),
        T(14.8947,  0.0000, 37.0000, 19.0000),
        T( 0.0000,  0.0000, 21.2348, 19.0000),
        T( 0.0000,  0.0000, 17.1296, 19.0000),
        T( 2.9560,  0.0000, 25.2678, 19.0000),
        T( 0.0000,  0.0000, 21.8768, 19.0000),
        T( 0.0000,  0.0000, 21.1777, 19.0000),
        T( 0.0000,  0.0000, 37.0000, 12.9645),
        T( 7.5263,  0.0000, 29.9973, 19.0000),
        T( 0.0000,  0.0000, 28.0624, 16.1052),
        T(14.3629,  0.0000, 37.0000, 19.0000),
        T(10.9161,  0.0000, 33.2830, 19.0000),
        T( 0.0000,  0.0000, 28.5583, 19.0000),
        T( 0.0000,  0.0000, 24.2979, 19.0000),
        T( 0.0000,  0.0000, 36.9026, 19.0000),
        T( 0.0000,  0.0000, 13.1744, 19.0000),
        T( 0.0000,  0.0000, 37.0000, 19.0000),
        T( 0.0000,  0.0000, 35.3306, 16.9414),
        T( 6.8610,  0.0000, 29.4768, 19.0000),
        T( 2.0814,  0.0000, 33.1002, 16.5095),
        T( 7.0757,  0.0000, 29.4296, 19.0000),
        T( 1.8985,  0.0000, 37.0000, 12.2108),
        T( 0.0000,  0.0000, 27.7941, 12.9797),
        T( 0.0000,  0.0000, 31.3791, 12.9397),
        T( 0.0000,  0.0000, 37.0000, 12.9046),
        T( 0.0000,  0.0000, 37.0000, 16.2695),
        T( 0.8843,  0.0000, 37.0000, 19.0000),
        T( 0.9825,  0.0000, 37.0000, 15.7812),
        T( 0.0000,  0.0000, 22.8811, 17.3911),
        T( 0.0000,  0.0000, 37.0000, 19.0000),
        T( 0.0000,  0.0000, 37.0000, 16.7101),
        T( 0.0000,  0.0000, 32.1652, 17.2077),
        T( 0.0000,  0.0000, 24.2108, 13.4736),
        T( 0.0000,  0.0000, 32.6558, 19.0000),
        T( 0.0000,  0.0000, 35.6184, 12.3386),
        T( 0.0000,  0.0000, 29.0507, 17.1434))
    )

    val indices = new Array[Int](10000)
    val res = nms.nms(scores, bbox, thresh = 0.7f, indices, sorted = true)

    // tensor([ 0,  1,  8, 13, 34])
    println("done")
  }

  "Porposal" should "be ok" in {
    val anchors = Tensor[Float](
      T(T(-22, -10,  25,  13),
        T(-14, -14,  17,  17),
        T(-10, -22,  13,  25),
        T(-18, -10,  29,  13),
        T(-10, -14,  21,  17),
        T( -6, -22,  17,  25),
        T(-14, -10,  33,  13),
        T( -6, -14,  25,  17),
        T( -2, -22,  21,  25),
        T(-10, -10,  37,  13),
        T( -2, -14,  29,  17),
        T(  2, -22,  25,  25),
        T( -6, -10,  41,  13),
        T(  2, -14,  33,  17),
        T(  6, -22,  29,  25),
        T( -2, -10,  45,  13),
        T(  6, -14,  37,  17),
        T( 10, -22,  33,  25),
        T(  2, -10,  49,  13),
        T( 10, -14,  41,  17),
        T( 14, -22,  37,  25),
        T(-22,  -6,  25,  17),
        T(-14, -10,  17,  21),
        T(-10, -18,  13,  29),
        T(-18,  -6,  29,  17),
        T(-10, -10,  21,  21),
        T( -6, -18,  17,  29),
        T(-14,  -6,  33,  17),
        T( -6, -10,  25,  21),
        T( -2, -18,  21,  29),
        T(-10,  -6,  37,  17),
        T( -2, -10,  29,  21),
        T(  2, -18,  25,  29),
        T( -6,  -6,  41,  17),
        T(  2, -10,  33,  21),
        T(  6, -18,  29,  29),
        T( -2,  -6,  45,  17),
        T(  6, -10,  37,  21),
        T( 10, -18,  33,  29),
        T(  2,  -6,  49,  17),
        T( 10, -10,  41,  21),
        T( 14, -18,  37,  29),
        T(-22,  -2,  25,  21),
        T(-14,  -6,  17,  25),
        T(-10, -14,  13,  33),
        T(-18,  -2,  29,  21),
        T(-10,  -6,  21,  25),
        T( -6, -14,  17,  33),
        T(-14,  -2,  33,  21),
        T( -6,  -6,  25,  25),
        T( -2, -14,  21,  33),
        T(-10,  -2,  37,  21),
        T( -2,  -6,  29,  25),
        T(  2, -14,  25,  33),
        T( -6,  -2,  41,  21),
        T(  2,  -6,  33,  25),
        T(  6, -14,  29,  33),
        T( -2,  -2,  45,  21),
        T(  6,  -6,  37,  25),
        T( 10, -14,  33,  33),
        T(  2,  -2,  49,  21),
        T( 10,  -6,  41,  25),
        T( 14, -14,  37,  33)))
    val box_regression = Tensor[Float](
      T(T(T(T(-1.6730e-02, -2.5040e-02, -3.8669e-02, -2.5333e-02, -1.4004e-02,
        -2.5377e-02, -1.2593e-02),
        T(-3.6522e-02, -1.0507e-02, -2.6155e-02, -3.6207e-02, -2.4963e-02,
          -2.1895e-02, -1.5993e-02),
        T(-1.6325e-02, -2.7535e-02, -1.6704e-02, -1.4899e-02, -1.1344e-02,
          -3.0802e-03, -1.2747e-02)),
        T(T(-7.5157e-03, -2.8978e-02, -2.8847e-02, -4.5879e-02, -3.0130e-02,
          -3.3889e-02, -5.1871e-02),
          T(-2.1900e-02, -2.2046e-02, -2.7110e-02, -3.2612e-02, -2.8986e-02,
            -6.6867e-02, -7.1081e-02),
          T( 3.0462e-03, -2.0255e-02, -3.9770e-02, -3.5203e-02, -4.7388e-02,
            -2.4220e-02, -4.6222e-02)),
        T(T( 5.7844e-04, -1.6412e-04,  9.7524e-03, -6.9274e-03,  1.7444e-06,
          5.4107e-03, -2.1182e-02),
          T(-1.5361e-02,  2.2865e-02,  1.7374e-02,  2.8522e-03,  3.3781e-02,
            1.0332e-02,  1.0356e-02),
          T( 3.3926e-03,  3.6011e-02,  1.8886e-02,  2.5415e-02,  2.0812e-02,
            2.1618e-02,  2.0776e-02)),
        T(T( 5.3066e-02,  5.4734e-02,  5.1326e-02,  3.5983e-02,  5.5721e-02,
          5.8108e-02,  3.7270e-02),
          T( 7.3613e-02,  5.4528e-02,  6.9086e-02,  5.8593e-02,  3.3255e-02,
            7.0331e-02,  3.9792e-02),
          T( 4.0440e-02,  4.5344e-02,  3.0102e-02,  3.9423e-02,  3.7462e-02,
            1.9178e-02,  3.4250e-02)),
        T(T( 9.3921e-03, -6.3640e-03,  6.6344e-03, -2.9477e-02,  2.8380e-03,
          2.4094e-04, -3.8125e-02),
          T( 1.3277e-02,  3.2003e-02,  9.2812e-03,  3.1793e-02,  3.5682e-02,
            5.4143e-03, -2.7538e-02),
          T(-1.4505e-02,  4.2906e-03, -5.5038e-03,  1.1895e-02, -8.9942e-03,
            9.1047e-03, -5.2846e-03)),
        T(T(-2.4140e-02, -4.9850e-02, -8.1354e-03, -4.0075e-02, -2.3858e-02,
          -1.0505e-02, -1.8872e-03),
          T(-5.3244e-02, -5.0973e-02, -5.3102e-02, -3.2843e-02, -4.9433e-02,
            -2.6899e-02, -2.1426e-02),
          T(-3.8070e-02, -3.4148e-02, -2.2365e-02, -1.0786e-02, -2.1428e-03,
            -2.9661e-02,  6.5642e-03)),
        T(T( 7.1718e-03, -1.8317e-02, -1.9746e-02,  3.5586e-04,  5.8551e-04,
          1.3969e-02, -2.5201e-03),
          T(-1.3888e-02, -9.6641e-03, -3.8934e-02, -2.8148e-02, -2.5934e-02,
            -1.8294e-02, -2.0061e-02),
          T( 1.0523e-02,  2.6551e-02, -2.9795e-02, -9.7123e-03, -1.4083e-03,
            -2.3482e-02, -1.5405e-02)),
        T(T( 2.5275e-02,  1.6022e-02,  2.1474e-02,  2.3938e-02,  1.6918e-02,
          2.9566e-02,  1.6430e-02),
          T(-8.9619e-03, -1.5747e-02,  2.2626e-02,  9.3860e-03, -2.7444e-03,
            1.0630e-02,  4.0585e-03),
          T(-2.6552e-02, -4.6460e-02, -1.1829e-02, -5.0394e-02, -2.1685e-02,
            -1.0684e-02, -3.7224e-02)),
        T(T( 8.2827e-03,  1.7244e-02,  2.7117e-02,  9.7096e-05,  3.1359e-02,
          4.6453e-03,  9.5188e-03),
          T( 4.0039e-02,  4.7410e-02,  9.9494e-03,  2.4956e-02,  2.7872e-02,
            2.4829e-02,  1.5199e-02),
          T( 2.1342e-02,  3.1655e-02,  2.1581e-02,  2.5497e-02,  5.2575e-02,
            2.4982e-02,  2.5912e-02)),
        T(T(-3.8185e-02, -3.9303e-02, -4.1358e-02, -4.0111e-02, -1.3078e-02,
          -2.2576e-02, -2.8542e-02),
          T(-3.6325e-02, -4.7150e-02, -1.7211e-02, -1.9650e-02,  5.6505e-04,
            -4.6043e-03, -4.4149e-02),
          T( 1.2474e-03, -2.1102e-02, -2.4141e-02,  9.8825e-03, -2.2259e-02,
            -1.1524e-02, -1.6652e-04)),
        T(T(-1.6188e-02, -2.3977e-02,  1.8660e-02, -1.5378e-02, -2.7290e-02,
          -2.5314e-02, -1.1265e-02),
          T(-2.8503e-02, -1.7718e-02, -5.1043e-03, -3.6894e-02, -1.6136e-02,
            -3.3021e-02, -1.9824e-02),
          T(-2.8551e-02, -3.7279e-02, -2.3878e-02, -2.9096e-02, -2.2290e-02,
            -2.6733e-02, -2.2998e-02)),
        T(T( 5.0010e-03, -8.0676e-03, -1.4430e-02, -1.5388e-02,  1.0738e-02,
          3.8478e-03,  2.1696e-03),
          T(-2.3630e-03, -4.0806e-02, -2.7923e-02, -1.1444e-02,  3.1605e-03,
            -1.7883e-02, -3.3700e-02),
          T( 5.6951e-03,  1.8676e-02, -2.4579e-03,  1.0234e-02,  3.3008e-03,
            3.0289e-03,  3.3703e-02))))
    )
    val objectness = Tensor[Float](T(T(T(
      T(-0.0429, -0.0315, -0.0317, -0.0458, -0.0145, -0.0326, -0.0305),
      T(-0.0361, -0.0716, -0.0414, -0.0237, -0.0399, -0.0334, -0.0345),
      T(-0.0168, -0.0163, -0.0441, -0.0193, -0.0388, -0.0227, -0.0345)),
      T(T( 0.0194, -0.0012,  0.0251, -0.0154, -0.0265, -0.0014,  0.0094),
        T( 0.0443,  0.0278,  0.0358,  0.0061,  0.0576,  0.0287,  0.0263),
        T(-0.0037, -0.0024,  0.0217,  0.0264,  0.0165,  0.0058,  0.0382)),
      T(T(-0.0011, -0.0058, -0.0089, -0.0017, -0.0266, -0.0007, -0.0156),
        T( 0.0087,  0.0164, -0.0103,  0.0014, -0.0262,  0.0151,  0.0157),
        T(-0.0223,  0.0009, -0.0051, -0.0074, -0.0148, -0.0156, -0.0043)))))

    val preNmsTopN: Int = 2000
    val postNmsTopN: Int = 2000
    val rpnPreNmsTopNTrain: Int = 2000

    //    val proposal = new Proposal(preNmsTopN, postNmsTopN,
//      ratios, scales, rpnPreNmsTopNTrain, rpnPostNmsTopNTrain, min_size = 0)

    val proposal = new RPNPostProcessor(2000, 2000, 0.7f, 0, 2000)
    proposal.forward(T(anchors, objectness, box_regression, Tensor[Float](T(20, 38))))

    println("done")
  }
}

class PriorBoxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    val minSizes = Array(460.8f)
    val maxSizes = Array(537.6f)
    val aspectRatios = Array(2f)
    val module = PriorBox[Float](minSizes = minSizes, maxSizes = maxSizes,
      _aspectRatios = aspectRatios, isFlip = isFlip, isClip = isClip,
      variances = variances, step = 0, offset = 0.5f, imgH = 512, imgW = 512)
    val input = Tensor[Float](8, 256, 1, 1)
    runSerializationTest(module, input)
  }
}
