/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.Sum
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

class TransformerLayerSpec extends ZooSpecHelper {
  "TransformerLayer" should "be able to work" in {
    val model = TransformerLayer[Float](vocab = 100, hiddenSize = 768, nBlock = 3)
    model.build(Shape(4, 77, 2))
    val w = model.parameters()._1
    require(w.length == 37)
    val input = Tensor[Float](Array(2, 2, 77, 2)).rand().resize(4, 77, 2)
    val gradOutput = Tensor[Float](4, 77, 768).rand()
    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)
  }

  "TransformerLayer with configured embedding" should "be able to work" in {
    val embedding = Sequential[Float]()
      .add(Reshape[Float](Array(77 * 2), inputShape = Shape(77, 2)))
      .add(Embedding[Float](100, 768, inputLength = 77 * 2))
      .add(Reshape[Float](Array(77, 2, 768)))
      .add(new KerasLayerWrapper[Float](Sum[Float](dimension = 3,
        squeeze = true).asInstanceOf[AbstractModule[Activity, Activity, Float]]))
    val model = TransformerLayer[Float](nBlock = 3,
      residPdrop = 0.1, attnPdrop = 0.1, nHead = 12, maskAttention = false,
      embeddingLayer = embedding.asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]])

    model.build(Shape(4, 77, 2))
    val input = Tensor[Float](Array(2, 2, 77, 2)).rand().resize(4, 77, 2)
    val gradOutput = Tensor[Float](4, 77, 768).rand()
    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)
  }

  "TransformerLayer" should "be able to generate correct result" in {
    RNG.setSeed(42)
    val layer = TransformerLayer[Float](vocab = 10, hiddenSize = 4, seqLen = 2, nHead = 2,
      residPdrop = 0, attnPdrop = 0, nBlock = 1)
    val data = Array[Float](6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7, 2, 5, 4)
    layer.build(Shape(4, 2, 2))
    val wb = layer.parameters()._1

    val embedingW = Tensor[Float](Array[Float](0.0035f, 0.0210f, 0.0001f, -0.0015f,
    0.0129f, 0.0115f, 0.0117f, -0.0004f,
    -0.0183f, 0.0297f, -0.0182f, -0.0106f,
    -0.0161f, 0.0103f, -0.0143f, 0.0044f,
     0.0113f, 0.0372f, 0.0209f, -0.0173f,
     0.0167f, -0.0063f, 0.0054f, 0.0017f,
     0.0195f, -0.0203f, -0.0108f, -0.0088f,
    -0.0063f, -0.0026f, -0.0143f, -0.0010f,
     0.0404f, 0.0051f, 0.0187f, 0.0142f,
    -0.0006f, 0.0020f, 0.0269f, 0.0143f),
      Array(10, 4))
    wb(0).set(embedingW)

    val conv1W = Tensor[Float](Array[Float](-0.0167f, -0.0184f, 0.0362f, 0.0032f, 0.0073f,
      0.0035f, 0.0277f, -0.0089f, -0.0240f, 0.0142f, -0.0215f, 0.0107f,
      0.0235f, 0.0112f, -0.0091f, -0.0154f, 0.0029f, 0.0046f, 0.0002f, -0.0028f, 0.0039f,
      -0.0229f, 0.0068f, 0.0340f, 0.0563f, 0.0072f, -0.0018f, 0.0092f, -0.0113f, 0.0211f,
      -0.0294f, 0.0287f, 0.0146f, -0.0142f, -0.0120f, 0.0192f, 0.0081f, -0.0271f, -0.0100f,
      0.0095f, -0.0040f, 0.0254f, 0.0245f, 0.0020f, 0.0348f, -0.0271f, 0.0044f, 0.0111f),
      Array(1, 1, 1, 4, 12))
    wb(1).set(conv1W)

    val conv2W = Tensor[Float](Array[Float](-0.0136f, 0.0115f, 0.0038f, -0.0072f,
    -0.0063f, 0.0118f, -0.0178f, 0.0082f,
    -0.0197f, 0.0025f, 0.0070f, 0.0123f,
    -0.0034f, 0.0047f, 0.0807f, 0.0256f), Array(1, 1, 1, 4, 4))
    wb(3).set(conv2W)

    val conv3W = Tensor[Float](Array[Float](0.0206f, -0.0141f, 0.0203f, -0.0066f, 0.0104f,
      0.0078f, -0.0116f, -0.0034f, -0.0115f, 0.0101f, -0.0095f, -0.0098f, 0.0054f, -0.0113f,
      0.0136f, 0.0088f, -0.0072f, -0.0012f, 0.0015f, 0.0164f, 0.0296f, 0.0069f, -0.0285f,
      -0.0023f, 0.0044f, -0.0009f, -0.0287f, -0.0113f, -0.0085f, 0.0053f, -0.0288f, 0.0104f,
      0.0208f, -0.0080f, -0.0459f, 0.0100f, -0.0085f, -0.0267f, -0.0039f, 0.0131f, -0.0061f,
      -0.0066f, -0.0196f, 0.0039f, -0.0331f, 0.0136f, 0.0292f, -0.0062f, 0.0193f, -0.0062f,
      0.0114f, 0.0224f, -0.0259f, 0.0010f, -0.0117f, -0.0078f, 0.0196f, -0.0128f, -0.0098f,
      0.0042f, -0.0232f, -0.0193f, -0.0075f, 0.0161f), Array(1, 1, 1, 4, 16))
    wb(7).set(conv3W)

    val conv4W = Tensor[Float](Array[Float](0.0143f, 0.0307f, -0.0290f, -0.0157f,
    -0.0191f, -0.0250f, -0.0150f, -0.0118f,
    -0.0307f, -0.0145f, 0.0093f, 0.0133f,
    -0.0009f, 0.0047f, -0.0141f, -0.0143f,
    -0.0032f, -0.0085f, 0.0189f, -0.0037f,
     0.0212f, 0.0042f, -0.0116f, 0.0065f,
     0.0052f, -0.0152f, -0.0409f, -0.0306f,
     0.0081f, 0.0126f, 0.0063f, -0.0007f,
     0.0261f, 0.0098f, 0.0227f, -0.0071f,
     0.0072f, 0.0400f, 0.0133f, 0.0141f,
     0.0004f, -0.0166f, -0.0216f, -0.0157f,
     0.0101f, 0.0016f, 0.0089f, -0.0145f,
    -0.0092f, -0.0013f, -0.0273f, 0.0066f,
    -0.0197f, 0.0060f, 0.0036f, -0.0026f,
    -0.0315f, 0.0450f, 0.0200f, 0.0273f,
     0.0127f, 0.0081f, 0.0068f, -0.0044f), Array(1, 1, 1, 16, 4))
    wb(9).set(conv4W)

    val input = Tensor[Float](data, Array(4, 2, 2))
    val output = layer.forward(input).toTensor[Float]

    val expect = Tensor[Float](Array[Float](1.1891f, -0.0895f, -1.5431f, 0.4436f,
    -0.1075f, 1.4737f, -0.0185f, -1.3477f,
    0.9136f, -1.6274f, 0.7188f, -0.0050f,
    0.6926f, 1.2211f, -1.2716f, -0.6420f,
    -0.1062f, 1.4747f, -0.0218f, -1.3466f,
    -0.7912f, 1.1205f, -1.1806f, 0.8513f,
    -0.6202f, 1.6365f, -0.9668f, -0.0495f,
    0.5538f, 0.7061f, 0.4657f, -1.7256f), Array(4, 2, 4))
    require(output.almostEqual(expect, 7e-3) == true)

    val gradOutput = Tensor[Float](Array[Float](1f, 23f, 43f, 29f,
      37f, 1f, 20f, 32f,

      11f, 21f, 43f, 24f,
      48f, 26f, 41f, 27f,

      15f, 14f, 46f, 43f,
      2f, 36f, 6f, 20f,

      8f, 38f, 17f, 3f,
      24f, 13f, 49f, 8f), Array(4, 2, 4))

    val gradOutput2 = Tensor[Float](Array[Float](0.0931f, 0.9193f, 0.2999f, 0.6325f,
      0.3265f, 0.5406f, 0.9662f, 0.7304f,

      0.0667f, 0.6985f, 0.9746f, 0.6315f,
      0.8352f, 0.9929f, 0.4234f, 0.6038f,

      0.1525f, 0.3970f, 0.8703f, 0.7563f,
      0.1836f, 0.0991f, 0.1583f, 0.0066f,

      0.1142f, 0.3764f, 0.8374f, 0.5837f,
      0.1197f, 0.0989f, 0.7487f, 0.1281f), Array(4, 2, 4))
    layer.backward(input, gradOutput2)
    val grads = layer.parameters()._2

//    val expectGrad = Array(Tensor[Float](Array[Float](0.0000f, 0.0000f, 0.0000f, 0.0000f,
//      0.0000f, 0.0000f, 0.0000f, 0.0000f,
//      764.1672f, -310.7225f, 717.7441f, -1164.1603f,
//      -932.1359f, 191.7078f, 89.8583f, 650.3801f,
//      -326.5101f, -1124.5602f, 1871.8000f, -410.8221f,
//      -202.6295f, -838.5383f, 1142.7347f, -97.7104f,
//      -839.6799f, -805.6633f, 1246.0300f, 407.5446f,
//      -498.9646f, 334.1792f, 1461.9297f, -1283.9550f,
//      0.0000f, 0.0000f, 0.0000f, 0.0000f,
//      -1046.7954f, -66.4475f, 1171.2919f, -49.5184f), Array(10, 4)),
//    Tensor[Float](Array[Float](-5.5860e-09f, 3.8053e-08f, -1.3828e-07f, 7.3740e-07f, 1.1079e-07f,
//      8.3889e-10f, 9.9948e-07f, -4.5448e-07f, -1.0786e-01f, -3.4225e-01f,
//      4.2096e-01f, 1.5764e+00f,
//      -7.4000e-09f, 3.0568e-08f, 1.2007e-08f, 7.7107e-07f, 7.5051e-08f,
//      4.1891e-08f, -4.7708e-08f, 2.1530e-07f, 5.5976e-01f, -1.3823e+00f,
//      1.7728e-01f, 3.8036e+00f,
//      -1.5474e-08f, 6.7919e-08f, -5.9570e-08f, 5.0120e-07f, 1.8792e-07f,
//      2.3124e-08f, 1.1170e-06f, -4.3537e-07f, 1.6990e-01f, -3.0353e-02f,
//      3.1043e-01f, 7.2220e-01f,
//      -3.2099e-10f, -8.5389e-09f, -2.1056e-08f, -4.4560e-07f, -3.4326e-08f,
//      -1.2771e-08f, 1.3796e-08f, -6.5944e-08f, -1.1607e-01f, 8.1057e-01f,
//      -1.0330e-01f, -2.1564e+00f), Array(4, 12)),
//    Tensor[Float](Array[Float](4.2036e-08f, -2.0282e-07f, -3.6947e-06f, 3.1246e-05f, -5.1159e-13f,
//      9.9476e-14f, 1.8190e-12f, -4.5475e-13f, 2.7192e+01f, -8.2303e+01f,
//      4.1947e+01f, 2.8423e+02f), Array(12)),
//    Tensor[Float](Array[Float](0.1774f, 0.9076f, -1.1504f, 0.0654f,
//      0.1908f, -0.4471f, -0.3714f, 0.6278f,
//      0.2843f, 0.3933f, -0.1789f, -0.4988f,
//      -0.7052f, -0.3330f, 1.9734f, -0.9352f), Array(4, 4)),
//    Tensor[Float](Array[Float](-1541.5975f, -1321.9561f, 3844.1738f, -980.6201f), Array(4)),
//    Tensor[Float](Array[Float](-9.9459f, -12.1985f, 4.9218f, 17.2262f), Array(4)),
//    Tensor[Float](Array[Float](-22.1844f, -21.9873f, 70.3989f, -26.1735f), Array(4)),
//    Tensor[Float](Array[Float](-0.8689f, 0.1771f, 0.5487f, -0.3014f, 0.2500f, -0.1771f
//      -0.4842f, -0.1526f,
//      -0.1424f, -0.3105f, -0.1275f, -0.1320f, -0.1052f, 0.0238f, 0.0893f, -0.1542f,
//      -0.0337f, -0.0737f, -0.5237f, 0.0713f, 0.5239f, -0.1239f, 0.2538f, 0.1524f,
//      0.8591f, -0.2829f, 0.1389f, 0.6192f, -0.7927f, -0.1691f, -1.0939f, 0.3557f,
//      -0.3724f, 0.0804f, 0.3145f, -0.1405f, 0.0544f, -0.1085f, -0.3000f, -0.0795f,
//      -0.1853f, -0.0412f, -0.1091f, -0.1348f, 0.0409f, 0.0916f, 0.2977f, -0.1148f,
//      1.2751f, -0.1838f, -0.3395f, 0.3707f, -0.8283f, 0.4095f, 0.5304f, 0.0797f,
//      -0.5313f, 0.6345f, 0.0977f, -0.3524f, 0.8569f, 0.0537f, 0.7070f, -0.0866f), Array(4, 16)),
//    Tensor[Float](Array[Float](-1.3110f, 0.1112f, 0.6377f, -0.3355f, 0.8961f, -0.7706f,
//      -0.9400f, -0.0027f,
//      0.4876f, -0.2555f, -0.3796f, 0.3587f, -0.9551f, 0.3090f, 0.2518f, 0.0713f), Array(16)),
//    Tensor[Float](Array[Float](-0.3871f, 0.1367f, -0.4065f, 0.6569f,
//      0.1152f, 0.0772f, 0.0187f, -0.2111f,
//      0.1490f, 0.1171f, -0.2207f, -0.0454f,
//      0.1296f, 0.4464f, -0.3639f, -0.2121f,
//      0.4684f, -0.9143f, 1.4119f, -0.9660f,
//      0.2093f, -0.0356f, 0.1024f, -0.2762f,
//      -0.2795f, 0.0744f, -0.2511f, 0.4562f,
//      -0.0738f, -0.1607f, 0.1879f, 0.0466f,
//      0.0918f, 0.5696f, -0.5661f, -0.0952f,
//      0.0105f, -0.3923f, 0.4206f, -0.0388f,
//      -0.2046f, 0.1405f, -0.3352f, 0.3993f,
//      -0.1374f, 0.2695f, -0.3777f, 0.2456f,
//      0.1211f, -0.4359f, 0.4789f, -0.1640f,
//      0.0889f, -0.3737f, 0.5909f, -0.3062f,
//      -0.6535f, -0.2480f, -0.0047f, 0.9061f,
//      0.0880f, 0.1994f, -0.2068f, -0.0806f), Array(16, 4)),
//    Tensor[Float](Array[Float](-22.1725f, -22.0525f, 70.3928f, -26.1679f), Array(4)),
//    Tensor[Float](Array[Float](45.6017f, 129.3080f, -89.2980f, -102.6128f), Array(4)),
//    Tensor[Float](Array[Float](146f, 172f, 265f, 186f), Array(4)))

    val expectGrad2 = Array(Tensor[Float](Array[Float](0.0000f, 0.0000f, 0.0000f, 0.0000f,
      0.0000f, 0.0000f, 0.0000f, 0.0000f,
      -19.7219f, 1.8679f, 11.6526f, 6.3669f,
      -31.3784f, 44.0520f, -24.5972f, 11.8294f,
      -48.7716f, -10.7519f, 57.9907f, 1.7853f,
      -10.0032f, -11.5610f, 23.2667f, -1.6240f,
      -65.2076f, 38.8184f, 6.3718f, 20.1353f,
      -54.8101f, 4.7251f, 46.2947f, 4.1267f,
      0.0000f, 0.0000f, 0.0000f, 0.0000f,
      -30.1489f, -3.1855f, 30.8872f, 2.6565f), Array(10, 4)),
      Tensor[Float](Array[Float](-1.4218e-11f, 2.9997e-11f, -8.5611e-09f, 1.8073e-08f,
        -6.2485e-11f,
        -5.4072e-11f, 2.1598e-08f, -7.1842e-09f, 7.7241e-03f, -3.2117e-03f,
        1.2788e-02f, 3.4644e-02f,
        -2.1551e-10f, 6.4366e-10f, -2.6124e-08f, 2.9787e-08f, 1.9608e-09f,
        1.6421e-09f, 3.2552e-09f, -1.4330e-08f, 1.0764e-02f, -2.8874e-02f,
        2.3433e-02f, 1.1676e-01f,
        -4.4891e-11f, 1.3365e-10f, -9.4939e-09f, 1.8279e-08f, 1.2599e-09f,
        1.0390e-09f, 2.4817e-08f, -1.7214e-08f, -2.0470e-02f, -3.2931e-02f,
        -2.3353e-02f, 2.6428e-02f,
        1.0735e-10f, -3.3585e-10f, 1.3839e-08f, -1.5290e-08f, -6.3179e-10f,
        -4.9743e-10f, -2.6125e-09f, 4.7287e-09f, -1.5175e-02f, 4.2444e-03f,
        -2.4610e-02f, -6.2192e-02f), Array(4, 12)),
      Tensor[Float](Array[Float](-0.0000f, 0.0000f, -0.0000f, 0.0000f, -0.0000f,
        0.0000f, 0.0000f, 0.0000f,
        2.2525f, 0.0301f, 3.4501f, 7.2811f), Array(12)),
      Tensor[Float](Array[Float](0.0541f, -0.0222f, -0.0191f, -0.0127f,
        -0.0268f, 0.0316f, -0.0179f, 0.0131f,
        -0.0030f, 0.0085f, -0.0083f, 0.0028f,
        -0.0036f, -0.0412f, 0.0579f, -0.0131f), Array(4, 4)),
      Tensor[Float](Array[Float](-129.9709f, 31.7035f, 75.8026f, 22.4648f), Array(4)),
      Tensor[Float](Array[Float](-0.6313f, 0.0065f, 0.6579f, -0.0330f), Array(4)),
      Tensor[Float](Array[Float](-2.1539f, 0.2713f, 1.5711f, 0.3132f), Array(4)),
      Tensor[Float](Array[Float](-0.0041f, 0.0001f, 0.0093f, -0.0012f, 0.0001f, -0.0057f,
        -0.0103f, -0.0004f,
        -0.0062f, 0.0062f, -0.0059f, -0.0039f, 0.0022f, 0.0067f, 0.0222f, -0.0029f,
        -0.0198f, 0.0026f, 0.0231f, -0.0076f, 0.0135f, -0.0183f, -0.0226f, -0.0015f,
        -0.0040f, 0.0035f, -0.0109f, -0.0021f, -0.0084f, 0.0136f, 0.0235f, -0.0041f,
        -0.0179f, 0.0028f, -0.0020f, -0.0046f, 0.0096f, -0.0003f, -0.0014f, -0.0013f,
        0.0113f, -0.0140f, 0.0027f, 0.0069f, -0.0126f, -0.0061f, -0.0227f, 0.0029f,
        0.0418f, -0.0055f, -0.0304f, 0.0133f, -0.0233f, 0.0244f, 0.0344f, 0.0031f,
        -0.0011f, 0.0043f, 0.0142f, -0.0009f, 0.0187f, -0.0143f, -0.0230f, 0.0040f),
        Array(4, 16)),
      Tensor[Float](Array[Float](-0.0356f, 0.0036f, 0.0403f, -0.0115f, 0.0178f, -0.0305f,
        -0.0440f, -0.0023f,
        -0.0101f, 0.0104f, -0.0218f, -0.0059f, -0.0106f, 0.0246f, 0.0582f, -0.0079f), Array(16)),
      Tensor[Float](Array[Float](0.0138f, -0.0006f, -0.0115f, -0.0017f,
        -0.0009f, -0.0004f, 0.0015f, -0.0002f,
        -0.0144f, 0.0286f, -0.0239f, 0.0097f,
        0.0089f, 0.0011f, -0.0085f, -0.0015f,
        -0.0410f, -0.0024f, 0.0384f, 0.0050f,
        -0.0136f, 0.0139f, -0.0058f, 0.0055f,
        0.0110f, -0.0035f, -0.0052f, -0.0022f,
        0.0010f, -0.0085f, 0.0099f, -0.0023f,
        0.0111f, 0.0066f, -0.0174f, -0.0003f,
        -0.0124f, 0.0007f, 0.0094f, 0.0022f,
        0.0074f, 0.0044f, -0.0122f, 0.0004f,
        0.0128f, -0.0020f, -0.0084f, -0.0024f,
        -0.0205f, 0.0092f, 0.0057f, 0.0056f,
        -0.0082f, -0.0132f, 0.0239f, -0.0025f,
        0.0126f, -0.0131f, 0.0049f, -0.0045f,
        -0.0008f, 0.0089f, -0.0105f, 0.0023f), Array(16, 4)),
      Tensor[Float](Array[Float](-2.1553f, 0.2705f, 1.5702f, 0.3146f), Array(4)),
      Tensor[Float](Array[Float](0.5485f, 2.1720f, -0.9802f, -2.3583f), Array(4)),
      Tensor[Float](Array[Float](1.8915f, 4.1225f, 5.2788f, 4.0728f), Array(4)))

    var i = grads.length - 1
    while (i >= 0) {
      // gradout2 is smaller than gradoutput, if use gradoutput, the diff is smaller than 7
      require(grads(i).squeeze().almostEqual(expectGrad2(i), 0.3) == true)
      i -= 1
    }
  }

  "Attention" should "be able to generate correct result" in {
    val transformerLayer = TransformerLayer[Float](vocab = 10, hiddenSize = 4,
      seqLen = 2, nHead = 2,
      residPdrop = 0, attnPdrop = 0, nBlock = 1)
    transformerLayer.build(Shape(2, 2, 2))

    val xValue = Tensor[Float](Array[Float](0.6532f, 0.3958f, 0.9147f, 0.2036f,
    0.2018f, 0.2018f, 0.9497f, 0.6666f,

    0.9811f, 0.0874f, 0.0041f, 0.1088f,
    0.1637f, 0.7025f, 0.6790f, 0.9155f), Array(2, 2, 4))
    val gradOValue = Tensor[Float](Array[Float](3f, 4f, 2f, 4f,
    4f, 1f, 2f, 2f,

    2f, 4f, 3f, 2f,
    4f, 1f, 3f, 1f), Array(2, 2, 4))
    val x = Variable[Float](inputShape = Shape(2, 4))

    val y = transformerLayer.multiHeadSelfAttention(x, 4)
    val model = Model[Float](input = x, output = y)

    val wb = model.parameters()._1
    val conv1W = Tensor[Float](Array[Float](3.8538f, 2.9746f, 1.8014f, -4.2110f, 1.3568f,
      -2.4691f, -0.0861f, -3.2093f,
      -1.5043f, 3.2974f, -0.7850f, -2.8072f,
      -1.4558f, -1.1189f, -1.5377f, 1.5249f, 3.2846f, -0.3192f, -0.9948f, 0.8792f,
      -1.5163f, 2.1566f, 1.6016f, 3.3612f,
      2.5582f, 2.5928f, 1.2209f, 2.6695f, -0.4632f, 0.0835f, -0.5032f, 1.7197f,
      -2.7693f, -1.7425f, -0.4467f, 3.4347f,
      0.6378f, -0.8490f, 0.6114f, -1.5492f, -3.1151f, 1.9913f, -1.7596f, -1.2023f,
      -2.5483f, 4.2456f, -2.4693f, -0.9758f),
      Array(1, 1, 1, 4, 12))
    wb(0).set(conv1W)

    val conv2W = Tensor[Float](Array[Float](-1.8276f, -1.3163f, 0.1560f, 1.0516f,
      -0.9760f, 2.3827f, -1.6280f, -1.4720f,
      -2.8065f, 0.0720f, -0.1270f, 1.3512f,
      -0.1956f, 3.6892f, -2.3691f, 2.7671f), Array(1, 1, 1, 4, 4))
    wb(2).set(conv2W)

    val output2 = model.forward(xValue).toTensor[Float]
    val expectOutput = Tensor[Float](Array[Float](7.9879f, 20.4734f, -10.1122f, -2.5445f,
      9.4308f, 20.9366f, -10.3584f, -2.8922f,

      2.7088f, 2.2557f, -0.4481f, -15.9850f,
      4.8519f, 19.4551f, -10.7697f, -6.6063f), Array(2, 2, 4))
    require(output2.almostEqual(expectOutput, 7e-3) == true)

    val expectGradients = Array(Tensor[Float](Array[Float](-0.8818f, 1.5712f, -0.1367f, 0.1635f,
      -4.3644f, -1.5773f, -1.2617f,
      -0.6638f, -19.5486f, -12.6856f, -12.9093f, 27.0076f,
      -3.6776f, 6.6610f, 0.1674f, -0.2800f, 3.1778f, 1.1145f, -0.6854f,
      -0.4695f, -7.2853f, -6.0781f, -9.4022f, 9.6217f,
      -3.6768f, 6.5332f, -0.6946f, 0.8445f, 3.5363f, 1.2565f, 0.0074f,
      -0.0650f, -12.8066f, -10.9420f, -15.7143f, 21.5642f,
      -4.8576f, 8.7308f, -0.2399f, 0.2316f, 4.3101f, 1.5587f, 1.2987f,
      0.6869f, -5.7595f, -4.9447f, -11.4458f, 6.1423f), Array(1, 1, 1, 4, 12)),
      Tensor[Float](Array[Float](-5.3502e+00f, 9.5711e+00f, -5.7053e-01f, 6.5464e-01f, 4.7684e-07f,
        8.3447e-07f, 1.1325e-06f, 5.2154e-07f, -2.5897e+01f, -1.8388e+01f,
        -2.4873e+01f, 3.5562e+01f), Array(12)),
      Tensor[Float](Array[Float](-48.0021f, -33.7029f, -33.0392f, -34.5581f,
        40.2182f, 31.0341f, 33.1337f, 25.5552f,
        -14.8660f, -9.4375f, -10.9988f, -8.9403f,
        18.8538f, 3.7079f, 7.1308f, 11.3839f), Array(1, 1, 1, 4, 4)),
      Tensor[Float](Array[Float](13, 10, 10, 9), Array(4)))
    model.backward(xValue, gradOValue)
    val gradients = model.parameters()._2
    for (i <- 0 until gradients.length) {
      require(gradients(i).almostEqual(expectGradients(i), 6e-3) == true)
    }
  }

  "KerasUtils tril" should "be able to work" in {
    val data = Tensor.ones[Float](3, 3)
    KerasUtils.tril(data)
    val expect = Array[Float](1, 0, 0, 1, 1, 0, 1, 1, 1)
    val res = data.storage().array()
    require(expect.deep == res.deep)

    val data2 = Tensor.ones[Float](4, 6)
    KerasUtils.tril(data2)
    val expect2 = Array[Float](1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0)
    val res2 = data2.storage().array()
    require(expect2.deep == res2.deep)
  }

  "Conv1D" should "be generate the same result with pytorch-openai conv1d" in {
    val x = Variable[Float](Shape(3, 5))
    val m = Conv1D[Float](3, 1)
    val o = m.from(x)
    val model = Model[Float](input = x, output = o)
    val data = (2 until 32).toArray.map(_.toFloat)

    val w = Tensor[Float]((30 until 45).toArray.map(_.toFloat),
      Array(1, 1, 1, 5, 3))
    model.setWeightsBias(Array(w, Tensor[Float](3)))
    val input = Tensor[Float](data, Array(2, 3, 5))
    val output2 = model.forward(input).toTensor[Float]

    val expect = Tensor[Float](Array[Float](750, 770, 790, 1650, 1695, 1740, 2550,
      2620, 2690, 3450,
      3545, 3640, 4350, 4470, 4590, 5250, 5395, 5540), Array(2, 3, 3))
    require(output2.almostEqual(expect, 1e-8) == true)

    val gradOutput = Tensor[Float]((2 until 20).toArray.map(_.toFloat), Array(2, 3, 3))
    model.backward(input, gradOutput)
    val grads = model.parameters()._2

    val expectGrad = Array(Tensor[Float](Array[Float](1089, 1176, 1263,
      1146, 1239, 1332,
      1203, 1302, 1401,
      1260, 1365, 1470,
      1317, 1428, 1539), Array(5, 3)),
      Tensor[Float](Array[Float](57, 63, 69), Array(3)))

    for (i <- 0 until grads.length) {
      require(grads(i).almostEqual(expectGrad(i), 1e-8) == true)
    }
  }
}

class TransformerLayerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = TransformerLayer[Float](vocab = 100, hiddenSize = 768, nBlock = 3)
    layer.build(Shape(2, 77, 2))
    val input = Tensor[Float](Array(2, 77, 2)).rand()
    runSerializationTest(layer, input)
  }
}
