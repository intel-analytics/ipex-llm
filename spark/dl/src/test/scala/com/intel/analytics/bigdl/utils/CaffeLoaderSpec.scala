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

package com.intel.analytics.bigdl.utils

import java.nio.file.Paths

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.caffe.{CaffeLoader, Customizable}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

class CaffeLoaderSpec extends FlatSpec with Matchers {

  val prototxt = getClass().getClassLoader().getResource("caffe/test.prototxt").getPath
  val modelPath = getClass().getClassLoader().getResource("caffe/test.caffemodel").getPath

  val dummyConverter = (layer : GeneratedMessage) => {
    Identity[Double].inputs()
  }

  private class LoadDummy[T: ClassTag](implicit ev: TensorNumeric[T]) extends Customizable[T] {
    override def convertor(layer: GeneratedMessage): Seq[ModuleNode[T]] = {
      Seq(Identity[T].setName("Dummy").inputs())
    }
  }

  val convertMap = new mutable.HashMap[String, Customizable[Double]]()
  convertMap("DUMMY") = new LoadDummy[Double]()

  "load caffe match all parameters" should "work properly" in {
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv2"))
      .add(Linear(2, 27, withBias = false).setName("ip"))

    val model = CaffeLoader.load[Double](module, prototxt, modelPath, true,
      convertMap)
    val parameters = model.getParametersTable()

    val conv1weight = Tensor(Storage(Array(
      0.4156779647, 0.3547672033, 0.1817495823, -0.1393318474,
      0.4004031420, 0.0634599924, 0.1571258903, 0.4180541039,
      0.3160640597, -0.0705609620, 0.4400904775, 0.1356888413,
      0.3336675167, -0.4155709147, 0.3805110455, -0.2870244980,
      -0.4660916030, 0.1869867444, 0.1779280305, -0.4752498865,
      -0.1661637127, 0.3539937139, -0.4798593223, -0.1620789468,
      -0.3706701994, 0.4643850327, -0.4701072574, 0.2113500834,
      -0.1643440127, -0.4747982025, -0.1300015152, 0.3333964944,
      0.1151610613, -0.4214298427, -0.4075299501, -0.1441932321,
      -0.3215276599, 0.4862193465, 0.0050434470, 0.4745523334,
      0.3657383919, -0.2879499197, 0.3388324380, 0.3669666648,
      -0.4454920888, -0.4200569391, -0.4690187573, -0.4590228796))).resize(4, 3, 2, 2)

    val conv1bias = Tensor(Storage(Array(0.0458712392, -0.0029324144, -0.0251041390, 0.0052924110)))

    val conv2weight = Tensor(Storage(Array(
      0.0154178329, 0.0157190431, 0.0033829932, -0.0048461366,
      -0.0026736879, 0.0009068546, -0.0020218866, 0.0096962797,
      0.0100709666, 0.0152738532, -0.0048784190, -0.0099342400,
      -0.0188637581, -0.0053443452, 0.0035097739, -0.0104582068,
      0.0212461911, -0.0026065058, 0.0000952507, 0.0113442009,
      0.0247142352, 0.0033546593, -0.0127880797, 0.0040104976,
      -0.0121186078, 0.0055492506, -0.0097251972, 0.0087026395,
      -0.0078147361, 0.0101692677, -0.0027364481, 0.0007095702,
      -0.0088762743, 0.0061115879, 0.0048167249, -0.0107875718,
      -0.0249741413, -0.0018652071, 0.0028419730, 0.0255292989,
      -0.0091862874, 0.0010728909, 0.0009157739, 0.0073709050,
      -0.0088602817, -0.0093507599, 0.0070853345, -0.0074293613))).resize(3, 4, 2, 2)

    val conv2bias = Tensor(Storage(Array[Double](0, 0, 0)))

    val linearWeight = Tensor(Storage(Array(
      0.0189033747, 0.0401176214, 0.0525088012, 0.3013394773,
      -0.0363914967, -0.3332226574, -0.2289898694, 0.3278202116,
      0.1829113662, 0.1653866768, -0.2218630016, -0.2914664149,
      0.0594480336, 0.1987790167, -0.2698714137, -0.2847212255,
      0.2896176279, 0.3278110921, -0.2233058512, 0.0355354548,
      -0.2735285461, 0.1467721164, -0.1557070315, -0.2237440944,
      0.2817622125, -0.0810049772, 0.1050063074, -0.0378594697,
      0.2178583443, 0.0811733305, -0.0678446293, 0.0180019736,
      -0.0949532837, 0.2528320253, -0.0265761316, -0.0096390843,
      -0.2360083759, 0.1942299902, -0.3302336931, -0.2042815089,
      -0.3027454615, 0.1254911423, 0.2114857137, 0.0392392874,
      0.1668677032, 0.0506658256, 0.1139862537, 0.2874754369,
      -0.3273061812, 0.2115428150, -0.2002333999, -0.1621897519,
      0.0032395422, 0.2072965205))).resize(2, 27)

    parameters[Table]("conv")[Tensor[Double]]("weight").map(conv1weight,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })

    parameters[Table]("conv")[Tensor[Double]]("bias").map(conv1bias,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })

    parameters[Table]("conv2")[Tensor[Double]]("weight").map(conv2weight,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })

    parameters[Table]("conv2")[Tensor[Double]]("bias").map(conv2bias,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })

    parameters[Table]("ip")[Tensor[Double]]("weight").map(linearWeight,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })
  }

  "load caffe match part parameters" should "work properly" in {
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv3"))
      .add(Linear(2, 27, withBias = false).setName("ip"))

    val model = CaffeLoader.load[Double](module, prototxt, modelPath, matchAll = false)
    val parameters = model.getParametersTable()

    val conv1weight = Tensor(Storage(Array(
      0.4156779647, 0.3547672033, 0.1817495823, -0.1393318474,
      0.4004031420, 0.0634599924, 0.1571258903, 0.4180541039,
      0.3160640597, -0.0705609620, 0.4400904775, 0.1356888413,
      0.3336675167, -0.4155709147, 0.3805110455, -0.2870244980,
      -0.4660916030, 0.1869867444, 0.1779280305, -0.4752498865,
      -0.1661637127, 0.3539937139, -0.4798593223, -0.1620789468,
      -0.3706701994, 0.4643850327, -0.4701072574, 0.2113500834,
      -0.1643440127, -0.4747982025, -0.1300015152, 0.3333964944,
      0.1151610613, -0.4214298427, -0.4075299501, -0.1441932321,
      -0.3215276599, 0.4862193465, 0.0050434470, 0.4745523334,
      0.3657383919, -0.2879499197, 0.3388324380, 0.3669666648,
      -0.4454920888, -0.4200569391, -0.4690187573, -0.4590228796))).resize(4, 3, 2, 2)

    val conv1bias = Tensor(Storage(Array(0.0458712392, -0.0029324144, -0.0251041390, 0.0052924110)))

    val linearWeight = Tensor(Storage(Array(
      0.0189033747, 0.0401176214, 0.0525088012, 0.3013394773,
      -0.0363914967, -0.3332226574, -0.2289898694, 0.3278202116,
      0.1829113662, 0.1653866768, -0.2218630016, -0.2914664149,
      0.0594480336, 0.1987790167, -0.2698714137, -0.2847212255,
      0.2896176279, 0.3278110921, -0.2233058512, 0.0355354548,
      -0.2735285461, 0.1467721164, -0.1557070315, -0.2237440944,
      0.2817622125, -0.0810049772, 0.1050063074, -0.0378594697,
      0.2178583443, 0.0811733305, -0.0678446293, 0.0180019736,
      -0.0949532837, 0.2528320253, -0.0265761316, -0.0096390843,
      -0.2360083759, 0.1942299902, -0.3302336931, -0.2042815089,
      -0.3027454615, 0.1254911423, 0.2114857137, 0.0392392874,
      0.1668677032, 0.0506658256, 0.1139862537, 0.2874754369,
      -0.3273061812, 0.2115428150, -0.2002333999, -0.1621897519,
      0.0032395422, 0.2072965205))).resize(2, 27)

    parameters[Table]("conv")[Tensor[Double]]("weight").map(conv1weight,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })

    parameters[Table]("conv")[Tensor[Double]]("bias").map(conv1bias,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })

    parameters[Table]("ip")[Tensor[Double]]("weight").map(linearWeight,
      (a, b) => {
        assert(Math.abs(a - b) < 1e-6)
        a
      })
  }

  "Dynamic loaded module" should "have the same result as static one" in {
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv2"))
      .add(View(27)).setName("view")
      .add(Linear(27, 2, withBias = false).setName("ip"))
      .add(SoftMax().setName("softmax"))

    val staticInput = Tensor[Double](1, 3, 5, 5).apply1( e => Random.nextDouble())

    val dynamicInput = Tensor[Double]()

    dynamicInput.resizeAs(staticInput).copy(staticInput)

    RandomGenerator.RNG.setSeed(1000)
    val staticModel = CaffeLoader.load[Double](module, prototxt, modelPath)

    val staticOutPut = staticModel.forward(staticInput)

    RandomGenerator.RNG.setSeed(1000)
    val dynamicLoadedModule = CaffeLoader.loadCaffe(prototxt, modelPath,
      customizedConverters = convertMap)._1

    val dynamicOutput = dynamicLoadedModule.forward(dynamicInput)

    staticOutPut should be (dynamicOutput)
  }

  "Dynamic loaded criterion" should "have the same criterion as static one" in {

    val criterion = ClassNLLCriterion[Double]()

    val staticCriterion = ParallelCriterion[Double]()
    staticCriterion.add(criterion)

    val (dynamicLoadedModule, dynamicLoadedCriterion) = CaffeLoader.loadCaffe(prototxt, modelPath,
      customizedConverters = convertMap)

    val labelInput = Tensor[Double](1, 3, 5, 5).apply1(e => Random.nextDouble())

    val target = dynamicLoadedModule.forward(labelInput).toTensor[Double].max(1)
      ._2.squeeze().storage().array()

    val inputTensor1 = Tensor[Double](2).apply1(e => Random.nextDouble())

    val inputTensor2 = Tensor[Double]()

    inputTensor2.resizeAs(inputTensor1).copy(inputTensor1)

    val input1 = T(inputTensor1)

    val input2 = T(inputTensor2)

    val target1 = T(Tensor[Double](Storage(target)))

    val target2 = T(Tensor[Double](Storage(target)))

    val res1 = staticCriterion.forward(input1, target1)

    val res2 = dynamicLoadedCriterion.forward(input2, target2)

    res1 should be (res2)
  }
  "Customized converter with existing type" should " throw exception " in {
    convertMap("INNERPRODUCT") = new LoadDummy[Double]
    intercept[IllegalArgumentException] {
      val (dynamicLoadedModule, dynamicLoadedCriterion) = CaffeLoader.loadCaffe(prototxt, modelPath,
        customizedConverters = convertMap)
    }
  }

}
