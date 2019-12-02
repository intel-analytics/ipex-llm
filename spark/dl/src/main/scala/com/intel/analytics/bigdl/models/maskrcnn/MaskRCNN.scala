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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.segmentation.{MaskUtils, RLEMasks}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.RoiImageInfo
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{T, Table}
import scala.reflect.ClassTag
import scala.reflect.runtime._

case class MaskRCNNParams(
  anchorSizes: Array[Float] = Array[Float](32, 64, 128, 256, 512),
  aspectRatios: Array[Float] = Array[Float](0.5f, 1.0f, 2.0f),
  anchorStride: Array[Float] = Array[Float](4, 8, 16, 32, 64),
  preNmsTopNTest: Int = 1000,
  postNmsTopNTest: Int = 1000,
  preNmsTopNTrain: Int = 2000,
  postNmsTopNTrain: Int = 2000,
  rpnNmsThread: Float = 0.7f,
  minSize: Int = 0,
  boxResolution: Int = 7,
  maskResolution: Int = 14,
  scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f),
  samplingRatio: Int = 2,
  boxScoreThresh: Float = 0.05f,
  boxNmsThread: Float = 0.5f,
  maxPerImage: Int = 100,
  outputSize: Int = 1024,
  layers: Array[Int] = Array[Int](256, 256, 256, 256),
  dilation: Int = 1,
  useGn: Boolean = false)

class MaskRCNN(val inChannels: Int,
               val outChannels: Int,
               val numClasses: Int = 81,
               val config: MaskRCNNParams = new MaskRCNNParams)(implicit ev: TensorNumeric[Float])
  extends Container[Activity, Activity, Float] {

  private val batchImgInfo : Tensor[Float] = Tensor[Float](2)
  initModules()
  // add layer to modules
  private def initModules(): Unit = {
      modules.clear()
      val backbone = buildBackbone(inChannels, outChannels)
      val rpn = RegionProposal(inChannels, config.anchorSizes, config.aspectRatios,
        config.anchorStride, config.preNmsTopNTest, config.postNmsTopNTest, config.preNmsTopNTrain,
        config.postNmsTopNTrain, config.rpnNmsThread, config.minSize)
      val boxHead = BoxHead(inChannels, config.boxResolution, config.scales,
        config.samplingRatio, config.boxScoreThresh, config.boxNmsThread, config.maxPerImage,
        config.outputSize, numClasses)
      val maskHead = MaskHead(inChannels, config.maskResolution, config.scales,
        config.samplingRatio, config.layers, config.dilation, numClasses)

      modules.append(backbone.asInstanceOf[Module[Float]])
      modules.append(rpn.asInstanceOf[Module[Float]])
      modules.append(boxHead.asInstanceOf[Module[Float]])
      modules.append(maskHead.asInstanceOf[Module[Float]])
    }

  private def buildResNet50(): Module[Float] = {

    def convolution (nInputPlane: Int, nOutputPlane: Int, kernelW: Int, kernelH: Int,
      strideW: Int = 1, strideH: Int = 1, padW: Int = 0, padH: Int = 0,
      nGroup: Int = 1, propagateBack: Boolean = true): SpatialConvolution[Float] = {
        val conv = SpatialConvolution[Float](nInputPlane, nOutputPlane, kernelW, kernelH,
          strideW, strideH, padW, padH, nGroup, propagateBack, withBias = false)
        conv.setInitMethod(MsraFiller(false), Zeros)
        conv
      }

    def sbn(nOutput: Int, eps: Double = 1e-3, momentum: Double = 0.1, affine: Boolean = true)
      : SpatialBatchNormalization[Float] = {
        SpatialBatchNormalization[Float](nOutput, eps, momentum, affine).setInitMethod(Ones, Zeros)
      }

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                 useConv: Boolean = false): Module[Float] = {
      if (useConv) {
        Sequential()
          .add(convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
          .add(sbn(nOutputPlane))
      } else {
        Identity()
      }
    }

    def bottleneck(nInputPlane: Int, internalPlane: Int, nOutputPlane: Int,
                   stride: Int, useConv: Boolean = false): Module[Float] = {
      val s = Sequential()
        .add(convolution(nInputPlane, internalPlane, 1, 1, stride, stride, 0, 0))
        .add(sbn(internalPlane))
        .add(ReLU(true))
        .add(convolution(internalPlane, internalPlane, 3, 3, 1, 1, 1, 1))
        .add(sbn(internalPlane))
        .add(ReLU(true))
        .add(convolution(internalPlane, nOutputPlane, 1, 1, 1, 1, 0, 0))
        .add(sbn(nOutputPlane))

      val m = Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, nOutputPlane, stride, useConv)))
        .add(CAddTable(true))
        .add(ReLU(true))
      m
    }

    def layer(count: Int, nInputPlane: Int, nOutputPlane: Int,
              downOutputPlane: Int, stride: Int = 1): Module[Float] = {
      val s = Sequential()
        .add(bottleneck(nInputPlane, nOutputPlane, downOutputPlane, stride, true))
      for (i <- 2 to count) {
        s.add(bottleneck(downOutputPlane, nOutputPlane, downOutputPlane, 1, false))
      }
      s
    }

    val model = Sequential[Float]()
      .add(convolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = false))
      .add(sbn(64))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))

    val input = Input()
    val node0 = model.inputs(input)

    val startChannels = 64
    val node1 = layer(3, startChannels, 64, inChannels, 1).inputs(node0)
    val node2 = layer(4, inChannels, 128, inChannels * 2, 2).inputs(node1)
    val node3 = layer(6, inChannels * 2, 256, inChannels * 4, 2).inputs(node2)
    val node4 = layer(3, inChannels * 4, 512, inChannels * 8, 2).inputs(node3)

    Graph(input, Array(node1, node2, node3, node4))
  }

  private def buildBackbone(inChannels: Int, outChannels: Int): Module[Float] = {
    val resnet = buildResNet50()
    val inChannelList = Array(inChannels, inChannels*2, inChannels * 4, inChannels * 8)
    val fpn = FPN(inChannelList, outChannels, topBlocks = 1)
    val model = Sequential[Float]().add(resnet).add(fpn)
    model
  }

  override def updateOutput(input: Activity): Activity = {
    val inputFeatures = input.toTable[Tensor[Float]](1)
    // image info with shape (batchSize, 4)
    // contains all images info (height, width, original height, original width)
    val imageInfo = input.toTable[Tensor[Float]](2)

    // get each layer from modules
    val backbone = modules(0)
    val rpn = modules(1)
    val boxHead = modules(2)
    val maskHead = modules(3)

    batchImgInfo.setValue(1, inputFeatures.size(3))
    batchImgInfo.setValue(2, inputFeatures.size(4))

    val features = backbone.forward(inputFeatures)
    val proposals = rpn.forward(T(features, batchImgInfo))
    val boxOutput = boxHead.forward(T(features, proposals, batchImgInfo)).toTable
    val postProcessorBox = boxOutput[Table](2)
    val labelsBox = postProcessorBox[Tensor[Float]](1)
    val proposalsBox = postProcessorBox[Table](2)
    val scores = postProcessorBox[Tensor[Float]](3)
    if (labelsBox.size(1) > 0) {
      val masks = maskHead.forward(T(features, proposalsBox, labelsBox)).toTable
      if (this.isTraining()) {
        output = T(proposalsBox, labelsBox, masks, scores)
      } else {
        output = postProcessorForMaskRCNN(proposalsBox, labelsBox, masks[Tensor[Float]](2),
          scores, imageInfo)
      }
    } else { // detect nothing
      for (i <- 1 to inputFeatures.size(1)) {
        output.toTable(i) = T()
      }
    }

    output
  }

  @transient var binaryMask : Tensor[Float] = null
  private def postProcessorForMaskRCNN(bboxes: Table, labels: Tensor[Float],
    masks: Tensor[Float], scores: Tensor[Float], imageInfo: Tensor[Float]): Table = {
    val batchSize = bboxes.length()
    val boxesInImage = new Array[Int](batchSize)
    for (i <- 0 to batchSize - 1) {
      boxesInImage(i) = bboxes[Tensor[Float]](i + 1).size(1)
    }

    if (binaryMask == null) binaryMask = Tensor[Float]()
    val output = T()
    var start = 1
    for (i <- 0 to batchSize - 1) {
      val info = imageInfo.select(1, i + 1)
      val height = info.valueAt(1).toInt // image height after scale, no padding
      val width = info.valueAt(2).toInt // image width after scale, no padding
      val originalHeight = info.valueAt(3).toInt // Original height
      val originalWidth = info.valueAt(4).toInt // Original width

      binaryMask.resize(originalHeight, originalWidth)

      // prepare for evaluation
      val postOutput = T()

      val boxNumber = boxesInImage(i)
      if (boxNumber > 0) {
        val maskPerImg = masks.narrow(1, start, boxNumber)
        val bboxPerImg = bboxes[Tensor[Float]](i + 1)
        val classPerImg = labels.narrow(1, start, boxNumber)
        val scorePerImg = scores.narrow(1, start, boxNumber)

        require(maskPerImg.size(1) == bboxPerImg.size(1), s"mask number ${maskPerImg.size(1)} " +
          s"should be the same with box number ${bboxPerImg.size(1)}")

        // resize bbox to original size
        if (height != originalHeight || width != originalWidth) {
          BboxUtil.scaleBBox(bboxPerImg,
            originalHeight.toFloat / height, originalWidth.toFloat / width)
        }
        // decode mask to original size
        val masksRLE = new Array[RLEMasks](boxNumber)
        for (j <- 0 to boxNumber - 1) {
          binaryMask.fill(0.0f)
          Utils.decodeMaskInImage(maskPerImg.select(1, j + 1), bboxPerImg.select(1, j + 1),
            binaryMask = binaryMask)
          masksRLE(j) = MaskUtils.binaryToRLE(binaryMask)
        }
        start += boxNumber

        postOutput.update(RoiImageInfo.MASKS, masksRLE)
        postOutput.update(RoiImageInfo.BBOXES, bboxPerImg)
        postOutput.update(RoiImageInfo.CLASSES, classPerImg)
        postOutput.update(RoiImageInfo.SCORES, scorePerImg)
      }

      output(i + 1) = postOutput
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    throw new UnsupportedOperationException("MaskRCNN model only support inference now")
  }
}

object MaskRCNN extends ContainerSerializable {
  def apply(inChannels: Int, outChannels: Int, numClasses: Int = 81,
    config: MaskRCNNParams = new MaskRCNNParams)(implicit ev: TensorNumeric[Float]): MaskRCNN =
    new MaskRCNN(inChannels, outChannels, numClasses, config)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap

    val inChannels = DataConverter
      .getAttributeValue(context, attrMap.get("inChannels")).
      asInstanceOf[Int]

    val outChannels = DataConverter
      .getAttributeValue(context, attrMap.get("outChannels"))
      .asInstanceOf[Int]

    val numClasses = DataConverter
      .getAttributeValue(context, attrMap.get("numClasses"))
      .asInstanceOf[Int]

    // get MaskRCNNParams
    val config = MaskRCNNParams(
    anchorSizes = DataConverter
      .getAttributeValue(context, attrMap.get("anchorSizes"))
      .asInstanceOf[Array[Float]],
    aspectRatios = DataConverter
      .getAttributeValue(context, attrMap.get("aspectRatios"))
      .asInstanceOf[Array[Float]],
    anchorStride = DataConverter
      .getAttributeValue(context, attrMap.get("anchorStride"))
      .asInstanceOf[Array[Float]],
    preNmsTopNTest = DataConverter
      .getAttributeValue(context, attrMap.get("preNmsTopNTest"))
      .asInstanceOf[Int],
    postNmsTopNTest = DataConverter
      .getAttributeValue(context, attrMap.get("postNmsTopNTest"))
      .asInstanceOf[Int],
    preNmsTopNTrain = DataConverter
      .getAttributeValue(context, attrMap.get("preNmsTopNTrain"))
      .asInstanceOf[Int],
    postNmsTopNTrain = DataConverter
      .getAttributeValue(context, attrMap.get("postNmsTopNTrain"))
      .asInstanceOf[Int],
    rpnNmsThread = DataConverter
      .getAttributeValue(context, attrMap.get("rpnNmsThread"))
      .asInstanceOf[Float],
    minSize = DataConverter
      .getAttributeValue(context, attrMap.get("minSize"))
      .asInstanceOf[Int],
    boxResolution = DataConverter
      .getAttributeValue(context, attrMap.get("boxResolution"))
      .asInstanceOf[Int],
    maskResolution = DataConverter
      .getAttributeValue(context, attrMap.get("maskResolution"))
      .asInstanceOf[Int],
    scales = DataConverter
      .getAttributeValue(context, attrMap.get("scales"))
      .asInstanceOf[Array[Float]],
    samplingRatio = DataConverter
      .getAttributeValue(context, attrMap.get("samplingRatio"))
      .asInstanceOf[Int],
    boxScoreThresh = DataConverter
      .getAttributeValue(context, attrMap.get("boxScoreThresh"))
      .asInstanceOf[Float],
    maxPerImage = DataConverter
      .getAttributeValue(context, attrMap.get("maxPerImage"))
      .asInstanceOf[Int],
    outputSize = DataConverter
      .getAttributeValue(context, attrMap.get("outputSize"))
      .asInstanceOf[Int],
    layers = DataConverter
      .getAttributeValue(context, attrMap.get("layers"))
      .asInstanceOf[Array[Int]],
    dilation = DataConverter
      .getAttributeValue(context, attrMap.get("dilation"))
      .asInstanceOf[Int],
    useGn = DataConverter
      .getAttributeValue(context, attrMap.get("useGn"))
      .asInstanceOf[Boolean])

    val maskrcnn = MaskRCNN(inChannels, outChannels, numClasses, config)
      .asInstanceOf[Container[Activity, Activity, T]]
    maskrcnn.modules.clear()
    loadSubModules(context, maskrcnn)

    maskrcnn
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    maskrcnnBuilder : BigDLModule.Builder)(implicit ev: TensorNumeric[T]) : Unit = {

    val maskrcnn = context.moduleData.module.asInstanceOf[MaskRCNN]

    val inChannelsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, inChannelsBuilder, maskrcnn.inChannels,
      universe.typeOf[Int])
    maskrcnnBuilder.putAttr("inChannels", inChannelsBuilder.build)

    val outChannelsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outChannelsBuilder, maskrcnn.outChannels,
      universe.typeOf[Int])
    maskrcnnBuilder.putAttr("outChannels", outChannelsBuilder.build)

    val numClassesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, numClassesBuilder, maskrcnn.numClasses,
      universe.typeOf[Int])
    maskrcnnBuilder.putAttr("numClasses", numClassesBuilder.build)

    // put MaskRCNNParams
    val config = maskrcnn.config

    val anchorSizesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, anchorSizesBuilder,
      config.anchorSizes, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("anchorSizes", anchorSizesBuilder.build)

    val aspectRatiosBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, aspectRatiosBuilder,
      config.aspectRatios, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("aspectRatios", aspectRatiosBuilder.build)

    val anchorStrideBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, anchorStrideBuilder,
      config.anchorStride, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("anchorStride", anchorStrideBuilder.build)

    val preNmsTopNTestBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preNmsTopNTestBuilder,
      config.preNmsTopNTest, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("preNmsTopNTest", preNmsTopNTestBuilder.build)

    val postNmsTopNTestBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, postNmsTopNTestBuilder,
      config.postNmsTopNTest, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("postNmsTopNTest", postNmsTopNTestBuilder.build)

    val preNmsTopNTrainBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preNmsTopNTrainBuilder,
      config.preNmsTopNTrain, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("preNmsTopNTrain", preNmsTopNTrainBuilder.build)

    val postNmsTopNTrainBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, postNmsTopNTrainBuilder,
      config.postNmsTopNTrain, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("postNmsTopNTrain", postNmsTopNTrainBuilder.build)

    val rpnNmsThreadBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, rpnNmsThreadBuilder,
      config.rpnNmsThread, universe.typeOf[Float])
    maskrcnnBuilder.putAttr("rpnNmsThread", rpnNmsThreadBuilder.build)

    val minSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, minSizeBuilder,
      config.minSize, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("minSize", minSizeBuilder.build)

    val boxResolutionBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, boxResolutionBuilder,
      config.boxResolution, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("boxResolution", boxResolutionBuilder.build)

    val maskResolutionBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maskResolutionBuilder,
      config.maskResolution, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("maskResolution", maskResolutionBuilder.build)

    val scalesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, scalesBuilder,
      config.scales, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("scales", scalesBuilder.build)

    val samplingRatioBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, samplingRatioBuilder,
      config.samplingRatio, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("samplingRatio", samplingRatioBuilder.build)

    val boxScoreThreshBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, boxScoreThreshBuilder,
      config.boxScoreThresh, universe.typeOf[Float])
    maskrcnnBuilder.putAttr("boxScoreThresh", boxScoreThreshBuilder.build)

    val maxPerImageBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maxPerImageBuilder,
      config.maxPerImage, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("maxPerImage", maxPerImageBuilder.build)

    val outputSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outputSizeBuilder,
      config.outputSize, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("outputSize", outputSizeBuilder.build)

    val layersBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, layersBuilder,
      config.layers, universe.typeOf[Array[Int]])
    maskrcnnBuilder.putAttr("layers", layersBuilder.build)

    val dilationBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, dilationBuilder,
      config.dilation, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("dilation", dilationBuilder.build)

    val useGnBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, useGnBuilder,
      config.useGn, universe.typeOf[Boolean])
    maskrcnnBuilder.putAttr("useGn", useGnBuilder.build)

    serializeSubModules(context, maskrcnnBuilder)
  }
}
