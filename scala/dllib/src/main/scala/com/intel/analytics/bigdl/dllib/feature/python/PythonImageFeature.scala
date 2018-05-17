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

package com.intel.analytics.zoo.feature.python

import java.util.{List => JList}

import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.zoo.feature.common.ImageProcessing
import com.intel.analytics.zoo.feature.image._
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.opencv.imgproc.Imgproc

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonImageFeature {

  def ofFloat(): PythonImageFeature[Float] = new PythonImageFeature[Float]()

  def ofDouble(): PythonImageFeature[Double] = new PythonImageFeature[Double]()
}

class PythonImageFeature[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {
  def transformImageSet(transformer: FeatureTransformer,
                          imageSet: ImageSet): ImageSet = {
    imageSet.transform(transformer)
  }

  def readImageSet(path: String, sc: JavaSparkContext, minPartitions: Int): ImageSet = {
    if (sc == null) {
      ImageSet.read(path, null, minPartitions)
    } else {
      ImageSet.read(path, sc.sc, minPartitions)
    }
  }

  def isLocalImageSet(imageSet: ImageSet): Boolean = imageSet.isLocal()

  def isDistributedImageSet(imageSet: ImageSet): Boolean = imageSet.isDistributed()

  def localImageSetToImageTensor(imageSet: LocalImageSet,
                                 floatKey: String = ImageFeature.floats,
                                 toChw: Boolean = true): JList[JTensor] = {
    imageSet.array.map(imageFeatureToImageTensor(_, floatKey, toChw)).toList.asJava
  }

  def localImageSetToLabelTensor(imageSet: LocalImageSet): JList[JTensor] = {
    imageSet.array.map(imageFeatureToLabelTensor).toList.asJava
  }

  def localImageSetToPredict(imageSet: LocalImageSet, key: String)
  : JList[JList[Any]] = {
    imageSet.array.map(x =>
      if (x.isValid && x.contains(key)) {
        List[Any](x.uri(), toJTensor(x[Tensor[T]](key))).asJava
      } else {
        List[Any](x.uri(), null).asJava
      }).toList.asJava
  }

  def distributedImageSetToImageTensorRdd(imageSet: DistributedImageSet,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JavaRDD[JTensor] = {
    imageSet.rdd.map(imageFeatureToImageTensor(_, floatKey, toChw)).toJavaRDD()
  }

  def distributedImageSetToLabelTensorRdd(imageSet: DistributedImageSet): JavaRDD[JTensor] = {
    imageSet.rdd.map(imageFeatureToLabelTensor).toJavaRDD()
  }

  def distributedImageSetToPredict(imageSet: DistributedImageSet, key: String)
  : JavaRDD[JList[Any]] = {
    imageSet.rdd.map(x => {
      if (x.isValid && x.contains(key)) {
        List[Any](x.uri(), toJTensor(x[Tensor[T]](key))).asJava
      } else {
        List[Any](x.uri(), null).asJava
      }
    })
  }

  def createDistributedImageSet(imageRdd: JavaRDD[JTensor], labelRdd: JavaRDD[JTensor])
  : DistributedImageSet = {
    require(null != imageRdd, "imageRdd cannot be null")
    val featureRdd = if (null != labelRdd) {
      imageRdd.rdd.zip(labelRdd.rdd).map(data => {
        createImageFeature(data._1, data._2)
      })
    } else {
      imageRdd.rdd.map(image => {
        createImageFeature(image, null)
      })
    }
    new DistributedImageSet(featureRdd)
  }

  def createLocalImageSet(images: JList[JTensor], labels: JList[JTensor])
  : LocalImageSet = {
    require(null != images, "images cannot be null")
    val features = if (null != labels) {
      (0 until images.size()).map(i => {
        createImageFeature(images.get(i), labels.get(i))
      })
    } else {
      (0 until images.size()).map(i => {
        createImageFeature(images.get(i), null)
      })
    }
    new LocalImageSet(features.toArray)
  }

  def createResize(resizeH: Int, resizeW: Int): Resize = {
    Resize(resizeH, resizeW)
  }

  def createImgBrightness(deltaLow: Double, deltaHigh: Double): Brightness = {
    Brightness(deltaLow, deltaHigh)
  }

  def createImgChannelNormalizer(
                                  meanR: Double, meanG: Double, meanB: Double,
                                  stdR: Double = 1, stdG: Double = 1, stdB: Double = 1
                                ): ChannelNormalize = {

    ChannelNormalize(meanR.toFloat, meanG.toFloat, meanB.toFloat,
      stdR.toFloat, stdG.toFloat, stdB.toFloat)
  }

  def createMatToTensor(): MatToTensor[T] = {
    MatToTensor()
  }

  def createCenterCrop(cropWidth: Int, cropHeight: Int): CenterCrop = {
    CenterCrop(cropWidth, cropHeight)
  }

  def createImgHue(deltaLow: Double, deltaHigh: Double): Hue = {
    Hue(deltaLow, deltaHigh)
  }

  def createImgSaturation(deltaLow: Double, deltaHigh: Double): Saturation = {
    Saturation(deltaLow, deltaHigh)
  }

  def createImgChannelOrder(): ChannelOrder = {
    ChannelOrder()
  }

  def createImgColorJitter(
                            brightnessProb: Double = 0.5, brightnessDelta: Double = 32,
                            contrastProb: Double = 0.5,
                            contrastLower: Double = 0.5, contrastUpper: Double = 1.5,
                            hueProb: Double = 0.5, hueDelta: Double = 18,
                            saturationProb: Double = 0.5,
                            saturationLower: Double = 0.5, saturationUpper: Double = 1.5,
                            randomOrderProb: Double = 0, shuffle: Boolean = false
                                ): ColorJitter = {

    ColorJitter(brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb, shuffle)
  }

  def createImgResize(resizeH: Int, resizeW: Int, resizeMode: Int = Imgproc.INTER_LINEAR,
                      useScaleFactor: Boolean): Resize = {
    Resize(resizeH, resizeW, resizeMode, useScaleFactor)
  }

  def createImgAspectScale(scale: Int,
                        scaleMultipleOf: Int,
                        maxSize: Int,
                        resizeMode: Int = 1,
                        useScaleFactor: Boolean = true,
                        minScale: Double = -1): AspectScale = {
    val minS = if (minScale == -1) None else Some(minScale.toFloat)
    AspectScale(scale, scaleMultipleOf, maxSize, resizeMode, useScaleFactor, minS)
  }

  def createImgRandomAspectScale(scales: JList[Int], scaleMultipleOf: Int = 1,
                              maxSize: Int = 1000): RandomAspectScale = {
    RandomAspectScale(scales.asScala.toArray, scaleMultipleOf, maxSize)
  }

  def createImgChannelNormalize(meanR: Double, meanG: Double, meanB: Double,
                             stdR: Double = 1, stdG: Double = 1,
                                stdB: Double = 1): ChannelNormalize = {
    ChannelNormalize(meanR.toFloat, meanG.toFloat, meanB.toFloat,
      stdR.toFloat, stdG.toFloat, stdB.toFloat)
  }

  def createImgPixelNormalize(means: JList[Double]): PixelNormalizer = {
    PixelNormalizer(means.asScala.toArray.map(_.toFloat))
  }

  def createImgRandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): RandomCrop = {
    RandomCrop(cropWidth, cropHeight, isClip)
  }

  def createImgCenterCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): CenterCrop = {
    CenterCrop(cropWidth, cropHeight, isClip)
  }

  def createImgFixedCrop(wStart: Double,
                      hStart: Double, wEnd: Double, hEnd: Double, normalized: Boolean,
                      isClip: Boolean): FixedCrop = {
    FixedCrop(wStart.toFloat, hStart.toFloat, wEnd.toFloat, hEnd.toFloat, normalized, isClip)
  }

  def createImgExpand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
                   minExpandRatio: Double = 1.0,
                   maxExpandRatio: Double = 4.0): Expand = {
    Expand(meansR, meansG, meansB, minExpandRatio, maxExpandRatio)
  }

  def createImgFiller(startX: Double, startY: Double, endX: Double, endY: Double,
                   value: Int = 255): Filler = {
    Filler(startX.toFloat, startY.toFloat, endX.toFloat, endY.toFloat, value)
  }

  def createImgHFlip(): HFlip = {
    HFlip()
  }
}
