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

import java.util
import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.python.api.JTensor
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.image3d._
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonImageFeature {

  def ofFloat(): PythonImageFeature[Float] = new PythonImageFeature[Float]()

  def ofDouble(): PythonImageFeature[Double] = new PythonImageFeature[Double]()
}

class PythonImageFeature[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def transformImageSet(transformer: Preprocessing[ImageFeature, ImageFeature],
                      imageSet: ImageSet): ImageSet = {
    imageSet.transform(transformer)
  }

  def transformImageSet(transformer: ImageProcessing3D,
                        imageSet: ImageSet): ImageSet = {
    imageSet.transform(transformer)
  }

  def readImageSet(path: String, sc: JavaSparkContext, minPartitions: Int,
                   resizeH: Int, resizeW: Int, imageCodec: Int,
                   withLabel: Boolean, oneBasedLabel: Boolean): ImageSet = {
    if (sc == null) {
      ImageSet.read(path, null, minPartitions, resizeH, resizeW,
        imageCodec, withLabel, oneBasedLabel)
    } else {
      ImageSet.read(path, sc.sc, minPartitions, resizeH, resizeW,
        imageCodec, withLabel, oneBasedLabel)
    }
  }

  def imageSetGetLabelMap(imageSet: ImageSet): util.Map[String, Int] = {
    if (imageSet.labelMap.isEmpty) {
      null
    } else {
      imageSet.labelMap.get.asJava
    }
  }

  def isLocalImageSet(imageSet: ImageSet): Boolean = imageSet.isLocal()

  def isDistributedImageSet(imageSet: ImageSet): Boolean = imageSet.isDistributed()

  def localImageSetToImageTensor(imageSet: LocalImageSet,
                                 floatKey: String = ImageFeature.floats,
                                 toChw: Boolean = true): JList[JTensor] = {
    imageSet.array.map(imf => {
      if (imf.getSize == null) {
        imageFeature3DToImageTensor(imf, floatKey)
      } else {
        toTensor(imf, toChw)
      }
    }).toList.asJava
  }

  def imageFeature3DToImageTensor(imageFeature: ImageFeature,
                                  tensorKey: String = ImageFeature.imageTensor): JTensor = {
    toJTensor(imageFeature(tensorKey).asInstanceOf[Tensor[T]])
  }

  def toTensor(imf: ImageFeature, toChw: Boolean = true): JTensor = {
    val (data, size) = if (imf.contains(ImageFeature.floats)) {
      (imf.floats(),
        Array(imf.getHeight(), imf.getWidth(), imf.getChannel()))
    } else {
      val mat = imf.opencvMat()
      val floats = new Array[Float](mat.height() * mat.width() * imf.getChannel())
      OpenCVMat.toFloatPixels(mat, floats)
      (floats, Array(mat.height(), mat.width(), imf.getChannel()))
    }
    var image = Tensor(Storage(data)).resize(size)
    if (toChw) {
      // transpose the shape of image from (h, w, c) to (c, h, w)
      image = image.transpose(1, 3).transpose(2, 3).contiguous()
    }
    toJTensor(image.asInstanceOf[Tensor[T]])
  }

  def localImageSetToLabelTensor(imageSet: LocalImageSet): JList[JTensor] = {
    imageSet.array.map(imageFeatureToLabelTensor).toList.asJava
  }

  def localImageSetToPredict(imageSet: LocalImageSet, key: String)
  : JList[JList[Any]] = {
    imageSet.array.map(x => imageSetToPredict(x, key)).toList.asJava
  }

  def distributedImageSetToImageTensorRdd(imageSet: DistributedImageSet,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JavaRDD[JTensor] = {
    imageSet.rdd.map(imf => {
      // 3D image
      if (imf.getSize == null) {
        imageFeature3DToImageTensor(imf, floatKey)
      } else toTensor(imf, toChw)
    }).toJavaRDD()
  }

  def distributedImageSetToLabelTensorRdd(imageSet: DistributedImageSet): JavaRDD[JTensor] = {
    imageSet.rdd.map(imageFeatureToLabelTensor).toJavaRDD()
  }

  def distributedImageSetToPredict(imageSet: DistributedImageSet, key: String)
  : JavaRDD[JList[Any]] = {
    imageSet.rdd.map(x => imageSetToPredict(x, key))
  }

  private def imageSetToPredict(imf: ImageFeature, key: String): JList[Any] = {
    if (imf.isValid && imf.contains(key)) {
        List[Any](imf.uri(), activityToJTensors(imf(key))).asJava
    } else {
      List[Any](imf.uri(), null).asJava
    }
  }

  def createDistributedImageSet(imageRdd: JavaRDD[JTensor], labelRdd: JavaRDD[JTensor])
  : DistributedImageSet = {
    require(null != imageRdd, "imageRdd cannot be null")
    val featureRdd = if (null != labelRdd) {
      imageRdd.rdd.zip(labelRdd.rdd).map(data => {
        if (data._1.shape.length == 4) {
          createImageFeature3D(data._1, data._2)
        } else {
          createImageFeature(data._1, data._2)
        }
      })
    } else {
      imageRdd.rdd.map(image => {
        if (image.shape.length == 4) {
          createImageFeature3D(image, null)
        } else {
          createImageFeature(image, null)
        }
      })
    }
    new DistributedImageSet(featureRdd)
  }

  def createLocalImageSet(images: JList[JTensor], labels: JList[JTensor])
  : LocalImageSet = {
    require(null != images, "images cannot be null")
    val features = if (null != labels) {
      (0 until images.size()).map(i => {
        val img = images.get(i)
        if (img.shape.length == 3) {
          createImageFeature(img, labels.get(i))
        } else {
          createImageFeature3D(img, labels.get(i))
        }
      })
    } else {
      (0 until images.size()).map(i => {
        val img = images.get(i)
        if (img.shape.length == 3) {
          createImageFeature(img, null)
        } else {
          createImageFeature3D(img, null)
        }
      })
    }
    new LocalImageSet(features.toArray)
  }

  def createImageFeature3D(data: JTensor = null, label: JTensor = null, uri: String = null)
  : ImageFeature = {
    val feature = new ImageFeature3D()
    if (null != data) {
      feature(ImageFeature.imageTensor) = toTensor(data)
      feature(ImageFeature.size) = data.shape
    }
    if (null != label) {
      // todo: may need a method to change label format if needed
      feature(ImageFeature.label) = toTensor(label)
    }
    if (null != uri) {
      feature(ImageFeature.uri) = uri
    }
    feature
  }

  def createImageBytesToMat(
      byteKey: String = ImageFeature.bytes,
      imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): ImageBytesToMat = {
    ImageBytesToMat(byteKey, imageCodec)
  }

  def createImagePixelBytesToMat(
      byteKey: String = ImageFeature.bytes): ImagePixelBytesToMat = {
    ImagePixelBytesToMat(byteKey)
  }

  def createImageBrightness(deltaLow: Double, deltaHigh: Double): ImageBrightness = {
    ImageBrightness(deltaLow, deltaHigh)
  }

  def createImageFeatureToTensor(): ImageFeatureToTensor[T] = {
    ImageFeatureToTensor()
  }

  def createImageFeatureToSample(): ImageFeatureToSample[T] = {
    ImageFeatureToSample()
  }

  def createImageChannelNormalizer(
                                  meanR: Double, meanG: Double, meanB: Double,
                                  stdR: Double = 1, stdG: Double = 1, stdB: Double = 1
                                ): ImageChannelNormalize = {

    ImageChannelNormalize(meanR.toFloat, meanG.toFloat, meanB.toFloat,
      stdR.toFloat, stdG.toFloat, stdB.toFloat)
  }

  def createPerImageNormalize(min: Double, max: Double, normType: Int = 32): PerImageNormalize = {
    PerImageNormalize(min, max, normType)
  }

  def createImageMatToTensor(toRGB: Boolean = false,
                             tensorKey: String = ImageFeature.imageTensor,
                             shareBuffer: Boolean = true,
                             format: String = "NCHW"): ImageMatToTensor[T] = {
    format match {
      case "NCHW" => ImageMatToTensor(toRGB, tensorKey, shareBuffer, DataFormat.NCHW)
      case "NHWC" => ImageMatToTensor(toRGB, tensorKey, shareBuffer, DataFormat.NHWC)
      case other => throw new IllegalArgumentException(s"Unsupported format:" +
        s" $format. Only NCHW and NHWC are supported.")
    }
  }

  def createImageHue(deltaLow: Double, deltaHigh: Double): ImageHue = {
    ImageHue(deltaLow, deltaHigh)
  }

  def createImageSaturation(deltaLow: Double, deltaHigh: Double): ImageSaturation = {
    ImageSaturation(deltaLow, deltaHigh)
  }

  def createImageChannelOrder(): ImageChannelOrder = {
    ImageChannelOrder()
  }

  def createImageColorJitter(
                            brightnessProb: Double = 0.5, brightnessDelta: Double = 32,
                            contrastProb: Double = 0.5,
                            contrastLower: Double = 0.5, contrastUpper: Double = 1.5,
                            hueProb: Double = 0.5, hueDelta: Double = 18,
                            saturationProb: Double = 0.5,
                            saturationLower: Double = 0.5, saturationUpper: Double = 1.5,
                            randomOrderProb: Double = 0, shuffle: Boolean = false
                                ): ImageColorJitter = {

    ImageColorJitter(brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb, shuffle)
  }

  def createImageResize(resizeH: Int, resizeW: Int, resizeMode: Int = Imgproc.INTER_LINEAR,
                      useScaleFactor: Boolean): ImageResize = {
    ImageResize(resizeH, resizeW, resizeMode, useScaleFactor)
  }

  def createImageAspectScale(scale: Int,
                        scaleMultipleOf: Int,
                        maxSize: Int,
                        resizeMode: Int = 1,
                        useScaleFactor: Boolean = true,
                        minScale: Double = -1): ImageAspectScale = {
    val minS = if (minScale == -1) None else Some(minScale.toFloat)
    ImageAspectScale(scale, scaleMultipleOf, maxSize, resizeMode, useScaleFactor, minS)
  }

  def createImageRandomAspectScale(scales: JList[Int], scaleMultipleOf: Int = 1,
                              maxSize: Int = 1000): ImageRandomAspectScale = {
    ImageRandomAspectScale(scales.asScala.toArray, scaleMultipleOf, maxSize)
  }

  def createImageChannelNormalize(meanR: Double, meanG: Double, meanB: Double,
                             stdR: Double = 1, stdG: Double = 1,
                                stdB: Double = 1): ImageChannelNormalize = {
    ImageChannelNormalize(meanR.toFloat, meanG.toFloat, meanB.toFloat,
      stdR.toFloat, stdG.toFloat, stdB.toFloat)
  }

  def createImagePixelNormalize(means: JList[Double]): ImagePixelNormalizer = {
    ImagePixelNormalizer(means.asScala.toArray.map(_.toFloat))
  }

  def createImageRandomPreprocessing(
      preprocessing: ImageProcessing,
      prob: Double
    ): ImageRandomPreprocessing = {
    ImageRandomPreprocessing(preprocessing, prob)
  }

  def createImageRandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): ImageRandomCrop = {
    ImageRandomCrop(cropWidth, cropHeight, isClip)
  }

  def createImageCenterCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): ImageCenterCrop = {
    ImageCenterCrop(cropWidth, cropHeight, isClip)
  }

  def createImageFixedCrop(wStart: Double,
                      hStart: Double, wEnd: Double, hEnd: Double, normalized: Boolean,
                      isClip: Boolean): ImageFixedCrop = {
    ImageFixedCrop(wStart.toFloat, hStart.toFloat, wEnd.toFloat, hEnd.toFloat, normalized, isClip)
  }

  def createImageExpand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
                   minExpandRatio: Double = 1.0,
                   maxExpandRatio: Double = 4.0): ImageExpand = {
    ImageExpand(meansR, meansG, meansB, minExpandRatio, maxExpandRatio)
  }

  def createImageFiller(startX: Double, startY: Double, endX: Double, endY: Double,
                   value: Int = 255): ImageFiller = {
    ImageFiller(startX.toFloat, startY.toFloat, endX.toFloat, endY.toFloat, value)
  }

  def createImageHFlip(): ImageHFlip = {
    ImageHFlip()
  }

  def createImageMirror(): ImageMirror = {
    ImageMirror()
  }

  def createImageSetToSample(inputKeys: JList[String],
                             targetKeys: JList[String],
                             sampleKey: String): ImageSetToSample[T] = {
    val targets = if (targetKeys == null) null else targetKeys.asScala.toArray
    ImageSetToSample[T](inputKeys.asScala.toArray, targets, sampleKey)
  }

  def imageSetToImageFrame(imageSet: ImageSet): ImageFrame = {
    imageSet.toImageFrame()
  }

  def imageFrameToImageSet(imageFrame: ImageFrame): ImageSet = {
    ImageSet.fromImageFrame(imageFrame)
  }

  def createCrop3D(start: JList[Int], patchSize: JList[Int]): Crop3D = {
    Crop3D(start.asScala.toArray, patchSize.asScala.toArray)
  }

  def createRandomCrop3D(cropDepth: Int, cropHeight: Int, cropWidth: Int): RandomCrop3D = {
    RandomCrop3D(cropDepth, cropHeight, cropWidth)
  }

  def createCenterCrop3D(cropDepth: Int, cropHeight: Int, cropWidth: Int): CenterCrop3D = {
    CenterCrop3D(cropDepth, cropHeight, cropWidth)
  }

  def createRotate3D(rotationAngles: JList[Double]): Rotate3D = {
    Rotate3D(rotationAngles.asScala.toArray)
  }

  def createAffineTransform3D(mat: JTensor, translation: JTensor,
                            clamp_mode: String, pad_val: Double): AffineTransform3D = {
    AffineTransform3D(toDoubleTensor(mat), toDoubleTensor(translation), clamp_mode, pad_val)
  }

  def toDoubleTensor(jTensor: JTensor): Tensor[Double] = {
    val tensor = if (jTensor == null) null else {
      Tensor(storage = Storage[Double](jTensor.storage.map(_.asInstanceOf[Double])),
        storageOffset = 1,
        size = jTensor.shape)
    }
    tensor
  }
}
