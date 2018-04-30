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

package com.intel.analytics.zoo.feature.image.python

import java.util.{List => JList}

import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, ImageFrame}
import com.intel.analytics.zoo.feature.image.{DistributedImageSet, ImageSet, LocalImageSet}
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonImageSet {

  def ofFloat(): PythonImageSet[Float] = new PythonImageSet[Float]()

  def ofDouble(): PythonImageSet[Double] = new PythonImageSet[Double]()
}

class PythonImageSet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {
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
}
