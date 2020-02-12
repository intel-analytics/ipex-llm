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

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer, Sample => JSample}
import com.intel.analytics.bigdl.python.api.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.pmem.MemoryType
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.api.java.JavaRDD

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonFeatureSet {

  def ofFloat(): PythonFeatureSet[Float] = new PythonFeatureSet[Float]()

  def ofDouble(): PythonFeatureSet[Double] = new PythonFeatureSet[Double]()
}

class PythonFeatureSet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def createFeatureSetFromImageFrame(
        imageFrame: ImageFrame,
        memoryType: String,
        sequentialOrder: Boolean, shuffle: Boolean): FeatureSet[ImageFeature] = {
    require(imageFrame.isDistributed(), "Only support distributed ImageFrame")
    FeatureSet.rdd(imageFrame.toDistributed().rdd, MemoryType.fromString(memoryType),
      sequentialOrder = sequentialOrder, shuffle = shuffle)
  }

  def createFeatureSetFromRDD(
        data: JavaRDD[Any],
        memoryType: String,
        sequentialOrder: Boolean,
        shuffle: Boolean): FeatureSet[Any] = {
    FeatureSet.rdd(data, MemoryType.fromString(memoryType),
      sequentialOrder = sequentialOrder, shuffle = shuffle)
  }

  def createSampleFeatureSetFromRDD(data: JavaRDD[Sample],
                                    memoryType: String,
                                    sequentialOrder: Boolean,
                                    shuffle: Boolean)
  : FeatureSet[JSample[T]] = {
    FeatureSet.rdd(toJSample(data),
      MemoryType.fromString(memoryType),
      sequentialOrder = sequentialOrder,
      shuffle = shuffle)
  }

  def transformFeatureSet(featureSet: FeatureSet[Any],
                       transformer: Transformer[Any, Any]): FeatureSet[Any] = {
    featureSet -> transformer
  }

  def featureSetToDataSet(featureSet: FeatureSet[Any]): DataSet[Any] = {
    featureSet.toDataSet()
  }

  def createFeatureSetFromTfDataset(
        dataset: Array[Byte],
        totalSize: Int): FeatureSet[MiniBatch[Float]] = {
    val nodeNumber = EngineRef.getNodeNumber()
    // set a random seed to make sure shuffle is the same in each executor
    val imports =
      s"""
        |import tensorflow as tf
        |from zoo.util.nest import flatten
        |sess = tf.Session()
        |""".stripMargin
    def getIterator(iterName: String, loaderName: String): String = {
      s"""
         |${iterName} = ${loaderName}.make_one_shot_iterator()
         |""".stripMargin
    }
    def getNext(iterName: String): String = {
      s"""
        |data = sess.run(${iterName}.get_next())
        |data = flatten(data)
        |""".stripMargin
    }
    FeatureSet.python[MiniBatch[Float]](dataset,
      getIterator, getNext,
      "data", "", totalSize, imports)
  }

}
