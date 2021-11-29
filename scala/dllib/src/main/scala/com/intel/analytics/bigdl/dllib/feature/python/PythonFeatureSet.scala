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

package com.intel.analytics.bigdl.dllib.feature.python

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.{MiniBatch, Transformer, Sample => JSample}
import com.intel.analytics.bigdl.opencv.OpenCV
import com.intel.analytics.bigdl.dllib.utils.python.api.Sample
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image._
import com.intel.analytics.bigdl.dllib.utils.Table
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import com.intel.analytics.bigdl.dllib.feature.FeatureSet
import com.intel.analytics.bigdl.dllib.feature.MemoryType
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonFeatureSet {

  def ofFloat(): PythonFeatureSet[Float] = new PythonFeatureSet[Float]()

  def ofDouble(): PythonFeatureSet[Double] = new PythonFeatureSet[Double]()

  protected def loadOpenCv(sc: SparkContext): Unit = {
    val nodeNumber = Engine.nodeNumber()
    val loadOpenCvRdd = sc.parallelize(
      Array.tabulate(nodeNumber)(_ => "dummy123123"), nodeNumber * 10)
      .mapPartitions(_ => (0 until 2000000).toIterator)
      .coalesce(nodeNumber)
      .setName("LoadOpenCvRdd")
    loadOpenCvRdd.count()
    loadOpenCvRdd.coalesce(nodeNumber).mapPartitions{v =>
      OpenCV.isOpenCVLoaded()
      v
    }.count()
    loadOpenCvRdd.unpersist()
  }
}

@SerialVersionUID(7610684191490849169L)
class PythonFeatureSet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  import PythonFeatureSet.loadOpenCv
  def createFeatureSetFromImageFrame(
        imageFrame: ImageFrame,
        memoryType: String,
        sequentialOrder: Boolean, shuffle: Boolean): FeatureSet[ImageFeature] = {
    require(imageFrame.isDistributed(), "Only support distributed ImageFrame")
    loadOpenCv(imageFrame.toDistributed().rdd.sparkContext)
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

}
