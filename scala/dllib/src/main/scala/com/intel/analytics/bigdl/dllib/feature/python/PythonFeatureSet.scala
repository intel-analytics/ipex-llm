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

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.pmem.MemoryType
import org.apache.spark.api.java.JavaRDD

import scala.reflect.ClassTag

object PythonFeatureSet {

  def ofFloat(): PythonFeatureSet[Float] = new PythonFeatureSet[Float]()

  def ofDouble(): PythonFeatureSet[Double] = new PythonFeatureSet[Double]()
}

class PythonFeatureSet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def createFeatureSetFromImageFrame(
        imageFrame: ImageFrame,
        memoryType: String): FeatureSet[ImageFeature] = {
    require(imageFrame.isDistributed(), "Only support distributed ImageFrame")
    FeatureSet.rdd(imageFrame.toDistributed().rdd, MemoryType.fromString(memoryType))
  }

  def createFeatureSetFromRDD(
        data: JavaRDD[Any],
        memoryType: String): FeatureSet[Any] = {
    FeatureSet.rdd(data, MemoryType.fromString(memoryType))
  }

  def transformFeatureSet(featureSet: FeatureSet[Any],
                       transformer: Transformer[Any, Any]): FeatureSet[Any] = {
    featureSet -> transformer
  }

  def featureSetToDataSet(featureSet: FeatureSet[Any]): DataSet[Any] = {
    featureSet.toDataSet()
  }

}
