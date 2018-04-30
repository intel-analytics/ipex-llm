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

package com.intel.analytics.zoo.pipeline.nnframes.python

import java.util.{ArrayList => JArrayList, List => JList}

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.FeatureTransformer
import com.intel.analytics.zoo.pipeline.nnframes._
import com.intel.analytics.zoo.pipeline.nnframes.transformers._
import com.intel.analytics.zoo.pipeline.nnframes.transformers.internal.FeatureToTupleAdapter
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonNNFrames {

  def ofFloat(): PythonNNFrames[Float] = new PythonNNFrames[Float]()

  def ofDouble(): PythonNNFrames[Double] = new PythonNNFrames[Double]()
}

class PythonNNFrames[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def nnReadImage(path: String, sc: JavaSparkContext, minParitions: Int): DataFrame = {
    NNImageReader.readImages(path, sc.sc, minParitions)
  }

  def createNNImageTransformer(transformer: FeatureTransformer): NNImageTransformer = {
    new NNImageTransformer(transformer)
  }

  def createNNEstimator(
      model: Module[T],
      criterion: Criterion[T],
      sampleTransformer: Transformer[(Any, Any), Sample[T]]
    ): NNEstimator[Any, Any, T] = {
    new NNEstimator(model, criterion, sampleTransformer)
  }

  def createNNClassifier(
        model: Module[T],
        criterion: Criterion[T],
        featureTransformer: Transformer[Any, Tensor[T]]): NNClassifier[Any, T] = {
    new NNClassifier(model, criterion, featureTransformer)
  }

  def createNNModel(
      model: Module[T],
      sampleTransformer: Transformer[Any, Sample[T]]): NNModel[Any, T] = {
    new NNModel(model, sampleTransformer)
  }

  def createNNClassifierModel(
      model: Module[T],
      featureTransformer: Transformer[Any, Tensor[T]]): NNClassifierModel[Any, T] = {
    new NNClassifierModel(model, featureTransformer)
  }

  def setOptimMethod(
      estimator: NNEstimator[Any, Any, T],
      optimMethod: OptimMethod[T]): NNEstimator[Any, Any, T] = {
    estimator.setOptimMethod(optimMethod)
  }

  def withOriginColumn(imageDF: DataFrame, imageColumn: String, originColumn: String): DataFrame = {
    NNImageSchema.withOriginColumn(imageDF, imageColumn, originColumn)
  }

  def createNumToTensor(): NumToTensor[T] = {
    new NumToTensor()
  }

  def createSeqToTensor(size: JArrayList[Int]): SeqToTensor[T] = {
    SeqToTensor(size.asScala.toArray)
  }

  def createMLlibVectorToTensor(size: JArrayList[Int]): MLlibVectorToTensor[T] = {
    MLlibVectorToTensor(size.asScala.toArray)
  }

  def createImageFeatureToTensor(): ImageFeatureToTensor[T] = {
    ImageFeatureToTensor()
  }

  def createRowToImageFeature(): RowToImageFeature[T] = {
    RowToImageFeature()
  }

  def createFeatureLabelTransformer(
      featureTransfomer: Transformer[Any, Tensor[T]],
      labelTransformer: Transformer[Any, Tensor[T]]
    ): FeatureLabelTransformer[Any, Any, Sample[T]] = {
    FeatureLabelTransformer(featureTransfomer, labelTransformer)
      .asInstanceOf[FeatureLabelTransformer[Any, Any, Sample[T]]]
  }

  def createChainedTransformer(list: JList[Transformer[Any, Any]]): Transformer[Any, Any] = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }

  def createTensorToSample(): TensorToSample[T] = {
    TensorToSample()
  }

  def createFeatureToTupleAdapter(
      sampleTransformer: Transformer[(Any, Any), Sample[T]]
    ): FeatureToTupleAdapter[Any, Sample[T]] = {
    FeatureToTupleAdapter(sampleTransformer).asInstanceOf[FeatureToTupleAdapter[Any, Sample[T]]]
  }
}
