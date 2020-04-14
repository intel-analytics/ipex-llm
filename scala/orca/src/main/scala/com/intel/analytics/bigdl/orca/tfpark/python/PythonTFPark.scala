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
package com.intel.analytics.zoo.tfpark.python

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.{PythonZoo, RDDWrapper}
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.tfpark._
import org.apache.spark.api.java.JavaRDD

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import java.util.{List => JList}

import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.tensorflow.DataType


object PythonTFPark {

  def ofFloat(): PythonTFPark[Float] = new PythonTFPark[Float]()

  def ofDouble(): PythonTFPark[Double] = new PythonTFPark[Double]()

}


class PythonTFPark[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def createTFTrainingHelper(modelPath: String, config: Array[Byte] = null): Module[Float] = {
    TFTrainingHelper(modelPath, config)
  }

  def saveCheckpoint(model: TFTrainingHelper): Unit = {
    model.saveCheckpoint()
  }

  def createIdentityCriterion(): IdentityCriterion = {
    new IdentityCriterion()
  }

  def createMergeFeatureLabelImagePreprocessing(): MergeFeatureLabel = {
    new MergeFeatureLabel()
  }

  def createMergeFeatureLabelFeatureTransformer(): MergeFeatureLabelFeatureTransformer = {
    new MergeFeatureLabelFeatureTransformer()
  }

  def createTFValidationMethod(valMethod: ValidationMethod[Float], name: String,
                               outputIndices: java.util.List[Int],
                               labelIndices: java.util.List[Int]): TFValidationMethod = {
    new TFValidationMethod(valMethod, name, outputIndices, labelIndices)
  }

  def createStatelessMetric(name: String, idx: Int): StatelessMetric = {
    new StatelessMetric(name, idx)
  }

  def createGanOptimMethod(dOptim: OptimMethod[T],
                           gOptim: OptimMethod[T],
                           dStep: Int, gStep: Int, gParamSize: Int): OptimMethod[T] = {
    new GanOptimMethod[T](dOptim, gOptim, dStep, gStep, gParamSize)
  }


  def createFakeOptimMethod(): OptimMethod[T] = {
    new FakeOptimMethod[T]()
  }

  def createMiniBatchRDDFromStringRDD(stringRDD: JavaRDD[Array[Byte]],
                                      batchSize: Int): RDDWrapper[TFMiniBatch] = {
    import TFTensorNumeric.NumericByteArray

    val rdd = stringRDD.rdd.mapPartitions { stringIter =>
      stringIter.grouped(batchSize).map { data =>
        val tensor = Tensor[Array[Byte]](data.toArray, shape = Array(data.length))
        new TFMiniBatch(Array(tensor))
      }
    }
      RDDWrapper[TFMiniBatch](rdd)
  }

  def createMiniBatchRDDFromTFDataset(graph: Array[Byte],
                                      initIteratorOp: String,
                                      outputNames: JList[String],
                                      outputTypes: JList[Int],
                                      shardIndex: String): RDDWrapper[TFMiniBatch] = {
    val types = outputTypes.asScala.map(TFUtils.tfenum2datatype).toVector
    val names = outputNames.asScala.toVector
    val sc = SparkContext.getOrCreate()
    val nodeNumber = EngineRef.getNodeNumber()
    val coreNumber = EngineRef.getCoreNumber()
    val totoalCoreNumber = nodeNumber * coreNumber

    val broadcastedGraph = sc.broadcast(graph)
    val originRdd = sc.parallelize(
      Array.tabulate(totoalCoreNumber * 20)(_ => 0), totoalCoreNumber * 10)
      .mapPartitions(_ => (0 until 10).toIterator)
      .coalesce(totoalCoreNumber)
    val resultRDD = originRdd.mapPartitionsWithIndex { case (idx, iter) =>
      val graphDef = broadcastedGraph.value
      val runner = GraphRunner(graphDef,
        null, null, null, null, SessionConfig(intraOpParallelismThreads = coreNumber).toByteArray())
      TFDataFeatureSet.makeIterators(
        runner,
        false,
        initIteratorOp,
        idx,
        shardIndex,
        types,
        names
      )
    }
    RDDWrapper(resultRDD)
  }

  def createMiniBatchRDDFromTFDatasetEval(graph: Array[Byte],
                                          initIteratorOp: String,
                                          outputNames: JList[String],
                                          outputTypes: JList[Int],
                                          shardIndex: String,
                                          featureLength: Int): RDDWrapper[TFMiniBatch] = {
    val rdd = createMiniBatchRDDFromTFDataset(graph, initIteratorOp, outputNames,
      outputTypes, shardIndex).value
    val resultRDD = rdd.map(batch => TFMiniBatch(batch.input.slice(0, featureLength),
      batch.input.slice(featureLength, batch.input.length)))
    RDDWrapper(resultRDD)
  }

  def createTFDataFeatureSet(graph: Array[Byte],
                             initIteratorOp: String,
                             outputNames: JList[String],
                             outputTypes: JList[Int],
                             shardIndex: String): TFDataFeatureSet = {


    TFDataFeatureSet(graph,
      initIteratorOp,
      outputNames.asScala.toArray, outputTypes.asScala.toArray, shardIndex)
  }

  def createMiniBatchFeatureSetFromStringRDD(stringRDD: JavaRDD[Array[Byte]],
                                             batchSize: Int, seqOrder: Boolean,
                                             shuffle: Boolean): FeatureSet[TFMiniBatch] = {
    FeatureSet.rdd(stringRDD,
      sequentialOrder = seqOrder,
      shuffle = shuffle).transform(new StringToMiniBatch(batchSize))
  }

  def loadZooCheckpoint(model: TFTrainingHelper, path: String): Unit = {
    model.loadZooCheckpoint(path)
  }

}
