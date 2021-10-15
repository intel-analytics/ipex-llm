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
package com.intel.analytics.bigdl.orca.tfpark.python

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.optim._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.{PythonZoo, RDDWrapper}
import com.intel.analytics.bigdl.dllib.utils.python.api.Sample
import com.intel.analytics.bigdl.dllib.feature.FeatureSet
import com.intel.analytics.bigdl.orca.tfpark._
import org.apache.spark.api.java.JavaRDD

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import java.util.{List => JList}

import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.tensorflow.DataType


object PythonTFPark {

  def ofFloat(): PythonTFPark[Float] = new PythonTFPark[Float]()

  def ofDouble(): PythonTFPark[Double] = new PythonTFPark[Double]()

}


class PythonTFPark[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def zooRDDSampleToMiniBatch(rdd: JavaRDD[Sample],
                              batchSizePerPartition: Int,
                              dropRemainder: Boolean): RDDWrapper[MiniBatch[T]] = {
    import com.intel.analytics.bigdl.orca.tfpark.SampleToMiniBatch
    val partitionNum = rdd.rdd.getNumPartitions
    val totalBatchSize = batchSizePerPartition * partitionNum
    val transBroad = rdd.sparkContext.broadcast(new SampleToMiniBatch(
      totalBatch = totalBatchSize,
      None,
      partitionNum = Some(partitionNum),
      featurePaddingParam = None,
      dropRemainder = dropRemainder))

    val miniBatchRdd = rdd.rdd.map(toJSample).mapPartitions { iter =>
      val localTransformer = transBroad.value.cloneTransformer()
      localTransformer(iter)
    }
    RDDWrapper(miniBatchRdd)
  }

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

  def createStatelessMetric(name: String, idx: Int, countIdx: Int): StatelessMetric = {
    new StatelessMetric(name, idx, countIdx)
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
                                      initTableOp: String,
                                      outputNames: JList[String],
                                      outputTypes: JList[Int],
                                      shardIndex: String): RDDWrapper[TFMiniBatch] = {
    val types = outputTypes.asScala.map(TFUtils.tfenum2datatype).toVector
    val names = outputNames.asScala.toVector
    val sc = SparkContext.getOrCreate()
    val nodeNumber = Engine.nodeNumber()
    val coreNumber = Engine.coreNumber()
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
        initTableOp,
        idx,
        shardIndex,
        types,
        names
      )
    }
    RDDWrapper(resultRDD)
  }

  def createMiniBatchRDDFromTFDataset(graphRDD: JavaRDD[Array[Byte]],
                                      initIteratorOp: String,
                                      initTableOp: String,
                                      outputNames: JList[String],
                                      outputTypes: JList[Int],
                                      shardIndex: String): RDDWrapper[TFMiniBatch] = {
    val types = outputTypes.asScala.map(TFUtils.tfenum2datatype).toVector
    val names = outputNames.asScala.toVector
    val coreNumber = Engine.coreNumber()

    val resultRDD = graphRDD.rdd.mapPartitionsWithIndex { case (idx, iter) =>
      if (iter.hasNext) {
        val graphDef = iter.next()
        val runner = GraphRunner(graphDef, null, null, null, null,
          SessionConfig(intraOpParallelismThreads = coreNumber).toByteArray())
        TFDataFeatureSet.makeIterators(
          runner,
          false,
          initIteratorOp,
          initTableOp,
          idx,
          shardIndex,
          types,
          names
        )
      } else {
        Iterator.empty
      }

    }
    RDDWrapper(resultRDD)
  }

  def createMiniBatchRDDFromTFDatasetEval(graph: Array[Byte],
                                          initIteratorOp: String,
                                          initTableOp: String,
                                          outputNames: JList[String],
                                          outputTypes: JList[Int],
                                          shardIndex: String): RDDWrapper[TFMiniBatch] = {
    val rdd = createMiniBatchRDDFromTFDataset(graph, initIteratorOp, initTableOp, outputNames,
      outputTypes, shardIndex).value
    RDDWrapper(rdd)
  }

  def createMiniBatchRDDFromTFDatasetEval(graphRDD: JavaRDD[Array[Byte]],
                                          initIteratorOp: String,
                                          initTableOp: String,
                                          outputNames: JList[String],
                                          outputTypes: JList[Int],
                                          shardIndex: String): RDDWrapper[TFMiniBatch] = {
    val rdd = createMiniBatchRDDFromTFDataset(graphRDD, initIteratorOp, initTableOp, outputNames,
      outputTypes, shardIndex).value
    RDDWrapper(rdd)
  }

  def createTFDataFeatureSet(graph: Array[Byte],
                             initIteratorOp: String,
                             initTableOp: String,
                             outputNames: JList[String],
                             outputTypes: JList[Int],
                             shardIndex: String,
                             interOpParallelismThreads: Int,
                             intraOpParallelismThreads: Int): TFDataFeatureSet = {


    TFDataFeatureSet(graph,
      initIteratorOp,
      initTableOp,
      outputNames.asScala.toArray,
      outputTypes.asScala.toArray, shardIndex, interOpParallelismThreads,
      intraOpParallelismThreads)
  }

  def createTFDataFeatureSet(graphRDD: JavaRDD[Array[Byte]],
                             initIteratorOp: String,
                             initTableOp: String,
                             outputNames: JList[String],
                             outputTypes: JList[Int],
                             shardIndex: String,
                             interOpParallelismThreads: Int,
                             intraOpParallelismThreads: Int): TFDataFeatureSet = {

    TFDataFeatureSet(graphRDD.rdd,
      initIteratorOp,
      initTableOp,
      outputNames.asScala.toArray,
      outputTypes.asScala.toArray, shardIndex, interOpParallelismThreads,
      intraOpParallelismThreads)
  }

  def createMiniBatchFeatureSetFromStringRDD(stringRDD: JavaRDD[Array[Byte]],
                                             batchSize: Int, seqOrder: Boolean,
                                             shuffle: Boolean): FeatureSet[TFMiniBatch] = {
    FeatureSet.rdd(stringRDD,
      sequentialOrder = seqOrder,
      shuffle = shuffle).transform(new StringToMiniBatch(batchSize))
  }

  import com.intel.analytics.bigdl.orca.tfpark
  def createTFParkSampleToMiniBatch(batchSize: Int,
                                    dropRemainder: Boolean): tfpark.SampleToMiniBatch[T] = {
    new tfpark.SampleToMiniBatch[T](totalBatch = batchSize,
      dropRemainder = dropRemainder)
  }

  def loadZooCheckpoint(model: TFTrainingHelper, path: String): Unit = {
    model.loadZooCheckpoint(path)
  }

}
