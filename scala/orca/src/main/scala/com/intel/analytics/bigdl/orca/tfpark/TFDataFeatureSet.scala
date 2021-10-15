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

package com.intel.analytics.bigdl.orca.tfpark

import com.intel.analytics.bigdl.dllib.feature.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.feature.{DistributedDataSetWrapper, DistributedFeatureSet}
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.tensorflow.DataType

import com.intel.analytics.bigdl.orca.tfpark.TFTensorNumeric.NumericByteArray


class TFDataFeatureSet(private val graphRDD: RDD[Array[Byte]],
                       private val initIteratorOp: String,
                       private val initTableOp: String,
                       private val outputNames: Array[String],
                       private val outputTypes: Array[DataType],
                       private val shardIndex: String,
                       sessionConfig: SessionConfig)
  extends DistributedFeatureSet[MiniBatch[Float]] {

  private val graphRunnerRDD = getGraphRunnerRDD(graphRDD)

  private def getGraphRunnerRDD(rdd: RDD[Array[Byte]]): RDD[GraphRunner] = {

    val config = sessionConfig.toByteArray()

    val graphRunnerRDD = rdd.mapPartitions { iter =>
      if (iter.hasNext) {
        val graphDef = iter.next()
        val runner = GraphRunner(graphDef,
          null, null, null, null, config)
        Iterator.single(runner)
      } else {
        throw new IllegalArgumentException("the input dataset rdd has an empty partition")
      }

    }.setName("GraphRunnerRDD").cache()
    graphRunnerRDD.count()
    graphRunnerRDD
  }
  override def originRDD(): RDD[_] = {
    graphRunnerRDD
  }

  override def data(train: Boolean): RDD[MiniBatch[Float]] = {
    val initOp = this.initIteratorOp
    val names = this.outputNames.toVector
    val types = this.outputTypes.toVector
    val shardIdx = this.shardIndex
    val initTableOp = this.initTableOp

    graphRunnerRDD.mapPartitionsWithIndex { case (idx, dataIter) =>
      val graphRunner = dataIter.next()
      TFDataFeatureSet.makeIterators(
        graphRunner,
        train,
        initOp,
        initTableOp,
        idx,
        shardIdx,
        types,
        names
      )
    }
  }

  override def shuffle(): Unit = {

  }

  override def size(): Long = {
    -1
  }

  override def toDistributed(): DistributedDataSet[MiniBatch[Float]] = {
    new DistributedDataSetWrapper[MiniBatch[Float]](this)
  }
}

object TFDataFeatureSet {
  def apply(graph: Array[Byte],
            initIteratorOp: String,
            initTableOp: String,
            outputNames: Array[String],
            outputTypes: Array[Int],
            shardIndex: String,
            interOpParallelismThreads: Int,
            intraOpParallelismThreads: Int
           ): TFDataFeatureSet = {
    val types = outputTypes.map(TFUtils.tfenum2datatype)
    new TFDataFeatureSet(createGraphRDD(graph),
      initIteratorOp, initTableOp, outputNames, types, shardIndex,
      sessionConfig = SessionConfig(intraOpParallelismThreads = intraOpParallelismThreads,
        interOpParallelismThreads = interOpParallelismThreads))
  }

  def apply(graphRDD: RDD[Array[Byte]],
            initIteratorOp: String,
            initTableOp: String,
            outputNames: Array[String],
            outputTypes: Array[Int],
            shardIndex: String,
            interOpParallelismThreads: Int,
            intraOpParallelismThreads: Int
           ): TFDataFeatureSet = {
    val types = outputTypes.map(TFUtils.tfenum2datatype)
    val nodeNumber = Engine.nodeNumber()
    require(nodeNumber == graphRDD.getNumPartitions,
      s"number partitions should be the same as node number, " +
      s"got number partitions ${graphRDD.getNumPartitions}, node number ${nodeNumber}")
    new TFDataFeatureSet(graphRDD, initIteratorOp, initTableOp, outputNames, types, shardIndex,
      sessionConfig = SessionConfig(intraOpParallelismThreads = intraOpParallelismThreads,
        interOpParallelismThreads = interOpParallelismThreads))
  }

  private[bigdl] def createGraphRDD(graph: Array[Byte]): RDD[Array[Byte]] = {
    val sc = SparkContext.getOrCreate()
    val nodeNumber = Engine.nodeNumber()
    val coreNumber = Engine.coreNumber()

    val broadcastedGraph = sc.broadcast(graph)
    val originRdd = sc.parallelize(
      Array.tabulate(nodeNumber * 20)(_ => 0), nodeNumber * 10)
      .mapPartitions(_ => (0 until 20).toIterator)
      .coalesce(nodeNumber)
      .setName("PartitionRDD")
      .persist(StorageLevel.DISK_ONLY)
    originRdd.count()
    originRdd.mapPartitions { _ =>
      val graphDef = broadcastedGraph.value
      Iterator.single(graphDef)
    }.setName("GraphRDD")
  }

  private[bigdl] def generateOutputTensors(types: Vector[DataType]) = {
    val outputs = Array.tabulate[Tensor[_]](types.length) { i =>
      if (types(i) == DataType.STRING) {
        Tensor[Array[Byte]]()
      } else {
        Tensor[Float]()
      }
    }
    outputs
  }

  private[bigdl] def makeIterators(graphRunner: GraphRunner,
                                 train: Boolean,
                                 initOp: String,
                                 initTableOp: String,
                                 idx: Int,
                                 shardIdx: String,
                                 types: Vector[DataType],
                                 names: Vector[String]): Iterator[TFMiniBatch] = {
    def intiIterator(): Unit = {
      if (shardIdx != null) {
        graphRunner.runTargets(Vector(initOp, initTableOp),
          inputs = Vector(Tensor.scalar[Float](idx.toFloat)),
          inputTypes = Vector(DataType.INT64),
          inputNames = Vector(shardIdx))
      } else {
        graphRunner.runTargets(Vector(initOp, initTableOp))
      }

    }
    if (train) {
      new Iterator[TFMiniBatch] {

        override def hasNext(): Boolean = {
          true
        }

        private def getNext() = {
          val outputs = TFDataFeatureSet.generateOutputTensors(types)
          val outputVec = outputs.toVector
          try {
            graphRunner.runOutputs(outputVec, names, types)
          } catch {
            case _: java.lang.IndexOutOfBoundsException =>
              intiIterator()
              graphRunner.runOutputs(outputVec, names, types)
            case _: java.lang.IllegalStateException =>
              intiIterator()
              graphRunner.runOutputs(outputVec, names, types)
            case e: Throwable => throw e
          }
          outputs
        }

        override def next(): TFMiniBatch = {
          TFMiniBatch(getNext())
        }
      }
    } else {
      intiIterator()
      new Iterator[TFMiniBatch] {

        private var buffer: Array[Tensor[_]] = null
        override def hasNext(): Boolean = {
          if (buffer != null) {
            true
          } else {
            val (success, result) = getNext()
            if (success) {
              buffer = result
            }
            success
          }
        }

        private def getNext() = {
          val outputs = TFDataFeatureSet.generateOutputTensors(types)
          val outputVec = outputs.toVector
          val success = try {
            graphRunner.runOutputs(outputVec, names, types)
            true
          } catch {
            case _: java.lang.IndexOutOfBoundsException => false
            case e: Throwable => throw e
          }
          (success, outputs)
        }

        override def next(): TFMiniBatch = {
          if (hasNext()) {
            val result = TFMiniBatch(buffer)
            buffer = null
            result
          } else {
            throw new NoSuchElementException("Next on an empty iterator")
          }
        }
      }
    }
  }
}
