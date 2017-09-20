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
package com.intel.analytics.bigdl.utils.tf

import java.nio.{ByteOrder, DoubleBuffer, FloatBuffer}

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Graph, Linear}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.tensorflow.framework.{GraphDef, NodeDef}
import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import TFTensorNumeric.NumericByteString

import scala.collection.mutable
import scala.reflect.ClassTag

abstract class Session[T: ClassTag] {

  def train(outputs: Seq[String],
            dataSet: DistributedDataSet[MiniBatch[T]],
            optMethod: OptimMethod[T],
            criterion: Criterion[T],
            endWhen: Trigger): Graph[T]
}

class BigDLSessionImpl[T: ClassTag](
       graph: Seq[NodeDef],
       context: mutable.HashMap[String, (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])])
                         (implicit ev: TensorNumeric[T]) extends Session[T] {
  import scala.collection.JavaConverters._

  val sc = SparkContext.getOrCreate()

  private val inputOp = Set("ReaderReadV2", "QueueDequeueV2", "QueueDequeueManyV2", "Placeholder")

  private val dequeueOp = Set("QueueDequeueV2", "QueueDequeueManyV2", "ReaderReadV2")

  private val enqueueOp = Set("QueueEnqueueV2", "QueueEnqueueManyV2")

  private val readerOps = Set("TFRecordReaderV2")

  private val (wholeTFGraph, _, _) = TensorflowLoader.buildTFGraph(graph.asJava, null)

  private val name2Node = wholeTFGraph.
    DFS.filter(_.element != null).map(node => (node.element.getName, node)).toMap

  private def tableToSeq(table: Table): Seq[Tensor[T]] = {
    for (i <- 0 until table.length()) yield {
      table(i).asInstanceOf[Tensor[T]]
    }
  }

  private def seqToTable(tensors: Seq[Tensor[T]]): Table = {
    val table = new Table()
    for (tensor <- tensors) {
      table.insert(tensor)
    }
    table
  }

  private def handleReaderNode(node: Node[NodeDef], cache: DataCache): RDD[Table] = {
    require(node.prevNodes.length == 2, "require ReaderReadV2 only has two inputs")
    val readerNode = node.prevNodes.head
    val queueNode = node.prevNodes(1)
    val dequeNodeNames = mutable.LinkedHashSet[String]()

    queueNode.nextNodes
      .filter(n => n.element != null && dequeueOp(n.element.getOp))
      .map(n => n.element.getName.split(":")(0)).foreach(dequeNodeNames.add)

    val nameToIndex = dequeNodeNames.zipWithIndex.toMap
    val index = nameToIndex(node.element.getName)
    val nSlices = dequeNodeNames.size

    val enqueueNodes = queueNode.nextNodes
      .filter(n => n.element != null && enqueueOp(n.element.getOp))
    val filesSeq = if (cache.contains(queueNode.element.getName)) {
      val resultArray = cache(queueNode.element.getName)
      val result = resultArray(index)
      resultArray(index) = null
      result
    } else {
      val allResult = enqueueNodes.map { enqueueNode =>
        val inputs = Seq(enqueueNode.element.getName)
        val result = constructLocalData(inputs, new DataCache())
        if (enqueueNode.element.getOp == "QueueEnqueueManyV2") {
          result.flatMap { table =>
            val nElem = table.length()
            require(nElem >= 1, "EnqueueManyV2 encounter a empty table")
            val first = table[Tensor[ByteString]](1)
            require(first.nDimension() >= 1)
            val depth = first.size(1)
            val result = new Array[Table](depth)
            var i = 0
            while(i < depth) {
              var j = 0
              val newTable = new Table()
              while (j < nElem) {
                val elem = table[Tensor[ByteString]](j + 1)
                newTable.insert(elem(i + 1))
                j = j + 1
              }
              result(i) = newTable
              i = i + 1
            }
            result
          }
        } else {
          result
        }
      }.reduce { (outerSeq1, outerSeq2) =>
        outerSeq1.zip(outerSeq2).map { case (seq1, seq2) =>
          seq1.add(seq2)
        }
      }
      val resultArray = split(allResult, nSlices)
      cache.put(queueNode.element.getName, resultArray)
      resultArray(index)
    }

    readerNode.element.getOp match {
      case "TFRecordReaderV2" => readTFRecord(filesSeq)
    }
  }

  private def split[A](xs: Seq[A], n: Int): Array[Seq[A]] = {
    val result = new Array[Seq[A]](n)
    var i = 0
    while (i < n) {
      result(i) = Vector[A]()
      i = i + 1
    }

    var j = 0
    while (j < xs.length) {
      result(j % n) = result(j % n) :+ xs(j)
      j = j + 1
    }

    result
  }

  private def readTFRecord(filesTable: Seq[Table]): RDD[Table] = {
    val result = filesTable.map { t =>
        require(t.length() == 1 && t(1).isInstanceOf[Tensor[ByteString]],
          "Reader can only read one file at a time")
        val fileTensor = t[Tensor[ByteString]](1)
        require(fileTensor.isScalar)
        val file = fileTensor.value()
        file
    }.flatMap { file =>
      val iter = new TFRecordIterator(new java.io.File(file.toStringUtf8))
      iter
    }.map { record =>
      val table = T()
      val key = Tensor[ByteString](Array(ByteString.copyFromUtf8("somekey")), Array[Int]())
      val value = Tensor[ByteString](Array(ByteString.copyFrom(record)), Array[Int]())
      table.insert(key)
      table.insert(value)
      table
    }
    val resultRdd = sc.parallelize(result, numSlices = Engine.coreNumber())
    resultRdd
  }

  private def handleLocalDequeue(node: Node[NodeDef], cache: DataCache): Seq[Table] = {
    require(node.prevNodes.length == 1, "require QueueDequeueV2 only has one input")
    val queueNode = node.prevNodes.head
    val enqueueNodes = queueNode.nextNodes.filter(n => enqueueOp(n.element.getOp))
    val dequeNodeNames = mutable.LinkedHashSet[String]()

    queueNode.nextNodes
      .filter(n => n.element != null && dequeueOp(n.element.getOp))
      .map(n => n.element.getName.split(":")(0)).foreach(dequeNodeNames.add)

    val nameToIndex = dequeNodeNames.zipWithIndex.toMap
    val index = nameToIndex(node.element.getName)
    val nSlices = dequeNodeNames.size

    val dataSeq = if (cache.contains(queueNode.element.getName)) {
      val resultArray = cache(queueNode.element.getName)
      val result = resultArray(index)
      resultArray(index) = null
      result
    } else {
      val allResult = enqueueNodes.map { enqueueNode =>
        val inputs = Seq(enqueueNode.element.getName)
        constructLocalData(inputs, new DataCache())
      }.reduce { (outerSeq1, outerSeq2) =>
        outerSeq1.zip(outerSeq2).map { case (seq1, seq2) =>
          seq1.add(seq2)
        }
      }
      val resultArray = split(allResult, nSlices)
      cache.put(queueNode.element.getName, resultArray)
      resultArray(index)
    }
    dataSeq
  }

  private def handleDistriDequeue(node: Node[NodeDef], cache: DataCache): RDD[Table] = {
    require(node.prevNodes.length == 1, "require QueueDequeueV2 only has one input")
    val queueNode = node.prevNodes.head
    val dequeueNodes = queueNode.nextNodes
      .filter(n => n.element != null && dequeueOp(n.element.getOp))
      .map(n => n.element.getName.split(":")(0)).toSet
    require(dequeueNodes.size == 1, "only support one dequeue node after reader")
    val enqueueNodes = queueNode.nextNodes
      .filter(n => n.element != null && enqueueOp(n.element.getOp))
    val rdd = enqueueNodes.map { enqueueNode =>
      val inputs = Seq(enqueueNode.element.getName)
      constructDistributeData(inputs, cache)
    }.reduce { (rdd1, rdd2) =>
      rdd1.union(rdd2)
    }
    rdd
  }

  private def handleDistriDequeueManyNode(node: Node[NodeDef], cache: DataCache): RDD[Table] = {
    require(node.prevNodes.length == 2, "require QueueDequeueManyV2 only has two input")
    val queueNode = node.prevNodes.head
    val enqueueNodes = queueNode.nextNodes.filter(n => enqueueOp(n.element.getOp))
    // get previous rdd
    val rdd = enqueueNodes.map { enqueueNode =>
      val inputs = Seq(enqueueNode.element.getName)
      constructDistributeData(inputs, cache)
    }.reduce { (rdd1, rdd2) =>
      rdd1.zip(rdd2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }

    // get batch size
    val batchSizeNode = node.prevNodes(1)
    require(batchSizeNode.element.getOp == "Const", "batchsize must be a const")

    val batchSize = batchSizeNode.element.getAttrMap.get("value").getI.toInt

    val batchRdd = rdd.mapPartitions { iter =>

      new Iterator[Table] {
        override def hasNext: Boolean = iter.hasNext

        override def next(): Table = {
          require(iter.hasNext, "Call next() on a empty iterator")
          val batch = for (_ <- 0 until batchSize if iter.hasNext) yield {
            iter.next()
          }
          pack(batch)
        }
      }

    }
    batchRdd
  }

  private def pack(tables: Seq[Table], dimension: Int = 1): Table = {
    val batch = tables.map(tableToSeq)
    val firstSeq = batch.head
    val sizes = firstSeq.map { tensor =>
      val nDim = tensor.nDimension()
      val size: Array[Int] = new Array[Int](nDim + 1)
      var i = 1
      while(i <= nDim + 1) {
        if (i < dimension) {
          size(i-1) = tensor.size(i)
        } else if (i == dimension) {
          size(i-1) = batch.length
        } else {
          size(i-1) = tensor.size(i - 1)
        }
        i = i + 1
      }
      size
    }

    val results = sizes.map { size =>
      Tensor[T](size)
    }

    for ((seq, index) <- batch.zipWithIndex) {
      results.zip(seq).foreach { case (result, tensor) =>
        result.narrow(dimension, index + 1, 1).copy(tensor)
      }
    }
    seqToTable(results)
  }

  type DataCache = mutable.HashMap[String, Array[Seq[Table]]]

  private def adjustInputNames(inputs: Seq[String]): Seq[String] = {
    val stripedNames = inputs.map(_.split(":")(0))
    val set = mutable.LinkedHashSet[String]()
    for (name <- stripedNames) {
      set.add(name)
    }
    set.toSeq
  }

  def constructLocalData(endPoints: Seq[String], cache: DataCache): Seq[Table] = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs, originInputs) = TensorflowLoader.
      buildTFGraph(graph.asJava, endPoints, isInputOp)

    val adjustedInputs = adjustInputNames(originInputs)
    val transformer = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputs,
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]


    if (adjustedInputs.nonEmpty) {
      val inputNodes = originInputs.map(name2Node)
      val inputDataSeq = inputNodes.map { node => // this is the input op
        node.element.getOp match {
          // only support Dequeue before reader
          case "QueueDequeueV2" => handleLocalDequeue(node, cache)
        }
      }

      val reducedInputSeq = inputDataSeq.reduce { (outerSeq1, outerSeq2) =>
        outerSeq1.zip(outerSeq2).map { case (seq1, seq2) =>
          seq1.add(seq2)
        }
      }

      reducedInputSeq.map { tensors =>
        val output = transformer.forward(tensors.flatten())
        toTable(output)
      }
    } else {
      Seq(toTable(transformer.forward(T())))
    }
  }

  private def toTable(activity: Activity): Table = {
    activity match {
      case t: Tensor[_] => T(t)
      case t: Table => t
    }
  }

  def constructDistributeData(endPoints: Seq[String], cache: DataCache): RDD[Table] = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs, originInputs) =
      TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    val adjustedInputs = adjustInputNames(originInputs)

    val inputNodes = adjustedInputs.map(name2Node)

    val transformer = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputs,
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]

    val inputRdds = inputNodes.map { node => // this is the input op
      node.element.getOp match {
        case "ReaderReadV2" => handleReaderNode(node, cache)
        case "QueueDequeueV2" => handleDistriDequeue(node, cache)
        case "QueueDequeueManyV2" => handleDistriDequeueManyNode(node, cache)
      }
    }
    val inputRdd = inputRdds.reduce { (rdd1, rdd2) =>
      rdd1.zip(rdd2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }

    if (!inputRdd.isEmpty()) {
      val first = inputRdd.first()
      println(first)
    }
    val modelBroadCast = ModelBroadcast[T].broadcast(sc, transformer)
    inputRdd.map { tensors =>
      val trans = modelBroadCast.value()
      val output = trans.forward(tensors.flatten())
      output.asInstanceOf[Table]
      tensors
    }
  }


  private def constructModel(endPoints: Seq[String]): (Graph[T], Node[NodeDef]) = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs, _) = TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    val inputNodes = inputs.map(name2Node)

    require(inputNodes.length == 1, "Only support one model input")

    val model = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputNodes.map(_.element.getName),
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]
    (model, inputNodes.head)
  }

  override def train(outputs: Seq[String],
                     dataSet: DistributedDataSet[MiniBatch[T]],
                     optMethod: OptimMethod[T],
                     criterion: Criterion[T],
                     endWhen: Trigger): Graph[T] = {

    val (model, input) = constructModel(outputs)

    require(input.element.getOp == "Placeholder",
      "only support Placeholder as input when in-memory input data is provided")

    val opt = new DistriOptimizer(
      model,
      dataSet,
      criterion
    )
    val optMethod = new SGD[T]()
    opt.setOptimMethod(optMethod).setEndWhen(endWhen)
      .optimize()
    model
  }


  def train(modelOutputs: Seq[String],
                     labels: Seq[String],
                     optMethod: OptimMethod[T],
                     criterion: Criterion[T],
                     endWhen: Trigger): Graph[T] = {
    val (model, modelInput) = constructModel(modelOutputs)

    val (transformerForLabel, labelInput) = constructModel(labels)

    require(modelInput == labelInput, "data and label should come from the same queue")

    val cache = new DataCache()

    val data = constructDistributeData(modelOutputs ++ labels, cache)

    throw new NotImplementedError()
  }

  def run(endPoints: Array[String], batchSize: Int): RDD[Array[Tensor[T]]] = {
    throw new NotImplementedError()
  }


}
