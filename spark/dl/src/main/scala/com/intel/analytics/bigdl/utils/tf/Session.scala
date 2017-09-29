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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
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

  /**
   * Train the model specified by the model output
   * @param outputs model output
   * @param dataSet the training data set
   * @return trained model
   */
  def train(outputs: Seq[String],
            dataSet: DistributedDataSet[MiniBatch[T]],
            optMethod: OptimMethod[T],
            criterion: Criterion[T],
            endWhen: Trigger): Graph[T]

  /**
   * Execute distributed training on the given graph
   * @param endPoints
   * @param optMethod
   * @param endWhen
   * @return
   */
  def fit(endPoints: Seq[String], optMethod: OptimMethod[T], endWhen: Trigger): this.type

  /**
   * Get the result from the given nodes
   * @return
   */
  def predict(outputs: Seq[String]): RDD[Sample[_]]

  /**
   * Export the weights to the bin file
   * @param binFile
   * @return
   */
  def exportWeights(binFile: String): this.type

  /**
   * Load the weights from the bin file
   * @param binFile
   * @return
   */
  def loadWeights(binFile: String): this.type
}

class BigDLSessionImpl[T: ClassTag](graph: Seq[NodeDef], context: Context[T],
  byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)
  (implicit ev: TensorNumeric[T]) extends Session[T] {

  import scala.collection.JavaConverters._

  override def train(outputs: Seq[String],
    dataSet: DistributedDataSet[MiniBatch[T]],
    optMethod: OptimMethod[T],
    criterion: Criterion[T],
    endWhen: Trigger): Graph[T] = {

    val (model, input) = constructModel(outputs, byteOrder)

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

  override def fit(endPoints: Seq[String], optMethod: OptimMethod[T], endWhen: Trigger)
  : this.type = {
    val outputs = endPoints
    val updaterSet = new mutable.HashSet[String]
    val weightsAndGrads = endPoints.map(e => name2Node(e)).map(n => n.graph(true).DFS).flatten
      .map(n => TFUpdater(n.element)).flatten.toSet

    require(weightsAndGrads.size != 0, "Cannot find updater nodes")
    context.setAssignGrads(weightsAndGrads)
    val (model, input) = constructModel(weightsAndGrads.map(_._2).toSeq, byteOrder)
    this
  }

  override def predict(outputs: Seq[String]): RDD[Sample[_]] = {
    null
  }

  override def exportWeights(binFile: String): this.type = {
    this
  }

  override def loadWeights(binFile: String): this.type = {
    this
  }

  private val inputOp = Set("ReaderReadV2", "QueueDequeueV2", "QueueDequeueManyV2", "Placeholder")

  private val dequeueOp = Set("QueueDequeueV2", "QueueDequeueManyV2", "ReaderReadV2")

  private val enqueueOp = Set("QueueEnqueueV2", "QueueEnqueueManyV2")

  private val queueOp = Set("RandomShuffleQueueV2", "FIFOQueueV2")

  private val (wholeTFGraph, _, _) = TensorflowLoader.buildTFGraph(graph.asJava, null)

  private val name2Node = wholeTFGraph.
    DFS.filter(_.element != null).map(node => (node.element.getName, node)).toMap

  private def handleReaderNode(node: Node[NodeDef], cache: DataCache,
    sc: SparkContext): RDD[Table] = {
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

    val enqueueNodes = findEnqueueNodes(queueNode)
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
          result.flatMap(BigDLSessionImpl.splitTensorByFirstDim)
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
      case "TFRecordReaderV2" => readTFRecord(filesSeq, sc)
      case "FixedLengthRecordReaderV2" => readFixedLengthRecord(filesSeq, readerNode.element, sc)
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

  private def readTFRecord(filesTable: Seq[Table], sc: SparkContext): RDD[Table] = {
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

  private def readFixedLengthRecord(filesTable: Seq[Table], readerNode: NodeDef, sc: SparkContext)
    : RDD[Table] = {

    val footerBytes = readerNode.getAttrMap.get("footer_bytes").getI.toInt
    val headerBytes = readerNode.getAttrMap.get("header_bytes").getI.toInt
    val hopBytes = readerNode.getAttrMap.get("hop_bytes").getI.toInt
    val recordBytes = readerNode.getAttrMap.get("record_bytes").getI.toInt

    val result = filesTable.map { t =>
      require(t.length() == 1 && t(1).isInstanceOf[Tensor[ByteString]],
        "Reader can only read one file at a time")
      val fileTensor = t[Tensor[ByteString]](1)
      require(fileTensor.isScalar)
      val file = fileTensor.value()
      file
    }.flatMap { file =>
      val iter = new FixedLengthRecordReader(
        new java.io.File(file.toStringUtf8),
        footerBytes,
        headerBytes,
        hopBytes,
        recordBytes)
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

  private val identityOp = Set("Switch", "Identity", "Merge")
  private def findEnqueueNodes(queueNode: Node[NodeDef]): Seq[Node[NodeDef]] = {
    val queue = mutable.Queue[Node[NodeDef]]()
    val enqueNodes = mutable.ArrayBuffer[Node[NodeDef]]()
    queue.enqueue(queueNode.nextNodes: _*)
    val visited = mutable.HashSet[Node[NodeDef]]()
    while(queue.nonEmpty) {
      val node = queue.dequeue()
      if (!visited(node)) {
        if (node.element != null && enqueueOp(node.element.getOp)) {
          enqueNodes += node
        } else if (node.element != null && identityOp(node.element.getOp)) {
          queue.enqueue(node.nextNodes: _*)
        }
      }
    }
    if (enqueNodes.isEmpty) {
      throw new IllegalArgumentException(
        s"Cannot find enqueue node for queue: ${queueNode.element}")
    } else {
      enqueNodes
    }
  }

  private def handleLocalDequeue(node: Node[NodeDef], cache: DataCache): Seq[Table] = {
    require(node.prevNodes.length == 1, "require QueueDequeueV2 only has one input")
    val queueNode = node.prevNodes.head
    val enqueueNodes = findEnqueueNodes(queueNode)
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
        val result = constructLocalData(inputs, new DataCache())
        if (enqueueNode.element.getOp == "QueueEnqueueManyV2") {
          result.flatMap(BigDLSessionImpl.splitTensorByFirstDim)
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
    dataSeq
  }

  private def handleDistriDequeue(node: Node[NodeDef], cache: DataCache,
    sc: SparkContext): RDD[Table] = {
    require(node.prevNodes.length == 1, "require QueueDequeueV2 only has one input")
    val queueNode = node.prevNodes.head
    val dequeueNodes = queueNode.nextNodes
      .filter(n => n.element != null && dequeueOp(n.element.getOp))
      .map(n => n.element.getName.split(":")(0)).toSet
    require(dequeueNodes.size == 1, "only support one dequeue node after reader")
    val enqueueNodes = findEnqueueNodes(queueNode)
    val rdd = enqueueNodes.map { enqueueNode =>
      val inputs = Seq(enqueueNode.element.getName)
      val result = constructDistributeData(inputs, cache, sc)
      if (enqueueNode.element.getOp == "QueueEnqueueManyV2") {
        result.flatMap(BigDLSessionImpl.splitTensorByFirstDim)
      } else {
        result
      }
    }.reduce { (rdd1, rdd2) =>
      rdd1.union(rdd2)
    }
    rdd
  }

  private def handleDistriDequeueManyNode(node: Node[NodeDef], cache: DataCache): RDD[Table] = {
    require(node.prevNodes.length == 2, "require QueueDequeueManyV2 only has two input")
    val queueNode = node.prevNodes.head
    val dequeueNodes = queueNode.nextNodes
      .filter(n => n.element != null && dequeueOp(n.element.getOp))
      .map(n => n.element.getName.split(":")(0)).toSet
    require(dequeueNodes.size == 1, "only support one dequeue node after reader")
    val enqueueNodes = findEnqueueNodes(queueNode)
    // get previous rdd
    val rdd = enqueueNodes.map { enqueueNode =>
      val inputs = Seq(enqueueNode.element.getName)
      val result = constructDistributeData(inputs, cache, sc)
      if (enqueueNode.element.getOp == "QueueEnqueueManyV2") {
        result.flatMap(BigDLSessionImpl.splitTensorByFirstDim)
      } else {
        result
      }
    }.reduce { (rdd1, rdd2) =>
      rdd1.union(rdd2)
    }

    // get batch size
    val batchSizeNode = node.prevNodes(1)
    require(batchSizeNode.element.getOp == "Const", "batchsize must be a const")

    val batchSize = batchSizeNode.element.getAttrMap.get("value").getTensor.getIntVal(0)

    val batchRdd = rdd.mapPartitions { iter =>

      new Iterator[Table] {
        private var firstBatch: Array[Table] = null
        override def hasNext: Boolean = iter.hasNext

        override def next(): Table = {
          require(iter.hasNext, "Call next() on a empty iterator")
          val tables = new Array[Table](batchSize)
          var index = 0
          for (i <- 0 until batchSize) {
            if (iter.hasNext) {
              tables(i) = iter.next()
            } else if (firstBatch == null) {
              tables(i) = tables(index)
              index = index + 1
            } else {
              tables(i) = firstBatch(index)
              index = index + 1
            }
          }
          if (firstBatch == null) {
            firstBatch = tables
          }
          val batch = tables.map(_.toSeq)
          val firstSeq = batch.head
          val sizes = firstSeq.map { tensor =>
            val nDim = tensor.nDimension()
            val size: Array[Int] = new Array[Int](nDim + 1)
            var i = 1
            while(i <= nDim + 1) {
              if (i < 1) {
                size(i-1) = tensor.size(i)
              } else if (i == 1) {
                size(i-1) = batch.length
              } else {
                size(i-1) = tensor.size(i - 1)
              }
              i = i + 1
            }
            size
          }

          val results = sizes.zipWithIndex.map { case (size, i) =>
            firstSeq(i).emptyInstance().resize(size)
          }

          for ((seq, index) <- batch.zipWithIndex) {
            results.zip(seq).foreach { case (result, tensor) =>
              result.asInstanceOf[Tensor[NumericWildcard]]
                .narrow(1, index + 1, 1)
                .copy(tensor.asInstanceOf[Tensor[NumericWildcard]])
            }
          }
          T.seq(results)
        }
      }

    }
    batchRdd
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

  private def checkAndRemoveQueueNode(tfGraph: DirectedGraph[NodeDef]) = {
    tfGraph.DFS.filter(n => n.element != null && enqueueOp(n.element.getOp))
      .foreach { node =>
        node.prevNodes.head.delete(node)
      }
  }

  private def constructLocalData(endPoints: Seq[String], cache: DataCache): Seq[Table] = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs, originInputs) = TensorflowLoader.
      buildTFGraph(graph.asJava, endPoints, isInputOp)

    checkAndRemoveQueueNode(tfGraph)

    val adjustedInputs = adjustInputNames(originInputs)
    val transformer = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputs.toSeq.map(_._2).flatten,
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context),
      generatedBackward = false
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

  private def constructDistributeData(endPoints: Seq[String], cache: DataCache,
      sc: SparkContext): RDD[Table] = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs, originInputs) =
      TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    checkAndRemoveQueueNode(tfGraph)

    val adjustedInputs = adjustInputNames(originInputs)

    val inputNodes = adjustedInputs.map(name2Node)

    val transformer = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputs.toSeq.map(_._2).flatten,
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context),
      generatedBackward = false
    ).asInstanceOf[Graph[T]]

    val inputRdds = inputNodes.map { node => // this is the input op
      node.element.getOp match {
        case "ReaderReadV2" => handleReaderNode(node, cache, sc)
        case "QueueDequeueV2" => handleDistriDequeue(node, cache, sc)
        case "QueueDequeueManyV2" => handleDistriDequeueManyNode(node, cache, sc)
      }
    }
    val inputRdd = inputRdds.reduce { (rdd1, rdd2) =>
      rdd1.zip(rdd2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }

    val modelBroadCast = ModelBroadcast[T]().broadcast(sc, transformer)
    inputRdd.map { tensors =>
      val trans = modelBroadCast.value()
      val output = trans.forward(tensors.flatten())
      output match {
        case t: Tensor[_] => T(t)
        case t: Table => t
      }
    }
  }


  private def constructModel(endPoints: Seq[String], byteOrder: ByteOrder)
      : (Graph[T], Node[NodeDef]) = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs, originInputs) =
      TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    checkAndRemoveQueueNode(tfGraph)

    val adjustedInputs = adjustInputNames(originInputs)

    val inputNodes = adjustedInputs.map(name2Node)

    require(inputNodes.length == 1, "Only support one model input")

    val model = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputs.toSeq.map(_._2).flatten,
      endPoints,
      byteOrder,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]
    (model, inputNodes.head)
  }

  /**
   * Get and calculate the data up to the specified endpoints, and
   * return as a RDD[Table]
   * @param endPoints output endpoints
   * @return
   */
  def getRDD(endPoints: Seq[String], sc: SparkContext): RDD[Table] = {
    val cache = new mutable.HashMap[String, Array[Seq[Table]]]()
    constructDistributeData(endPoints, cache, sc)
  }
}

object TFUpdater {
  def apply(node: NodeDef): Option[(String, String)] = {
    node.getOp match {
      case "ApplyRMSProp" =>
        Some((node.getInput(0), node.getInput(7)))
      case _ => None
    }
  }
}

object BigDLSessionImpl {
  private def splitTensorByFirstDim(table: Table): Array[Table] = {
    val nElem = table.length()
    require(nElem >= 1, "EnqueueManyV2 encounter a empty table")
    val first = table[Tensor[_]](1)
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
}
