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
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Graph, Linear}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.mutable
import scala.reflect.ClassTag

abstract class Session[T: ClassTag] {

  def train(outputs: Seq[String],
            data: Seq[Tensor[T]],
            label: Seq[Tensor[T]],
            optMethod: OptimMethod[T],
            criterion: Criterion[T],
            batchSize: Int,
            endWhen: Trigger): Graph[T]
}

class BigDLSessionImpl[T: ClassTag](
       graph: Seq[NodeDef],
       context: mutable.HashMap[String, (Tensor[T], Tensor[T])])
                         (implicit ev: TensorNumeric[T]) extends Session[T] {
  import scala.collection.JavaConverters._

  val sc = SparkContext.getOrCreate()

  private val inputOp = Set("ReaderReadV2", "QueueDequeueV2", "QueueDequeueManyV2", "Placeholder")

  private val (wholeTFGraph, _) = TensorflowLoader.buildTFGraph(graph.asJava, null)

  private val name2Node = wholeTFGraph.
    DFS.filter(n => n.element != null).map(node => (node.element.getName, node)).toMap

  private def constructModel(endPoints: Seq[String]): (Graph[T], Node[NodeDef]) = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs) = TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

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
                     data: Seq[Tensor[T]],
                     label: Seq[Tensor[T]],
                     optMethod: OptimMethod[T],
                     criterion: Criterion[T],
                     batchSize: Int, endWhen: Trigger): Graph[T] = {

    val samples = data.zip(label).map { elem =>
      Sample(elem._1, elem._2)
    }

    val coreNum = Engine.coreNumber()
    val rdd = sc.parallelize(samples, coreNum)

    val (model, input) = constructModel(outputs)

    require(input.element.getOp == "Placeholder",
      "only support Placeholder as input when in-memory input data is provided")

    val opt = Optimizer(
      model,
      rdd,
      criterion,
      batchSize
    )
    val optMethod = new SGD[T]()
    opt.setOptimMethod(optMethod).setEndWhen(endWhen)
      .optimize()
    model
  }

}
