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

package com.intel.analytics.bigdl.nn.bigquant

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.bigquant.Utils._
import com.intel.analytics.bigdl.nn.{Cell, Container, Graph}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, nn}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.reflect.ClassTag

object Quantizer extends Quantable {
  val registerMaps = new HashMap[String, Quantable]()

  init()

  override def quantize[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val className = model.getClass.getName

    val quantizedModel = if (registerMaps.contains(className)) {
      registerMaps(className).quantize(model)
    } else {
      model match {
        case container: Container[Activity, Activity, T] =>
          container match {
            case graph: Graph[T] => GraphQuantizer.quantize(graph)
            case _ => ContainerQuantizer.quantize(container)
          }
        /**
        case container: Container[_, _, _] => // TODO scala will throw compling exception
          container match {
            case graph: Graph[_] => GraphQuantizer.quantize(model)
            case _ => ContainerQuantizer.quantize(model)
          }
         */
        case cell if cell.isInstanceOf[Cell[T]] =>
          // because Cell[T] extends AbstractModule[Table, Table, T], and the Table is a class,
          // which is not as same as trait Tensor. So if we use this form:
          //   case cell: Cell[T] => CellQuantizer.quantize(cell)
          // scalac will throw an compiler error.
          CellQuantizer.quantize(cell)
        case default => ModuleQuantizer.quantize(model)
      }
    }

    quantizedModel
  }

  private def init(): Unit = {
    registerModules()
  }

  private def registerModule(name: String, module: Quantable): Unit = {
    require(!registerMaps.contains(name), s"Module: $name has been registered.")
    registerMaps(name) = module
  }

  private def registerModules(): Unit = {
    registerModule("com.intel.analytics.bigdl.nn.SpatialConvolution",
      nn.SpatialConvolution)
    registerModule("com.intel.analytics.bigdl.nn.SpatialDilatedConvolution",
      nn.SpatialDilatedConvolution)
    registerModule("com.intel.analytics.bigdl.nn.Linear", nn.Linear)

    registerModule("com.intel.analytics.bigdl.nn.TimeDistributed", nn.TimeDistributed)
    registerModule("com.intel.analytics.bigdl.nn.Recurrent", nn.Recurrent)
    registerModule("com.intel.analytics.bigdl.nn.BiRecurrent", nn.BiRecurrent)
//    registerModule("com.intel.analytics.bigdl.nn.LSTMPeephole", nn.LSTMPeephole)
    registerModule("com.intel.analytics.bigdl.nn.GRU", nn.GRU)
  }
}

object ContainerQuantizer extends Quantable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val container = module.asInstanceOf[Container[Activity, Activity, T]]
    for (i <- container.modules.indices) {
      val currModule = container.modules(i)
      container.modules(i) = Quantizer.quantize(currModule)
    }
    container
  }
}

object CellQuantizer extends Quantable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val cell = module.asInstanceOf[Cell[T]]
    cell.cell = Quantizer.quantize(cell.cell)
    if (cell.preTopology != null) {
      cell.preTopology = Quantizer.quantize(cell.preTopology)
    }
    cell
  }
}

object GraphQuantizer extends Quantable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val graph = module.asInstanceOf[Graph[T]]
    val sortedNodes = graph.getForwardExecutions
    val inputIndexes = graph.inputs.map(i => sortedNodes.indexOf(i))
    val outputIndexes = graph.getOutputs.map(i => sortedNodes.indexOf(i))

    for (i <- sortedNodes.indices) {
      val currNode = sortedNodes(i)
      val currModule = currNode.element
      val waitedModule = Quantizer.quantize(currModule)

      if (waitedModule != currModule) {
        replaceNode(currNode, waitedModule.asInstanceOf[ModuleNode[T]], sortedNodes, i)
      }
    }

    val inputs = new Array[ANode[T]](inputIndexes.length)

    for (i <- inputIndexes.indices) {
      inputs(i) = sortedNodes(inputIndexes(i))
    }

    // because all outputs point to dummy nodes, we should filter these nodes as outputs of Graph
    val outputs = new Array[ANode[T]](outputIndexes.length)

    for (i <- outputIndexes.indices) {
      outputs(i) = sortedNodes(outputIndexes(i))
    }

    // delete all dummy nodes
//    outputs.foreach { node =>
//      node.nextNodes.zipWithIndex.filter(_._1.element.isInstanceOf[Dummy[T]])
//              .foreach(x => node.nextNodes.asInstanceOf[ArrayBuffer[ANode[T]]].remove(x._2))
//    }

    // create a new Graph, much simpler than replacing others in the old graph
    Graph[T](inputs, outputs)
  }
}

object ModuleQuantizer extends Quantable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    module
  }
}
