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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.quantized.Utils._
import com.intel.analytics.bigdl.nn.{Cell, Container, Graph}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, nn}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.reflect.ClassTag

object Quantizer extends Quantizable {
  val registerMaps = new HashMap[String, Quantizable]()

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

  private def registerModule(name: String, module: Quantizable): Unit = {
    require(!registerMaps.contains(name), s"Module: $name has been registered.")
    registerMaps(name) = module
  }

  private def registerModules(): Unit = {
    registerModule("com.intel.analytics.bigdl.nn.SpatialConvolution",
      nn.SpatialConvolution)
    registerModule("com.intel.analytics.bigdl.nn.SpatialDilatedConvolution",
      nn.SpatialDilatedConvolution)
    registerModule("com.intel.analytics.bigdl.nn.Linear", nn.Linear)
  }
}

object ContainerQuantizer extends Quantizable {
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

object CellQuantizer extends Quantizable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val cell = module.asInstanceOf[Cell[T]]
    cell.cell = Quantizer.quantize(cell.cell)
    cell
  }
}

object GraphQuantizer extends Quantizable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val graph = module.asInstanceOf[Graph[T]]
    val sortedNodes = graph.getForwardExecutions

    for (i <- sortedNodes.indices) {
      val currNode = sortedNodes(i)
      val currModule = currNode.element
      val waitedModule = Quantizer.quantize(currModule)

      if (waitedModule != currModule) {
        currNode.setElement(waitedModule)
      }
    }

    // modules in container need to rebuild
    graph.resetModules()
    // nodes in backward executions need to rebuild
    graph.buildBackwardGraph()

    graph
  }
}

object ModuleQuantizer extends Quantizable {
  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    module
  }
}
