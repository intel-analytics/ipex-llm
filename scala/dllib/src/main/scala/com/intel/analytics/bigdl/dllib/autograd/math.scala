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

package com.intel.analytics.zoo.pipeline.api.autograd

import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, InferShape}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{ModuleSerializable, ModuleSerializer}
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import com.intel.analytics.bigdl.{nn => bnn}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models._

import scala.reflect.ClassTag

object AutoGrad {

  val EPSILON = 10e-8

  // TODO: Get the nDim from Variable
  private def normalizeAxis(axis: Int, nDim: Int = -1) = {
    if (axis < 0) {
      throw new IllegalArgumentException("We don't support axis < 0 for now") // axis + nDim
    } else {
      axis
    }
  }

  def abs[T: ClassTag](a: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](bnn.Abs[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(a.node))
  }

  def sum[T: ClassTag](a: Variable[T], axis: Int = 0, keepdims: Boolean = false)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](bnn.Sum[T](dimension = normalizeAxis(axis) + 1,
        squeeze = !keepdims).asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(a.node))
  }

  def clip[T: ClassTag](a: Variable[T], min: Double, max: Double)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](
        bnn.HardTanh[T](minValue = min,
          maxValue = max).asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(a.node))
  }

  def square[T: ClassTag](a: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Square[T]().inputs(a.node))
  }


  def sqrt[T: ClassTag](a: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Sqrt[T]().inputs(a.node))
  }

  def maximum[T: ClassTag](x: Variable[T], y: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](
        bnn.CMaxTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node, y.node))
  }

  def maximum[T: ClassTag](x: Variable[T], y: Double)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    clip(x, min = y, max = Double.MaxValue)
  }

  /**
   * Mean of a tensor, alongside the specified axis.
   * @param axis axis to compute the mean. 0-based indexed.
   * @param keepDims A boolean, whether to keep the dimensions or not.
   *If `keepdims` is `False`, the rank of the tensor is reduced
   *by 1. If `keep_dims` is `True`,
   *the reduced dimensions are retained with length 1.
   * @return
   *         A tensor with the mean of elements of `x`.
   */
  def mean[T: ClassTag](x: Variable[T], axis: Int = 0, keepDims: Boolean = false)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](bnn.Mean[T](dimension = normalizeAxis(axis) + 1,
        squeeze = !keepDims).asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node))
  }

  def log[T: ClassTag](a: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Log[T]().inputs(a.node))
  }

  def epsilon[T: ClassTag]()(
      implicit ev: TensorNumeric[T]): Double = {
    EPSILON
  }

  def exp[T: ClassTag](x: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Exp[T]().inputs(x.node))
  }

  def pow[T: ClassTag](x: Variable[T], a: Double)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Power[T](a).inputs(x.node))
  }

  def softsign[T: ClassTag](a: Variable[T])(
    implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper(bnn.SoftSign[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(a.node))
  }

  def softplus[T: ClassTag](a: Variable[T])(
    implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper(bnn.SoftPlus[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(a.node))
  }
}

object Variable extends {

  private[zoo] def apply[T: ClassTag](node: ModuleNode[T])(
      implicit ev: TensorNumeric[T]) = {
    new Variable[T](node)
  }

  def apply[T: ClassTag](inputShape: Shape)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    new Variable[T](Input(inputShape))
  }
}

class Variable[T: ClassTag] private[zoo] (val node: ModuleNode[T], var name: String = null)(
    implicit ev: TensorNumeric[T]) extends Serializable {

  if (name == null) {
    name = node.element.getName()
  } else {
    node.element.setName(name)
  }

  require(node.element.isInstanceOf[KerasLayer[Activity, Activity, T]])
  require(node.element.asInstanceOf[InferShape].getOutputShape() != null)

  private[zoo] def getRoots(): Array[ModuleNode[T]] = {
    val dfs = this.node.graph(reverse = true).DFS.toList.reverse
    val roots = dfs.filter(_.prevNodes.size == 0).toArray[ModuleNode[T]]
    roots
  }


  private[zoo] def toGraph(inputs: Array[Variable[T]]): Model[T] = {
    Model(input = inputs.map(_.node), output = this.node)
  }

  // "tensorboard --logdir path" to visualize this Variable
  private[zoo] def toTensorBoard(path: String) = {
    def toGraph(): Model[T] = {
      val dfs = this.node.graph(reverse = true).DFS.toList.reverse
      val roots = dfs.filter(_.prevNodes.size == 0).toArray
      Model(input = roots, output = this.node)
    }
    val props = System.getProperties()
    val tmp: Option[String] = if (props.contains("bigdl.localMode")) {
      Some(props.getProperty("bigdl.localMode"))
    } else {
      None
    }
    props.setProperty("bigdl.localMode", "true")
    Engine.init
    toGraph().saveGraphTopology(path)  // TODO: add saveGraphTopology
    if (!tmp.isEmpty) {
      props.setProperty("bigdl.localMode", tmp.get)
    } else {
      props.remove("bigdl.localMode")
    }
  }

  // scalastyle:off
  def +(a: Variable[T]): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](bnn.CAddTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    val (x, y) = broadcast(this, a)
    Variable(o.inputs(Array(x.node, y.node)))
  }

  def +(a: Double): Variable[T] = {
    Variable(AddConstant[T](a).inputs(Array(this.node)))
  }

  def -(a: Variable[T]): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](bnn.Negative[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    val neg = new Variable(o.inputs(a.node))
    val (x, y) = broadcast(this, neg)
    x + y
  }

  def -(a: Double): Variable[T] = {
    Variable(AddConstant[T](-a).inputs(Array(this.node)))
  }

  def unary_-(): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](bnn.Negative[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(this.node))
  }

  def *(a: Variable[T]): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](bnn.CMulTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    val (x, y) = broadcast(this, a)
    Variable(o.inputs(Array(x.node, y.node)))
  }

  def *(a: Double): Variable[T] = {
    Variable(MulConstant[T](a).inputs(Array(this.node)))
  }

  def /(other: Variable[T]): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](bnn.CDivTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    val (x, y) = broadcast(this, other)
    Variable(o.inputs(Array(x.node, y.node)))
  }

  def /(a: Double): Variable[T] = {
    this * (1/a)
  }

  /**
   * Delete the singleton dimension(s).
   * The batch dimension needs to be unchanged.
   * For example, if input has size (2, 1, 3, 4, 1):
   * Squeeze(dim = 1) will give output size (2, 3, 4, 1)
   * Squeeze(dims = null) will give output size (2, 3, 4)
   */
  def squeeze(dim: Int): Variable[T] = {
    val layer = Squeeze[T](dim)
    Variable(layer.inputs(this.node))
  }

  /**
   * Narrow the input with the number of dimensions not being reduced.
   * The batch dimension needs to be unchanged.
   * For example, if input is:
   * 1 2 3
   * 4 5 6
   * Narrow(1, 1, 2) will give output
   * 2 3
   * 5 6
   * Narrow(1, 2, -1) will give output
   * 3
   * 6
   */
  def narrow(dim: Int, startIndex: Int, length: Int): Variable[T] = {
    val layer = Narrow[T](dim = dim,
      offset = startIndex,
      length = length)
    Variable(layer.inputs(this.node))
  }

  /**
   * Select an index of the input in the given dim and return the subset part.
   * The batch dimension needs to be unchanged.
   * The selected dim would be remove after this operation.
   * For example, if input is:
   * 1 2 3
   * 4 5 6
   * Select(1, 1) will give output [2 5]
   * Select(1, -1) will give output [3 6]
   *
   * @param dim The dimension to select. 0-based index. Cannot select the batch dimension.
   *            -1 means the last dimension of the input.
   * @param index The index of the dimension to be selected. 0-based index.
   *              -1 means the last dimension of the input.
   */
  def indexSelect(dim: Int, index: Int): Variable[T] = {
    val layer = Select[T](dim = dim,
      index = index)
    Variable(layer.inputs(this.node))
  }

  private[zoo] def broadcast(x: Variable[T], y: Variable[T]): (Variable[T], Variable[T]) = {
    val yShape = y.getOutputShape().toSingle()
    val xShape = x.getOutputShape().toSingle()
    require(xShape.size == yShape.size,
      s"The two variables should have the same dims," +
        s"but got: ${xShape.size} and ${yShape.size}")
    var xx = x
    var yy = y
    var i = yShape.length - 1
    while (i >= 1) { // Ignore the batch dim
      if (yShape(i) != xShape(i)) {
        if (yShape(i) == 1) {
          yy = yy.replicate(i, xShape(i))
        } else if (xShape(i) == 1) {
          xx = xx.replicate(i, yShape(i))
        } else {
          throw new IllegalArgumentException(
            s"Shape mismatch: x - ${xShape}, y -${yShape}")
        }
      }
      i -= 1
    }
    (xx, yy)
  }
  // scalastyle:on

  def replicate(axis: Int, times: Int): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](
        bnn.Replicate[T](dim = axis + 1,
          nFeatures = times).asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(this.node))
  }

  def getOutputShape(): Shape = {
    this.node.element.getOutputShape()
  }

  def getInputShape(): Shape = {
    this.node.element.getInputShape()
  }

  private[zoo] def getDummyTensor(fillValue: T, batchSize: Int): Tensor[T] = {
    Tensor[T](getInputShape().copyAndUpdate(0, batchSize).toSingle().toArray).fill(fillValue)
  }
}

