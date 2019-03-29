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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, InferShape}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.{nn => bnn}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.{InternalCAddTable, InternalCMulTable, InternalExpand, InternalMM}
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
  /**
   * Element-wise absolute value.
   * @param x A variable.
   * @return A variable.
   */
  def abs[T: ClassTag](x: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](bnn.Abs[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node))
  }

  /**
   * Sum of the values in a variable, alongside the specified axis.
   * @param x A variable.
   * @param axis axis to compute the mean. 0-based indexed.
   * @param keepDims A boolean, whether to keep the dimensions or not.
   * If `keepDims` is `False`, the rank of the variable is reduced
   * by 1. If `keepDims` is `True`,
   * the reduced dimensions are retained with length 1.
   * @return A variable with the mean of elements of `x`.
   */
  def sum[T: ClassTag](x: Variable[T], axis: Int = 0, keepDims: Boolean = false)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](bnn.Sum[T](dimension = normalizeAxis(axis) + 1,
        squeeze = !keepDims).asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node))
  }

  /**
   * Element-wise value clipping.
   * @param x A variable.
   * @param min Double.
   * @param max Double.
   * @return A variable.
   */
  def clip[T: ClassTag](x: Variable[T], min: Double, max: Double)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](
        bnn.HardTanh[T](minValue = min,
          maxValue = max).asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node))
  }

  /**
   * Element-wise square.
   * @param x A variable.
   * @return A variable.
   */
  def square[T: ClassTag](x: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Square[T]().inputs(x.node))
  }

  /**
   * Element-wise square root.
   * @param x A variable.
   * @return A variable.
   */
  def sqrt[T: ClassTag](x: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Sqrt[T]().inputs(x.node))
  }

  /**
   * Element-wise maximum of two variables
   * @param x A variable.
   * @param y A variable.
   * @return A variable.
   */
  def maximum[T: ClassTag](x: Variable[T], y: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper[T](
        bnn.CMaxTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node, y.node))
  }

  /**
   * Element-wise maximum of two variables
   * @param x A variable.
   * @param y Double
   * @return A variable.
   */
  def maximum[T: ClassTag](x: Variable[T], y: Double)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    clip(x, min = y, max = Double.MaxValue)
  }

  /**
   * Mean of a tensor, alongside the specified axis.
   * @param axis axis to compute the mean. 0-based indexed.
   * @param keepDims A boolean, whether to keep the dimensions or not.
   *If `keepDims` is `False`, the rank of the tensor is reduced
   *by 1. If `keepDims` is `True`,
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

  /**
   * Element-wise log.
   * @param x A variable.
   * @return A variable.
   */
  def log[T: ClassTag](x: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Log[T]().inputs(x.node))
  }

  /**
   * Define the value of epsilon.
   * @return A value of type Double.
   */
  def epsilon[T: ClassTag]()(
      implicit ev: TensorNumeric[T]): Double = {
    EPSILON
  }

  /**
   * Element-wise exponential.
   * @param x A variable.
   * @return A variable.
   */
  def exp[T: ClassTag](x: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Exp[T]().inputs(x.node))
  }

  /**
   * Element-wise exponentiation.
   * @param x A variable.
   * @param a Double.
   * @return A variable.
   */
  def pow[T: ClassTag](x: Variable[T], a: Double)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(Power[T](a).inputs(x.node))
  }

  /**
   * Softsign of a variable.
   * @param x A variable.
   * @return A variable.
   */
  def softsign[T: ClassTag](x: Variable[T])(
    implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper(bnn.SoftSign[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node))
  }

  /**
   * Softplus of a variable.
   * @param x A variable.
   * @return A variable.
   */
  def softplus[T: ClassTag](x: Variable[T])(
    implicit ev: TensorNumeric[T]): Variable[T] = {
    val o: KerasLayer[Activity, Activity, T] =
      new KerasLayerWrapper(bnn.SoftPlus[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    Variable(o.inputs(x.node))
  }

  /**
   * Stacks a list of rank `R` tensors into a rank `R+1` tensor.
   * @param inputs: List of variables (tensors).
   * @param axis axis along which to perform stacking.
   */
  def stack[T: ClassTag](inputs: List[Variable[T]], axis: Int = 1)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val stacked = Variable(Merge.merge[T](inputs.map(expandDims(_, axis).node), mode = "concat",
      concatAxis = axis))
    contiguous(stacked)
  }

  /**
   * Adds a 1-sized dimension at index "axis".
   * The axis is 0 based and if you set the axis to 0, you would change the batch dim.
   * @param axis Position where to add a new axis.
   */
  def expandDims[T: ClassTag](x: Variable[T], axis: Int)(
      implicit ev: TensorNumeric[T]): Variable[T] = {
    val layer = ExpandDim(axis).asInstanceOf[AbstractModule[Activity, Activity, T]]
    val expanded = Variable(layer.inputs(x.node))
    contiguous(expanded)
  }

  /**
   * Turn the output and grad to be contiguous for the input Variable
   */
  def contiguous[T: ClassTag](input: Variable[T])(implicit ev: TensorNumeric[T]): Variable[T] = {
    val contiguousNode = new KerasLayerWrapper(
      bnn.Contiguous[T]().asInstanceOf[AbstractModule[Activity, Activity, T]]).inputs(input.node)
    Variable(contiguousNode)
  }

  /**
   * Module to perform matrix multiplication on two mini-batch inputs,
   * producing a mini-batch.
   *
   * @param x A variable.
   * @param y A variable.
   * @param axes Axes along which to perform multiplication.
   */
  def mm[T: ClassTag](
      x: Variable[T],
      y: Variable[T],
      axes: List[Int] = null)(implicit ev: TensorNumeric[T]): Variable[T] = {
    require(x.getOutputShape().isInstanceOf[SingleShape], "Only accept single shape")
    require(y.getOutputShape().isInstanceOf[SingleShape], "Only accept single shape")
    var xx = x
    var yy = y
    var yShape = yy.getOutputShape().toSingle()
    var xShape = xx.getOutputShape().toSingle()
    if (yShape.size > xShape.size) {
      xx = AutoGrad.expandDims(x, 0)
    } else if (yShape.size < xShape.size) {
      yy = AutoGrad.expandDims(y, 0)
    }
    xShape = xx.getOutputShape().toSingle()
    yShape = yy.getOutputShape().toSingle()
    var transposeX = false
    var transposeY = false
    var left = 0
    var right = 0
    if (xShape.length == 2 && yShape.length == 2) {
      left = 0
      right = 1
    } else if (xShape.length == 3 && yShape.length == 3) {
      left = 1
      right = 2
    } else if (xShape.length == 4 && yShape.length == 4) {
      left = 2
      right = 3
    } else if (xShape.length > 4 && yShape.length > 4) {
        throw new IllegalArgumentException(s"Only support 2D/3D/4D input for now," +
          s"but got [${xShape.mkString(",")}] and [${xShape.mkString(",")}]")
      }
    if (axes != null) {
      require(axes.length == 2, s"axes.length should be 2, but got: ${axes.length}")
      require(axes(0) >= left && axes(0) <= right,
        s"axes should between [$left, $right], not ${axes(0)}")
      require(axes(1) >= left && axes(1) <= right,
        s"axes should between [$left, $right], not ${axes(1)}")
      transposeX = if (axes(0) != xShape.length - 1) {true} else {false}
      transposeY = if (axes(1) == yShape.length - 1) {true} else {false}
    }

    val mm = InternalMM[T](transA = transposeX,
      transB = transposeY)
    val kmm = new KerasLayerWrapper[T](mm.asInstanceOf[AbstractModule[Activity, Activity, T]])

    if (xShape.length > 3 || yShape.length > 3) {
      TimeDistributed(kmm.asInstanceOf[KerasLayer[Activity, Tensor[T], T]]).from(xx, yy)
    } else kmm.from(xx, yy)
  }

  /**
   * Normalizes a tensor wrt the L2 norm alongside the specified axis.
   *
   * @param x A variable.
   * @param axis Axis along which to perform multiplication.
   */
  def l2Normalize[T: ClassTag](x: Variable[T], axis: Int)
      (implicit ev: TensorNumeric[T]): Variable[T] = {
    val l2Normalize = x / sqrt(maximum(sum(x * x, axis, keepDims = true), epsilon()))
    l2Normalize
  }

  /**
   * Operator that computes a dot product between samples in two tensors.
   *
   * @param x A variable.
   * @param y A variable.
   * @param axes Axes along which to perform multiplication.
   * @param normalize Whether to L2-normalize samples along the
   *                  dot product axis before taking the dot product.
   *                  If set to True, then the output of the dot product
   *                  is the cosine proximity between the two samples.
   */
  def batchDot[T: ClassTag](x: Variable[T], y: Variable[T],
                            axes: List[Int], normalize: Boolean = false)
      (implicit ev: TensorNumeric[T]): Variable[T] = {
  val xShape = x.getOutputShape().toSingle().toArray
  if (!normalize) {
    require(xShape.length == 2 || xShape.length == 3,
      s"Only support 2D and 3D for now, but got: ${xShape.length}")
    if (xShape.length == 2) {
      sum(x*y, axis = 1, keepDims = true)
    } else {
      mm(x, y, axes)
    }
  } else {
    val l2_x = l2Normalize(x, axes(0))
    val l2_y = l2Normalize(y, axes(1))
    batchDot(l2_x, l2_y, axes = axes)
    }
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

class Variable[T: ClassTag] private[zoo] (private[zoo] var node: ModuleNode[T],
    var name: String = null)(
    implicit ev: TensorNumeric[T]) extends Serializable {

  if (node != null) {
    if (name == null) {
      name = node.element.getName()
    } else {
      node.element.setName(name)
    }

    require(node.element.isInstanceOf[KerasLayer[Activity, Activity, T]])
    require(node.element.asInstanceOf[InferShape].getOutputShape() != null)
  }

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
      new KerasLayerWrapper[T](new InternalCAddTable[T, T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
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
      new KerasLayerWrapper[T](InternalCMulTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
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
    val dims = new Array[Int](1)
    dims(0) = dim
    squeeze(dims)
  }

  def squeeze(dims: Array[Int]): Variable[T] = {
    val blayer = if (dims == null){
       com.intel.analytics.bigdl.nn.Squeeze[T](null, batchMode = false)
    } else {
      com.intel.analytics.bigdl.nn.Squeeze[T](dims.map(x => x + 1), batchMode = false)
    }
    val klayer = new KerasLayerWrapper[T](blayer)
    Variable(klayer.inputs(this.node))
  }

  /**
   * Same as Narrow in torch.
   * Slice the input with the number of dimensions not being reduced.
   * The batch dimension needs to be unchanged.
   * For example, if input is:
   * 1 2 3
   * 4 5 6
   * slice(1, 1, 2) will give output
   * 2 3
   * 5 6
   * slice(1, 2, -1) will give output
   * 3
   * 6
   *  @param dim The dimension to narrow. 0-based index. Cannot narrow the batch dimension.
   *            -1 means the last dimension of the input.
   * @param startIndex Non-negative integer. The start index on the given dimension. 0-based index.
   * @param length The length to be sliced. Default is 1.
   */
  def slice(dim: Int, startIndex: Int, length: Int): Variable[T] = {
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
    var xx = x
    var yy = y

    var yShape = yy.getOutputShape().toSingle()
    var xShape = xx.getOutputShape().toSingle()

    if (yShape.size > xShape.size) {
      xx = AutoGrad.expandDims(xx, 0)
      xShape = xx.getOutputShape().toSingle()
    } else if (yShape.size < xShape.size) {
      yy = AutoGrad.expandDims(yy, 0)
      yShape = yy.getOutputShape().toSingle()
    }

    require(xShape.size == yShape.size,
      s"The two variables should have the same dims," +
        s"but got: ${x.getOutputShape().toSingle().mkString(",")}" +
        s"and ${y.getOutputShape().toSingle().mkString(",")}")

    // Ignore the batch dim
    val xElements = xShape.drop(1).reduceLeft(_+_)
    val yElements = yShape.drop(1).reduceLeft(_+_)
    // should not expand batch dim here as it's -1
    if (xElements < yElements) {
      xx = xx.expand(yShape)
    } else if (xElements > yElements) {
      yy = yy.expand(xShape)
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

  /**
   * Expand variable to configured size
   * @param sizes target variable sizes, dim whose size is -1 will be ignored
   */
  def expand(sizes: List[Int]): Variable[T] = {
    val o =
      new KerasLayerWrapper[T](
        InternalExpand[T](sizes.toArray).asInstanceOf[AbstractModule[Activity, Activity, T]])
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