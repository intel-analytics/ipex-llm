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
package com.intel.analytics.bigdl.utils.caffe

import scala.collection.JavaConverters._
import caffe.Caffe
import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.{BlobProto, PoolingParameter, _}
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 *  [[Converter]] implementation for caffe recommended LayerParameter conversion
 */
class LayerConverter[T: ClassTag](implicit ev: TensorNumeric[T]) extends Converter[T]{

  override protected def fromCaffeConvolution(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val name = getLayerName(layer)
    val param = getConvolutionParam(layer).get
    val group = if (param.getGroup == 0)  1 else param.getGroup
    val  weightBlob = getBlob(layer, 0).get
    val biasBlob = getBlob(layer, 1)
    if (!biasBlob.isDefined) {
      throw new RuntimeException(s"${getLayerName(layer)} without bias is not supported now")
    }
    val nInputPlane = if (weightBlob.hasShape) weightBlob.getShape.getDim(1)
    else weightBlob.getChannels * group
    val nOutPlane = if (weightBlob.hasShape) weightBlob.getShape.getDim(0)
    else weightBlob.getNum
    var kw = param.getKernelW
    var kh = param.getKernelH
    var dw = param.getStrideW
    var dh = param.getStrideH
    if (kw ==0 || kh == 0) {
      kw = param.getKernelSize(0)
      kh = kw
    }
    if (dw == 0 || dh == 0) {
      if (param.getStrideList.size() != 0) {
        dw = param.getStride(0)
        dh = dw
      } else {
        // use default values if not found
        dw = 1
        dh = 1
      }
    }
    var pw = param.getPadW
    var ph = param.getPadH
    if (pw == 0 || ph == 0) {
      if (param.getPadList.size() != 0) {
        pw = param.getPad(0)
        ph = pw
      }
    }
    Seq(SpatialConvolution[T](nInputPlane.toInt, nOutPlane.toInt,
      kw, kh, dw, dh, pw, ph, group).setName(getLayerName(layer)).apply())
  }

  override protected def fromCaffeInnerProduct(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getInnerProductParam(layer).get
    val withBias = param.getBiasTerm
    val layerName = getLayerName(layer)
    val weightBlob = getBlob(layer.asInstanceOf[LayerParameter], 0).get
    var nInputPlane = 0
    if (weightBlob.hasShape) {
      nInputPlane = weightBlob.getShape.getDim(1).toInt
    }
    else {
      nInputPlane = weightBlob.getWidth
    }
    val nOutputPlane = param.getNumOutput
    val linear = Linear[T](nInputPlane, nOutputPlane, withBias = withBias).setName(layerName)
    val node = linear.apply()
    if(nInputPlane != nOutputPlane) {
      // Construct a view layer in between
      val view = View[T](nInputPlane).apply()
      view -> node
      Seq(view, node)
    } else {
      Seq(node)
    }
  }

  override protected def fromCaffeBatchNormalization(layer : GeneratedMessage) :
  Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getBatchNormParam
    val eps = param.getEps
    Seq(BatchNormalization[T](3, eps).apply())
  }

  override protected def fromCaffeELU(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getEluParam
    var alpha = 1.0
    if (param.hasAlpha) alpha = param.getAlpha
    Seq(ELU[T](alpha).apply())
  }

  override protected def fromCaffeReshape(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getReshapeParam
    val shapeSize = param.getShape.getDimList.toArray.asInstanceOf[Array[Int]]
    Seq(Reshape[T](shapeSize).setName(getLayerName(layer)).apply())
  }

  override protected def fromCaffeScale(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getScaleParam
    val layerName = getLayerName(layer)
    // second blob as weight for scale
    val weightBlob = getBlob(layer, 1)
    if (weightBlob.isDefined) {
      val blob = weightBlob.get
      val size = blob.getShape.getDimList.toArray.asInstanceOf[Array[Int]]
      Seq(Scale[T](size).setName(layerName).apply())
    } else {
      val inputBlob = getBlob(layer, 0).get
      val shape = inputBlob.getShape
      val axis = param.getAxis
      var numOfAxis = param.getNumAxes
      if (numOfAxis == -1) {
        numOfAxis = shape.getDimList.size() - 1
      } else {
        numOfAxis = numOfAxis + axis
      }
      val size = shape.getDimList.subList(axis, numOfAxis).asInstanceOf[Array[Int]]
      Seq(Scale[T](size).setName(layerName).apply())
    }
  }

  override protected def fromCaffeBias(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getBiasParam
    // input blob
    val weightBlob = getBlob(layer, 0)
    val size = weightBlob.get.getShape.getDimList.toArray().asInstanceOf[Array[Int]].product
    Seq(Add[T](size).setName(getLayerName(layer)).apply())
  }

  override protected def fromCaffeTile(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getTileParam
    val axis = param.getAxis
    val tiles = param.getTiles
    Seq(Replicate[T](tiles, axis).setName(getLayerName(layer)).apply())
  }

  override protected def getLayerName(layer : GeneratedMessage) : String = {
    layer.asInstanceOf[LayerParameter].getName
  }

  override protected def getLayerType(layer : GeneratedMessage) : String = {
    layer.asInstanceOf[LayerParameter].getType
  }

  protected def getConvolutionParam(layer : GeneratedMessage): Option[ConvolutionParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getConvolutionParam)
  }

  override protected def getLRNParam(layer: GeneratedMessage): Option[LRNParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getLrnParam)
  }

  override protected def getPoolingParam(layer : GeneratedMessage): Option[PoolingParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getPoolingParam)
  }

  protected def getInnerProductParam(layer : GeneratedMessage): Option[InnerProductParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getInnerProductParam)
  }

  protected def getDropoutParam(layer : GeneratedMessage): Option[DropoutParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getDropoutParam)
  }

  protected def getConcatParam(layer : GeneratedMessage): Option[ConcatParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getConcatParam)
  }

  override protected def getPowerParam(layer : GeneratedMessage) : Option[PowerParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getPowerParam)
  }

  override protected def getThresholdParam(layer : GeneratedMessage): Option[ThresholdParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getThresholdParam)
  }

  override protected def getSliceParam(layer : GeneratedMessage): Option[SliceParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getSliceParam)
  }

  override protected def getEltWiseParam(layer : GeneratedMessage): Option[EltwiseParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getEltwiseParam)
  }

  private def getBlob(layer : GeneratedMessage, ind: Int): Option[Caffe.BlobProto] = {
    if (layer.asInstanceOf[LayerParameter].getBlobsCount > ind) {
      Some(layer.asInstanceOf[LayerParameter].getBlobs(ind))
    } else {
      None
    }
  }
}
