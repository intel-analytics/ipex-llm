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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, SpatialConvolution, VariableFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class ConvolutionDnn[T: ClassTag](
     val nInputPlane: Int, // The number of expected input planes in the image given into forward()
     val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
     val kernelW: Int, // The kernel width of the convolution
     val kernelH: Int, // The kernel height of the convolution
     val strideW: Int = 1, // The step of the convolution in the width dimension.
     val strideH: Int = 1, // The step of the convolution in the height dimension
     val padW: Int = 0, // The additional zeros added per width to the input planes.
     val padH: Int = 0, // The additional zeros added per height to the input planes.
     val nGroup: Int = 1, // Kernel group number
     val initWeight: Tensor[Float] = null,
     val initBias: Tensor[Float] = null,
     val initGradWeight: Tensor[Float] = null,
     val initGradBias: Tensor[Float] = null,
     val withBias: Boolean = true,
     val format: DataFormat = DataFormat.NCHW
   )(implicit ev: TensorNumeric[T])
  extends TensorModule[Float] with Initializable {

  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L

  private val internal_format = MklDnn.MemoryFormat.any
  private val input_format = this.format.value match {
    case "NHWC" => MklDnn.MemoryFormat.nhwc
    case "NCHW" => MklDnn.MemoryFormat.nchw
  }
  this.output_format = this.input_format

  val internal_input = Tensor[Float]()
  val internal_gradOutput = Tensor[Float]()
  val internal_weights = Tensor[Float]()
  val internal_gradWeight = Tensor[Float]()
  private val original_gradWeights = Tensor[Float]()
  private val original_gradBias = Tensor[Float]()

  require(nOutputPlane % nGroup == 0, s"Number of input channels " +
    s"should be multiples of group " +
    s"number of input channels ${nInputPlane}, " +
    s"group ${nGroup}.")
  require(nOutputPlane % nGroup == 0,
    "Number of output channels " +
      "should be multiples of group " +
      s"(number of output channels ${nOutputPlane}, " +
      s"group ${nGroup}).")

  if (nGroup != 1) {
    require(format == DataFormat.NCHW, "group convolution is not supported in NHWC format " )
  }
  require((padW >= 0 && padH >= 0) || (padW == -1 && padH == -1),
    s"Illegal padding configuration (padW: $padW, padH: $padH)")

  private var weightDnnFormat = MklDnn.MemoryFormat.oihw
  private val weightShape = format match {
    case DataFormat.NCHW =>
      if (nGroup == 1) {
        weightDnnFormat = MklDnn.MemoryFormat.oihw
        Array(nOutputPlane, nInputPlane, kernelH, kernelW)
      } else {
        weightDnnFormat = MklDnn.MemoryFormat.goihw
        Array(nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
      }
    case DataFormat.NHWC =>
      weightDnnFormat = MklDnn.MemoryFormat.hwio
      Array(kernelH, kernelW, nInputPlane, nOutputPlane)
  }

  private val weightFormat = format match {
    case DataFormat.NCHW =>
      VariableFormat.GP_OUT_IN_KW_KH
    case DataFormat.NHWC =>
      VariableFormat.GP_KH_KW_IN_OUT
  }

  val weight: Tensor[Float] = if (initWeight != null) {
    initWeight
  } else {
    Tensor[Float](weightShape)
  }

  val bias: Tensor[Float] = if (!withBias) null
  else if (initBias != null) initBias else Tensor[Float](nOutputPlane)

  val gradWeight: Tensor[Float] = if (initGradWeight != null) {
    initGradWeight
  } else {
    Tensor[Float](weightShape)
  }

  val gradBias: Tensor[Float] = if (!withBias) null
  else if (initGradBias != null) initGradBias else Tensor[Float](nOutputPlane)

  {
    val stdv = 1.0 / math.sqrt(kernelW * kernelH * nInputPlane)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = if (withBias) RandomUniform(-stdv, stdv)
    else null
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, weightFormat)
    }
    if (withBias && initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }
    zeroGradParameters()
  }

  private val weightSize = weight.size()
  private val dimWeight = weight.dim()
  private val biasSize = bias.size()
  private val strides = Array(strideW, strideH)
  private val padding = Array(padH, padW)
  private val dataType = MklDnn.DataType.f32

  @transient
  var fwd_pd: Long = 0L
  @transient
  var src_memory : Long = 0L
  @transient
  var src_md : Long = 0L
  @transient
  var internal_src_memory : Long = 0L
  @transient
  var weights_memory: Long = 0L
  @transient
  var weights_md: Long = 0L
  @transient
  var internal_weights_memory : Long = 0L
  @transient
  var bias_memory: Long = 0L
  @transient
  var bias_md : Long = 0L
  @transient
  private var dst_memory: Long = 0L
  @transient
  private var conv_fwd: Long = 0L
  @transient
  private var conv_bwd: Long = 0L
  @transient
  private var conv_acc: Long = 0L
  @transient
  private var input_size: Array[Int] = null
  @transient
  private var inputElement : Int = 0
  @transient
  private var gradInput_memory: Long = 0L
  @transient
  private var gradOutput_memory: Long = 0L
  @transient
  private var gradWeights_memory: Long = 0L
  @transient
  private var gradBias_memory: Long = 0L
  @transient
  private var gradOutput_md: Long = 0L
  @transient
  private var  reorder_gradOutput: Long = 0L
  @transient
  private var reorder_src: Long = 0L
  @transient
  private var reorder_weights: Long = 0L
  @transient
  private var reorder_gradWeights: Long = 0L
  @transient
  private var internal_gradOutput_memory: Long = 0L

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    if (inputElement != input.nElement()) {
      require(input.dim() == 4 && input.isContiguous())
      val (dimHeight, dimWidth, channelDim) = format.getHWCDims(input.dim())
      require(input.size(channelDim) == nInputPlane, s"input channel size " +
        s"${input.size(channelDim)} is not the same as nInputPlane $nInputPlane")

      val input_size = input.size()
      val dim1 = math.max(0, (input_size(2) - kernelW)/strideW) + 1
      val dim2 = math.max(0, (input_size(3) - kernelH)/strideH) + 1
      val dst_sizes = Array(input_size(0), nOutputPlane, dim1, dim2)
      output.resize(dst_sizes)

      // create memory desc, for input
      if (input.getPrimitiveDesc() != 0L) {
        val input_pd = input.getPrimitiveDesc()
        src_memory = MklDnn.PrimitiveCreate0(input_pd)
        src_md = MklDnnOps.primitiveDescQueryMemory(input_pd)
      } else {
        src_memory = MklDnnOps.initDataMemory(input.dim(), input_size, this.input_format, dataType, engine)
        src_md = MklDnnOps.memoryDescInit(input.dim(), input_size, dataType, this.internal_format)
      }

      // for weight
      weights_memory = MklDnnOps.initDataMemory(dimWeight, weightSize, weightDnnFormat, dataType, engine)
      weights_md = MklDnnOps.memoryDescInit(dimWeight, weightSize, dataType, this.internal_format)

      // for bias
      bias_md = MklDnnOps.memoryDescInit(1, biasSize, dataType, MklDnn.MemoryFormat.x)
      bias_memory = MklDnnOps.createMemoryPrimitive(bias_md, engine)

      // for output
      val dst_md = MklDnnOps.memoryDescInit(output.dim(), dst_sizes, dataType, this.internal_format)

      /* create a convolution */
      val conv_desc = MklDnnOps.convForwardDescInit(
                                  MklDnn.PropKind.forward, MklDnn.AlgKind.convolutionDirect,
                                  src_md, weights_md, bias_md, dst_md,
                                  strides, padding, padding, MklDnn.PaddingKind.mkldnnPaddingZero)
      fwd_pd = MklDnnOps.primitiveDescCreate(conv_desc, engine, 0L)

      /* create reorder primitives between user data and convolution srcs */
      var reorder_src_memory: Long = 0L
      var reorder_weights_memory: Long = 0L
      if (this.internal_format != this.input_format) {
        val internal_src_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.src_pd, 0)
        val res = MklDnnOps.prepareReorder(src_memory, internal_src_pd, true)
        reorder_src = res._1
        reorder_src_memory = res._2
      }
      if (this.internal_format != this.input_format) {
        val internal_weights_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.weights_pd, 0)
        val res = MklDnnOps.prepareReorder(weights_memory, internal_weights_pd, true)
        reorder_weights = res._1
        reorder_weights_memory = res._2
      }

      /* create memory for dst data, we don't need to reorder it to user data */
      val dst_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.dst_pd, 0)
      dst_memory = MklDnn.PrimitiveCreate0(dst_pd)
      output.setPrimitiveDesc(dst_pd)

      internal_src_memory = if (reorder_src_memory == 0L) {
        src_memory
      } else {
        println("updateOutput input reorder")
        reorder_src_memory
      }
      internal_weights_memory = if (reorder_weights_memory == 0L) {
        weights_memory
      } else {
        println("updateOutput weight reorder")
        reorder_weights_memory
      }

      val conv_inputs = Array(internal_src_memory, internal_weights_memory, bias_memory)
      val conv_outputs = Array(dst_memory)
      val indexes = Array(0, 0, 0)
      conv_fwd = MklDnnOps.primitiveCreate2(fwd_pd, conv_inputs, indexes, 3, conv_outputs, 1)
    }

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, weights_memory, bias_memory, dst_memory, internal_src_memory, internal_weights_memory)
    val buffer = Array(input, weight, bias, output, input, weight)
    val stream_fwd = new ArrayBuffer[Long]
    if (reorder_src != 0L) stream_fwd.append(reorder_src)
    if (reorder_weights != 0L) stream_fwd.append(reorder_weights)
    stream_fwd.append(conv_fwd)
    val n_fwd = stream_fwd.length
    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd.toArray, n_fwd, memoryPrimitives, buffer)

    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    if (inputElement != input.nElement()) {
      if (gradOutput.getPrimitiveDesc() != 0L) {
        val gradOutput_pd = gradOutput.getPrimitiveDesc()
        gradOutput_memory = MklDnn.PrimitiveCreate0(gradOutput_pd)
        gradOutput_md = MklDnnOps.primitiveDescQueryMemory(gradOutput_pd)
      } else {
        gradOutput_memory = MklDnnOps.initDataMemory(gradOutput.dim(), gradOutput.size(), this.input_format, dataType, engine)
        gradOutput_md = MklDnnOps.memoryDescInit(gradOutput.dim(), gradOutput.size(), dataType, this.internal_format)
      }

      // for gradInput
      gradInput.resizeAs(input)
      val gradInput_md = MklDnnOps.memoryDescInit(input.dim(), input.size(), dataType, this.internal_format)

      val bwd_data_desc = MklDnnOps.convBackwardDataDescInit(
        MklDnn.AlgKind.convolutionDirect, gradInput_md, weights_md,
        gradOutput_md, strides, padding, padding,
        MklDnn.PaddingKind.mkldnnPaddingZero)
      val bwd_data_pd = MklDnnOps.primitiveDescCreate(bwd_data_desc, engine, fwd_pd)

      /* create memory primities for relu gradInput */
      val gradInput_pd = MklDnnOps.primitiveDescQueryPd(bwd_data_pd, MklDnn.Query.diff_src_pd, 0)
      gradInput_memory = MklDnn.PrimitiveCreate0(gradInput_pd)
      gradInput.setPrimitiveDesc(gradInput_pd)

      /* create reorder primitives between user gradOutput and convolution gradOutput */
      var reorder_gradOutput_memory : Long = 0L
      if (this.internal_format != this.input_format) {
        val internal_pd = MklDnnOps.primitiveDescQueryPd(bwd_data_pd, MklDnn.Query.diff_dst_pd, 0)
        val res = MklDnnOps.prepareReorder(gradOutput_memory, internal_pd, true)
        reorder_gradOutput = res._1
        reorder_gradOutput_memory = res._2
      }

      internal_gradOutput_memory = if (reorder_gradOutput_memory == 0L) {
        gradOutput_memory
      } else {
        println("updateGradInput reorder")
        reorder_gradOutput_memory
      }

      val conv_inputs = Array(internal_gradOutput_memory, internal_weights_memory, internal_src_memory)
      val conv_outputs = Array(gradInput_memory)
      val indexes = Array(0, 0, 0)

      /* finally create a convolution primitive */
     conv_bwd = MklDnnOps.primitiveCreate2(bwd_data_pd, conv_inputs, indexes, 3, conv_outputs, 1)
    }
    /* build a simple net */
    val memoryPrimitives = Array(internal_weights_memory, internal_src_memory, internal_gradOutput_memory, gradInput_memory, gradOutput_memory)
    val buffer = Array(weight, input, gradOutput, gradInput, gradOutput)
    val stream_bwd = new ArrayBuffer[Long]
    if (reorder_gradOutput != 0L) stream_bwd.append(reorder_gradOutput)
    stream_bwd.append(conv_bwd)
    val n_bwd = stream_bwd.length
    MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd, memoryPrimitives, buffer)

    gradInput
  }

 override def accGradParameters(input: Tensor[Float], gradOutput: Tensor[Float]): Unit = {
   if (inputElement != input.nElement()) {
     // for gradWeights
     val gradWeights_md = MklDnnOps.memoryDescInit(dimWeight, weightSize, dataType, this.internal_format)
     gradWeights_memory = MklDnnOps.initDataMemory(dimWeight, weightSize, weightDnnFormat, dataType, engine)

     // for gradBias
     val gradBias_md = MklDnnOps.memoryDescInit(1, biasSize, dataType, MklDnn.MemoryFormat.x)
     gradBias_memory = MklDnnOps.createMemoryPrimitive(gradBias_md, engine)

     val weights_desc = MklDnnOps.convBackwardWeightsDescInit(
       MklDnn.AlgKind.convolutionDirect, src_md, gradWeights_md,
       gradBias_md, gradOutput_md, strides, padding,
       padding, MklDnn.PaddingKind.mkldnnPaddingZero)
     val weights_pd = MklDnnOps.primitiveDescCreate(weights_desc, engine, fwd_pd)


     var reorder_gradWeights : Long = 0L
     var reorder_gradWeights_memory : Long = 0L
     // reorder gradWeight
     if (this.internal_format != this.input_format) {
       val internal_pd = MklDnnOps.primitiveDescQueryPd(weights_pd, MklDnn.Query.diff_weights_pd, 0)
       val res = MklDnnOps.prepareReorder(gradWeights_memory, internal_pd, true)
       reorder_gradWeights = res._1
       reorder_gradWeights_memory = res._2
     }

     if (reorder_gradWeights_memory != 0L) {
       internal_gradWeight.resizeAs(gradWeight)
       MklDnnOps.memorySetDataHandle(reorder_gradWeights_memory, internal_gradWeight, 0)

     }

     val internal_gradWeights_memory = if (reorder_gradWeights_memory == 0L) {
       gradWeights_memory
     } else {
       println("accGradParameters reorder")
       reorder_gradWeights_memory
     }

     val conv_inputs = Array(internal_src_memory, internal_gradOutput_memory)
     val conv_outputs = Array(internal_gradWeights_memory, gradBias_memory)
     val indexes = Array(0, 0)

     /* finally create a convolution primitive */
     conv_acc = MklDnnOps.primitiveCreate2(weights_pd, conv_inputs, indexes, 2, conv_outputs, 2)
   }
   /* build a simple net */
   // keep original data
   original_gradWeights.resizeAs(gradWeight).copy(gradWeight)
   original_gradBias.resizeAs(gradBias).copy(gradBias)

   val memoryPrimitives = Array(gradWeights_memory, gradBias_memory, internal_src_memory, internal_gradOutput_memory)
   val buffer = Array(gradWeight, gradBias, input, gradOutput)
   val stream_bwd = new ArrayBuffer[Long]
   if (reorder_gradWeights != 0L) stream_bwd.append(reorder_gradWeights)
   stream_bwd.append(conv_acc)
   val n_bwd = stream_bwd.length
   MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd, memoryPrimitives, buffer)

   gradWeight.add(original_gradWeights)
   gradBias.add(original_gradBias)
 }

  override def updateParameters(learningRate: Float): Unit = {
    weight.map(gradWeight, (a, b) => a - learningRate * b)
    if (withBias) {
      bias.map(gradBias, (a, b) => a - learningRate * b)
    }
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    if (withBias) {
      gradBias.zero()
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    if (withBias) {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    } else {
      (Array(this.weight), Array(this.gradWeight))
    }
  }

  override def getParametersTable(): Table = {
    if (withBias) {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias))
    } else {
      T(getName() -> T("weight" -> weight,
        "gradWeight" -> gradWeight))
    }
  }
}

object ConvolutionDnn {
  def apply(
   nInputPlane: Int,
   nOutputPlane: Int,
   kW: Int,
   kH: Int,
   dW: Int = 1,
   dH: Int = 1,
   padW: Int = 0,
   padH: Int = 0,
   nGroup: Int = 1,
   initWeight: Tensor[Float] = null,
   initBias: Tensor[Float] = null,
   initGradWeight: Tensor[Float] = null,
   initGradBias: Tensor[Float] = null,
   withBias: Boolean = true,
   format: DataFormat = DataFormat.NCHW): ConvolutionDnn[Float] = {
    new ConvolutionDnn[Float](nInputPlane, nOutputPlane, kW, kH, dW,
    dH, padW, padH, nGroup, initWeight, initBias, initGradWeight, initGradBias, withBias, format)
  }
}
