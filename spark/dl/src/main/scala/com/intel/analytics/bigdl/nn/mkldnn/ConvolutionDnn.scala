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

import breeze.linalg.max
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.dmg.pmml.False

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
                                 val propagateBack: Boolean = true,
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
  this.output_format = input_format

  private var internal_inputFormat: Int = input_format
  private var internal_gradOutputFormat: Int = input_format
  private var internal_weightFormat: Int = weightDnnFormat

  private var internal_gradWeightFormat: Int = weightDnnFormat

  private var inputBuffer : MklDnnTensor[Float] = null
  private var gradOutputBuffer : MklDnnTensor[Float] = null
  private var weightsBuffer : MklDnnTensor[Float] = null
  private var gradWeightBuffer : MklDnnTensor[Float] = null

  private var inputBuffer_sync : Boolean = false
  private var gradOutputBuffer_sync : Boolean = false
  private var weightsBuffer_sync : Boolean = false

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
      if (nGroup == 1) {
        VariableFormat.OUT_IN_KW_KH
      } else {
        VariableFormat.GP_OUT_IN_KW_KH
      }
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


  override val isMklDnnModel: Boolean = true

  @transient
  private var inputElement : Int = 0

  // primitive for stream submit
  @transient
  private var fwd_pd: Long = 0L
  @transient
  private var conv_fwd: Long = 0L
  @transient
  private var conv_bwd: Long = 0L
  @transient
  private var conv_acc: Long = 0L

  // original memory for user input
  @transient
  private var src_memory : Long = 0L
  @transient
  private var weights_memory: Long = 0L
  @transient
  private var bias_memory: Long = 0L
  @transient
  private var gradOutput_memory: Long = 0L
  @transient
  private var gradWeights_memory: Long = 0L
  @transient
  private var gradBias_memory: Long = 0L

  // memory for user output
  @transient
  private var dst_memory: Long = 0L
  @transient
  private var gradInput_memory: Long = 0L

  // for gradInput and acc
  @transient
  private var src_md : Long = 0L
  @transient
  private var weights_md: Long = 0L
  @transient
  private var bias_md : Long = 0L
  @transient
  private var gradOutput_md: Long = 0L

  // for reorder memory primitive
  @transient
  private var reorder_gradOutput_memory: Long = 0L
  @transient
  private var reorder_src_memory: Long = 0L
  @transient
  private var reorder_weights_memory: Long = 0L
  @transient
  private var reorder_gradWeights_memory: Long = 0L

  // internal memory
  @transient
  private var internal_gradOutput_memory: Long = 0L
  @transient
  private var internal_gradWeights_memory: Long = 0L
  @transient
  private var internal_weights_memory : Long = 0L
  @transient
  private var internal_src_memory : Long = 0L

  // record whether to create primitive again
  @transient
  private var update_primitive: Boolean = true

  private var memoryPrimitives = new  ArrayBuffer[Long]
  private var buffer = new ArrayBuffer[Tensor[Float]]

  val stream_fwd = new ArrayBuffer[Long]
  val stream_bwd = new ArrayBuffer[Long]
  val stream_acc = new ArrayBuffer[Long]

  var dataTime: Long = 0L

  def reorderToUser(user_md: Long, internal_pd: Long):
  (Long, Long) = {
    if (internal_pd != 0L) {
      MklDnnOps.prepareReorder(user_md, internal_pd, false)
    } else {
      (0L, 0L)
    }
  }

  private def getOutputShape(oh: Int, ow: Int, batchSize: Int = -1): Array[Int] = {
    format match {
      case DataFormat.NCHW =>
        if (batchSize == -1) {
          Array(nOutputPlane, oh, ow)
        } else {
          Array(batchSize, nOutputPlane, oh, ow)
        }
      case DataFormat.NHWC =>
        if (batchSize == -1) {
          Array(oh, ow, nOutputPlane)
        } else {
          Array(batchSize, oh, ow, nOutputPlane)
        }

    }
  }

  def reorderTwoTensor(input: Tensor[Float], inputFormat: Int,
                       output: Tensor[Float], outputFormat: Int): Unit = {
    // val dataType = MklDnn.DataType.f32
    // val engine = MklDnn.EngineCreate( MklDnn.EngineType.cpu, 0)
    // val stream = MklDnn.StreamCreate(MklDnn.StreamType.eager)
    val stream_fwd = new ArrayBuffer[Long]
    val sizes = input.size()
    val dim = input.dim()
    output.resizeAs(input)

    val src_md = MklDnnOps.memoryDescInit(dim, sizes, dataType, inputFormat)
    val src_pd = MklDnnOps.memoryPrimitiveDescCreate(src_md, engine)

    val dst_memory = MklDnnOps.initDataMemory(dim, sizes, outputFormat, dataType, engine)
    val res = MklDnnOps.prepareReorder(dst_memory, src_pd, false)
    val src_memory = res._2

    stream_fwd.clear()
    stream_fwd.append(res._1)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(input, output)
    MklDnnOps.streamSubmit(stream, 1, stream_fwd.toArray, 1, memoryPrimitives, buffer)
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    if (inputElement != input.nElement()) {
      update_primitive = true
      inputElement = input.nElement()
    } else {
      update_primitive = false
    }
    if (update_primitive) {
      require(input.dim() == 4 && input.isContiguous())
      val (dimHeight, dimWidth, channelDim) = format.getHWCDims(input.dim())
      require(input.size(channelDim) == nInputPlane, s"input channel size " +
        s"${input.size(channelDim)} is not the same as nInputPlane $nInputPlane")


      val inputWidth = input.size(dimWidth)
      val inputHeight = input.size(dimHeight)

      val sizes =
        if (padW == -1 && padH == -1) {
          Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW,
            kernelH, kernelW)
        } else {
          Utils.getOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW,
            kernelH, kernelW, padH, padW, ceilMode = false)
        }

      val padTop = sizes(0)
      val padBottom = sizes(1)
      val padLeft = sizes(2)
      val padRight = sizes(3)
      val outputHeight = sizes(4)
      val outputWidth = sizes(5)

      val input_size = input.size()
      val dst_sizes = getOutputShape(outputHeight, outputWidth, input_size(0))
      // todo: output with Dense Tensor
      output = MklDnnTensor[Float](dst_sizes)

      src_md = MklDnnOps.memoryDescInit(input.dim(), input_size, dataType, this.internal_format)
      weights_md = MklDnnOps.memoryDescInit(dimWeight, weightSize, dataType, this.internal_format)
      bias_md = MklDnnOps.memoryDescInit(1, biasSize, dataType, MklDnn.MemoryFormat.x)
      // for output
      val dst_md = MklDnnOps.memoryDescInit(output.dim(), dst_sizes, dataType, this.internal_format)

      /* create a convolution */
      val conv_desc = MklDnnOps.convForwardDescInit(
                                MklDnn.PropKind.forwardTraining, MklDnn.AlgKind.convolutionDirect,
                                src_md, weights_md, bias_md, dst_md,
                                strides, padding, padding, MklDnn.PaddingKind.mkldnnPaddingZero)
      fwd_pd = MklDnnOps.primitiveDescCreate(conv_desc, engine, 0L)

      // create memory desc, for input
      if (input.getPrimitiveDesc() != 0L) {
        src_memory = MklDnn.PrimitiveCreate0(input.getPrimitiveDesc())
      } else {
        src_memory = MklDnnOps.initDataMemory(
                      input.dim(), input_size, this.input_format, dataType, engine)
        input.setFormat(input_format)
      }
      weights_memory = MklDnnOps.initDataMemory(
                      dimWeight, weightSize, weightDnnFormat, dataType, engine)
      bias_memory = MklDnnOps.createMemoryPrimitive(bias_md, engine)

      /* create reorder primitives between user data and convolution srcs */
      var reorder_src: Long = 0L
      var reorder_weights: Long = 0L
      internal_inputFormat = MklDnnOps.queryFormat(fwd_pd, MklDnn.Query.src_pd, 0)
      if (internal_inputFormat != input.getFormat()) {
        val res = MklDnnOps.reorderToInternal(
                  src_memory, fwd_pd, MklDnn.Query.src_pd, inputBuffer, input.size())
        reorder_src = res._1
        reorder_src_memory = res._2
      }
      internal_src_memory = if (reorder_src_memory == 0L) {
        src_memory
      } else {
        reorder_src_memory
      }

      internal_weightFormat = MklDnnOps.queryFormat(fwd_pd, MklDnn.Query.weights_pd, 0)
      if (internal_inputFormat != this.weightDnnFormat) {
        val res = MklDnnOps.reorderToInternal(
                  weights_memory, fwd_pd, MklDnn.Query.weights_pd, weightsBuffer, weight.size())
        reorder_weights = res._1
        reorder_weights_memory = res._2
      }
      internal_weights_memory = if (reorder_weights_memory == 0L) {
        weights_memory
      } else {
        reorder_weights_memory
      }

      /* create memory for dst data, we don't need to reorder it to user data */
      val dst_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.dst_pd, 0)
      dst_memory = MklDnn.PrimitiveCreate0(dst_pd)
      output.setPrimitiveDesc(dst_pd)

      val conv_inputs = Array(internal_src_memory, internal_weights_memory, bias_memory)
      val conv_outputs = Array(dst_memory)
      val indexes = Array(0, 0, 0)
      conv_fwd = MklDnnOps.primitiveCreate2(fwd_pd, conv_inputs, indexes, 3, conv_outputs, 1)

      /* build a simple net */
      stream_fwd.clear()
      if (reorder_src != 0L) {
        println(s"conv updateOutput input reorder ${this.getName()} " + internal_inputFormat)
        stream_fwd.append(reorder_src)
      }
      if (reorder_weights != 0L) {
        println("conv updateOutput weight reorder " + internal_weightFormat)
        stream_fwd.append(reorder_weights)
      }
      stream_fwd.append(conv_fwd)

      // for input
      if ((inputBuffer == null && input.getTensorType != MklDnnType)
        || (reorder_src_memory != 0L && inputBuffer == null)) {
        inputBuffer = MklDnnTensor[Float](input.size())
      } else if (inputBuffer != null && inputBuffer.nElement() != input.nElement()) {
        inputBuffer.release()
        inputBuffer = MklDnnTensor[Float](input.size())
      }

      // for weight
      if ((weightsBuffer == null && weight.getTensorType != MklDnnType)
        || (reorder_weights_memory != 0L && weightsBuffer == null)) {
          weightsBuffer = MklDnnTensor[Float](weight.size())
      } else if (weightsBuffer != null && weightsBuffer.nElement() != weight.nElement()) {
        weightsBuffer.release()
        weightsBuffer = MklDnnTensor[Float](weight.size())
      }
    }
    val n_fwd = stream_fwd.length
    memoryPrimitives.clear()
    buffer.clear()
    memoryPrimitives.append(bias_memory, dst_memory)
    buffer.append(bias, output)

    if (reorder_src_memory == 0L) {
      // sync here
      if (input.getTensorType != MklDnnType) {
        MklDnnTensor.syncFromHeap(inputBuffer, input.storage().array(), input.storageOffset() - 1)
      } else {
        inputBuffer = input.asInstanceOf[MklDnnTensor[Float]]
      }
      inputBuffer_sync = true
      memoryPrimitives.append(src_memory)
      buffer.append(inputBuffer)
    } else {
      memoryPrimitives.append(src_memory, reorder_src_memory)
      buffer.append(input, inputBuffer)
    }

    if (reorder_weights_memory == 0L) {
      // sync here
      if (weight.getTensorType != MklDnnType) {
        MklDnnTensor.syncFromHeap(weightsBuffer, weight.storage().array(), weight.storageOffset() - 1)
      } else {
        weightsBuffer = weight.asInstanceOf[MklDnnTensor[Float]]
      }
      weightsBuffer_sync = true
      memoryPrimitives.append(weights_memory)
      buffer.append(weightsBuffer)
    } else {
      memoryPrimitives.append(weights_memory, reorder_weights_memory)
      buffer.append(weight, weightsBuffer)
    }

    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd.toArray, n_fwd,
      memoryPrimitives.toArray, buffer.toArray)

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugFwInfo(this.getName(), end1, input.getFormat(), output.getFormat())
    }
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    if (!propagateBack) {
      return gradInput
    }
    if (update_primitive) {
      gradOutput_md = MklDnnOps.memoryDescInit(gradOutput.dim(), gradOutput.size(), dataType,
        this.internal_format)
      // for gradInput
      gradInput = MklDnnTensor[Float](input.size())
      val gradInput_md = MklDnnOps.memoryDescInit(gradInput.dim(), gradInput.size(), dataType,
        this.internal_format)

      val bwd_data_desc = MklDnnOps.convBackwardDataDescInit(
                                      MklDnn.AlgKind.convolutionDirect, gradInput_md, weights_md,
                                      gradOutput_md, strides, padding, padding,
                                      MklDnn.PaddingKind.mkldnnPaddingZero)
      val bwd_data_pd = MklDnnOps.primitiveDescCreate(bwd_data_desc, engine, fwd_pd)

      /* create memory primities for gradInput */
      val gradInput_pd = MklDnnOps.primitiveDescQueryPd(bwd_data_pd, MklDnn.Query.diff_src_pd, 0)
      gradInput_memory = MklDnn.PrimitiveCreate0(gradInput_pd)
      gradInput.setPrimitiveDesc(gradInput_pd)

      if (gradOutput.getPrimitiveDesc() != 0L) {
        gradOutput_memory = MklDnn.PrimitiveCreate0(gradOutput.getPrimitiveDesc())
      } else {
        gradOutput_memory = MklDnnOps.initDataMemory(
                       gradOutput.dim(), gradOutput.size(), this.input_format, dataType, engine)
        gradOutput.setFormat(input_format)
      }
      /* create reorder primitives between user gradOutput and convolution gradOutput */
      var reorder_gradOutput: Long = 0L
      internal_gradOutputFormat = MklDnnOps.queryFormat(bwd_data_pd, MklDnn.Query.diff_dst_pd, 0)
      if (internal_gradOutputFormat != gradOutput.getFormat()) {
        val res = MklDnnOps.reorderToInternal(gradOutput_memory,
                      bwd_data_pd, MklDnn.Query.diff_dst_pd, gradOutputBuffer, gradOutput.size())
        reorder_gradOutput = res._1
        reorder_gradOutput_memory = res._2
      }
      internal_gradOutput_memory = if (reorder_gradOutput_memory == 0L) {
        gradOutput_memory
      } else {
        reorder_gradOutput_memory
      }

      var reorder_weights: Long = 0L
      val f1 = MklDnnOps.queryFormat(bwd_data_pd, MklDnn.Query.weights_pd, 0)
      if (f1 == weightDnnFormat) {
        reorder_weights_memory = 0L
      } else if (f1 != internal_weightFormat) {
        internal_weightFormat = f1
        val res = MklDnnOps.reorderToInternal(weights_memory, bwd_data_pd, MklDnn.Query.weights_pd,
          weightsBuffer, weight.size())
        reorder_weights = res._1
        reorder_weights_memory = res._2
      }
      internal_weights_memory = if (reorder_weights_memory == 0L) {
        weights_memory
      } else {
        reorder_weights_memory
      }

      val conv_inputs = Array(internal_gradOutput_memory, internal_weights_memory,
        internal_src_memory)
      val conv_outputs = Array(gradInput_memory)
      val indexes = Array(0, 0, 0)

      /* finally create a convolution primitive */
      conv_bwd = MklDnnOps.primitiveCreate2(bwd_data_pd, conv_inputs, indexes, 3, conv_outputs, 1)
      /* build a simple net */
      stream_bwd.clear()
      if (reorder_gradOutput != 0L) {
        println("conv updateGradInput gradOutput reorder " + internal_gradOutputFormat)
        stream_bwd.append(reorder_gradOutput)
      }
      if (reorder_weights != 0L) {
        println("conv updateGradInput weight reorder " + internal_weightFormat)
        stream_bwd.append(reorder_weights)
      }
      stream_bwd.append(conv_bwd)

      // for gradOutput
      if ((gradOutputBuffer == null && gradOutput.getTensorType != MklDnnType)
        || (reorder_gradOutput_memory != 0L && gradOutputBuffer == null)) {
        gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
      } else if (gradOutputBuffer != null && gradOutputBuffer.nElement() != gradOutput.nElement()) {
        gradOutputBuffer.release()
        gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
      }
    }

    val n_bwd = stream_bwd.length
    memoryPrimitives.clear()
    buffer.clear()

    if (reorder_gradOutput_memory == 0L) {
      // sync here
      if (gradOutput.getTensorType != MklDnnType) {
        MklDnnTensor.syncFromHeap(
          gradOutputBuffer, gradOutput.storage().array(), gradOutput.storageOffset() - 1)
      } else {
        gradOutputBuffer = gradOutput.asInstanceOf[MklDnnTensor[Float]]
      }
      gradOutputBuffer_sync = true
      memoryPrimitives.append(gradOutput_memory, gradInput_memory)
      buffer.append(gradOutputBuffer, gradInput)
    } else {
      memoryPrimitives.append(reorder_gradOutput_memory, gradInput_memory, gradOutput_memory)
      buffer.append(gradOutputBuffer, gradInput, gradOutput)
    }
    if (reorder_weights_memory == 0L) {
      // sync here
      if (!weightsBuffer_sync) {
        if (weight.getTensorType != MklDnnType) {
          MklDnnTensor.syncFromHeap(weightsBuffer, weight.storage().array(), weight.storageOffset() - 1)
        } else {
          weightsBuffer = weight.asInstanceOf[MklDnnTensor[Float]]
        }
      }
      memoryPrimitives.append(weights_memory)
      buffer.append(weightsBuffer)
    } else {
      memoryPrimitives.append(weights_memory, reorder_weights_memory)
      buffer.append(weight, weightsBuffer)
    }

    MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd,
      memoryPrimitives.toArray, buffer.toArray)

    val end1 = System.nanoTime() - s1
    dataTime = end1

    gradInput
  }

  override def accGradParameters(input: Tensor[Float], gradOutput: Tensor[Float]): Unit = {
    val s1 = System.nanoTime()
    if (update_primitive) {
      if (!propagateBack) {
        gradOutput_md = MklDnnOps.memoryDescInit(gradOutput.dim(), gradOutput.size(), dataType,
          this.internal_format)
        if (gradOutput.getPrimitiveDesc() != 0L) {
          gradOutput_memory = MklDnn.PrimitiveCreate0(gradOutput.getPrimitiveDesc())
        } else {
          gradOutput_memory = MklDnnOps.initDataMemory(
            gradOutput.dim(), gradOutput.size(), this.input_format, dataType, engine)
          gradOutput.setFormat(input_format)
        }
      }
      // for gradWeights
      val gradWeights_md = MklDnnOps.memoryDescInit(dimWeight, weightSize, dataType, this.internal_format)
      gradWeights_memory = MklDnnOps.initDataMemory(dimWeight, weightSize, weightDnnFormat,
        dataType, engine)

     // for gradBias
     val gradBias_md = MklDnnOps.memoryDescInit(1, biasSize, dataType, MklDnn.MemoryFormat.x)
     gradBias_memory = MklDnnOps.createMemoryPrimitive(gradBias_md, engine)

     val weights_desc = MklDnnOps.convBackwardWeightsDescInit(
                                   MklDnn.AlgKind.convolutionDirect, src_md, gradWeights_md,
                                   gradBias_md, gradOutput_md, strides, padding,
                                   padding, MklDnn.PaddingKind.mkldnnPaddingZero)
     val weights_pd = MklDnnOps.primitiveDescCreate(weights_desc, engine, fwd_pd)

      // reorder gradWeight
      var reorder_gradWeights: Long = 0L
      var reorder_gradWeights_pd: Long = 0L
      internal_gradWeightFormat = MklDnnOps.queryFormat(weights_pd, MklDnn.Query.diff_weights_pd, 0)
      if ( internal_gradWeightFormat != this.weightDnnFormat) {
        val res = MklDnnOps.reorderToInternal(gradWeights_memory, weights_pd,
          MklDnn.Query.diff_weights_pd, gradWeightBuffer, gradWeight.size())
        reorder_gradWeights = res._1
        reorder_gradWeights_memory = res._2
        reorder_gradWeights_pd = res._3
      }
      internal_gradWeights_memory = if (reorder_gradWeights_memory == 0L) {
        gradWeights_memory
      } else {
        reorder_gradWeights_memory
      }

      var reorder_gradOutput: Long = 0L
      val f1 = MklDnnOps.queryFormat(weights_pd, MklDnn.Query.diff_dst_pd, 0)
      if (f1 == gradOutput.getFormat()) {
        reorder_gradOutput_memory = 0L
      } else if ( f1 != internal_gradOutputFormat) {
        internal_gradOutputFormat = f1
        val res = MklDnnOps.reorderToInternal(gradOutput_memory, weights_pd,
          MklDnn.Query.diff_dst_pd, gradOutputBuffer, gradOutput.size())
        reorder_gradOutput = res._1
        reorder_gradOutput_memory = res._2
      }
      internal_gradOutput_memory = if (reorder_gradOutput_memory == 0L) {
        gradOutput_memory
      } else {
        reorder_gradOutput_memory
      }

      var reorder_src: Long = 0L
      val f2 = MklDnnOps.queryFormat(weights_pd, MklDnn.Query.src_pd, 0)
      if (f2 == input.getFormat()) {
        reorder_src_memory = 0L
      } else if ( f2 != internal_inputFormat) {
        internal_inputFormat = f2
        val res = MklDnnOps.reorderToInternal(src_memory, weights_pd, MklDnn.Query.src_pd,
          inputBuffer, input.size())
        reorder_src = res._1
        reorder_src_memory = res._2
      }
      internal_src_memory = if (reorder_src_memory == 0L) {
        src_memory
      } else {
        reorder_src_memory
      }

      val conv_inputs = Array(internal_src_memory, internal_gradOutput_memory)
      val conv_outputs = Array(internal_gradWeights_memory, gradBias_memory)
      val indexes = Array(0, 0)

      /* finally create a convolution primitive */
      conv_acc = MklDnnOps.primitiveCreate2(weights_pd, conv_inputs, indexes, 2, conv_outputs, 2)

      stream_acc.clear()
      if (reorder_gradWeights != 0L) {
        println("conv accGradParameters gradWeights reorder " + internal_gradWeightFormat)
        stream_acc.append(reorder_gradWeights)
      }
      if (reorder_gradOutput != 0L) {
        println("conv accGradParameters gradOutput reorder " + internal_gradOutputFormat)
        stream_acc.append(reorder_gradOutput)
      }
      if (reorder_src != 0L) {
        println(s"conv accGradParameters input reorder ${this.getName()} " + internal_inputFormat)
        stream_acc.append(reorder_src)
      }
      stream_acc.append(conv_acc)

      // for gradWeight
      if ((gradWeightBuffer == null && gradWeight.getTensorType != MklDnnType)
        || (reorder_gradWeights_memory != 0L && gradWeightBuffer == null)) {
        gradWeightBuffer = MklDnnTensor[Float](gradWeight.size())
      } else if (gradWeightBuffer != null && gradWeightBuffer.nElement() != gradWeight.nElement()) {
        gradWeightBuffer.release()
        gradWeightBuffer = MklDnnTensor[Float](gradWeight.size())
      }

      // for gradOutput
      if ((gradOutputBuffer == null && gradOutput.getTensorType != MklDnnType)
        || (reorder_gradOutput_memory != 0L && gradOutputBuffer == null)) {
        gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
      } else if (gradOutputBuffer != null && gradOutputBuffer.nElement() != gradOutput.nElement()) {
        gradOutputBuffer.release()
        gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
      }
    }
    /* build a simple net */
    // keep original data
    original_gradWeights.resizeAs(gradWeight).copy(gradWeight)
    original_gradBias.resizeAs(gradBias).copy(gradBias)

    val n_bwd = stream_acc.length
    memoryPrimitives.clear()
    buffer.clear()

    if (reorder_gradWeights_memory == 0L) {
      // sync here
      if (gradWeight.getTensorType != MklDnnType) {
        MklDnnTensor.syncFromHeap(
          gradWeightBuffer, gradWeight.storage().array(), gradWeight.storageOffset() - 1)
      } else {
        gradWeightBuffer = gradWeight.asInstanceOf[MklDnnTensor[Float]]
      }
      memoryPrimitives.append(gradBias_memory, gradWeights_memory)
      buffer.append(gradBias, gradWeightBuffer)
    } else {
      memoryPrimitives.append(gradWeights_memory, gradBias_memory, reorder_gradWeights_memory)
      buffer.append(gradWeight, gradBias, gradWeightBuffer)
    }

    if (reorder_src_memory == 0L) {
      // sync here
      if (!inputBuffer_sync) {
        if (input.getTensorType != MklDnnType) {
          MklDnnTensor.syncFromHeap(inputBuffer, input.storage().array(), input.storageOffset() - 1)
        } else {
          inputBuffer = input.asInstanceOf[MklDnnTensor[Float]]
        }
      }
      memoryPrimitives.append(src_memory)
      buffer.append(inputBuffer)
    } else {
      memoryPrimitives.append(src_memory, reorder_src_memory)
      buffer.append(input, inputBuffer)
    }

    if (reorder_gradOutput_memory == 0L) {
      // sync here
      if (!gradOutputBuffer_sync) {
        if (gradOutput.getTensorType != MklDnnType) {
          MklDnnTensor.syncFromHeap(gradOutputBuffer, gradOutput.storage().array(), gradOutput.storageOffset() - 1)
        } else {
          gradOutputBuffer = gradOutput.asInstanceOf[MklDnnTensor[Float]]
        }
      }
      memoryPrimitives.append(gradOutput_memory)
      buffer.append(gradOutputBuffer)
    } else {
      memoryPrimitives.append(gradOutput_memory, reorder_gradOutput_memory)
      buffer.append(gradOutput, gradOutputBuffer)
    }

    MklDnnOps.streamSubmit(stream, n_bwd, stream_acc.toArray, n_bwd, memoryPrimitives.toArray, buffer.toArray)

    // sync from native to heap
    if (internal_gradWeightFormat != weightDnnFormat) {
      reorderTwoTensor(gradWeightBuffer, internal_gradWeightFormat, gradWeight, weightDnnFormat)
    } else if (gradWeight.getTensorType != MklDnnType) {
      MklDnnTensor.syncToHeap(gradWeightBuffer, gradWeight.storage().array(), gradWeight.storageOffset() - 1)
    }

    val end1 = (System.nanoTime() - s1 + dataTime)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, gradOutput.getFormat(), gradInput.getFormat())
    }

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

  override def clearState() : this.type = {
    super.clearState()
    gradOutputBuffer.set()
    inputBuffer.set()
    weightsBuffer.set()
    gradWeightBuffer.set()
    original_gradWeights.set()
    original_gradBias.set()
    this
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
   propagateBack: Boolean = true,
   initWeight: Tensor[Float] = null,
   initBias: Tensor[Float] = null,
   initGradWeight: Tensor[Float] = null,
   initGradBias: Tensor[Float] = null,
   withBias: Boolean = true,
   format: DataFormat = DataFormat.NCHW): ConvolutionDnn[Float] = {
    new ConvolutionDnn[Float](nInputPlane, nOutputPlane, kW, kH, dW,
    dH, padW, padH, nGroup, propagateBack, initWeight, initBias,
    initGradWeight, initGradBias, withBias, format)
  }
}
