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

import com.intel.analytics.bigdl.mkl.DataType
import com.intel.analytics.bigdl.nn.DynamicContainer
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{DenseType, DnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable.ArrayBuffer

/**
 * Helper utilities when integrating Module with MKL-DNN
 */
trait MklDnnModule extends MklDnnModuleHelper {
  /**
   * MklDnn runtime, which includes a MKL-DNN engine and a MKL-DNN stream.
   * Note that this instance will be erased when send to remote worker, so you
   * should recreate a MklDnnRuntime.
   */
  @transient protected var runtime: MklDnnRuntime = _

  def setRuntime(runtime: MklDnnRuntime): Unit = {
    this.runtime = runtime
  }

  private[bigdl] def getRuntime: MklDnnRuntime = {
    require(runtime != null, s"you should init the mkldnn runtime first")
    runtime
  }

  /**
   * Init the MKL-DNN primitives for the layer. Note that these primitives will be erased when
   * sent to a remote worker.
   */
  private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase)
  : (Array[MemoryData], Array[MemoryData])

  private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase)
  : (Array[MemoryData], Array[MemoryData])

  private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData], phase: Phase): Array[MemoryData]
  = grad

  private[bigdl] def initFwdPrimitives(inputs: Array[MemoryData])
  : (Array[MemoryData], Array[MemoryData]) = initFwdPrimitives(inputs, null)

  private[bigdl] def initBwdPrimitives(grad: Array[MemoryData])
  : (Array[MemoryData], Array[MemoryData]) = initBwdPrimitives(grad, null)

  private[bigdl] def initGradWPrimitives(grad: Array[MemoryData])
  : Array[MemoryData] = initGradWPrimitives(grad, null)

  private[mkldnn] def inputFormats(): Array[MemoryData]

  private[mkldnn] def gradInputFormats(): Array[MemoryData]

  private[mkldnn] def outputFormats(): Array[MemoryData]

  private[mkldnn] def gradOutputFormats(): Array[MemoryData]

  private[mkldnn] def gradOutputWeightFormats(): Array[MemoryData]

  def setQuantize(value: Boolean): this.type
}

trait MklDnnModuleHelper extends MemoryOwner {

  @transient protected implicit lazy val _this : MemoryOwner = this

  protected def initActivity(formats: Array[MemoryData]): Activity = {
    if (formats.length == 1) {
      initTensor(formats(0))
    } else {
      T.array(formats.map(initTensor(_)))
    }
  }

  protected def initTensor(format: MemoryData): Tensor[_] = {
    val paddingShape = format.getPaddingShape
    val realSize = format.getRealSize

    format match {
      case d: NativeData =>
        d.dataType match {
          case DataType.S8 => DnnTensor[Byte](paddingShape, realSize)
          case DataType.U8 => DnnTensor[Byte](paddingShape, realSize)
          case DataType.S32 => DnnTensor[Int](paddingShape, realSize)
          case DataType.F32 => DnnTensor[Float](paddingShape, realSize)
        }
      case d: HeapData =>
        Tensor[Float](paddingShape)
      case _ => throw new UnsupportedOperationException("memory format is not supported")
    }
  }

  protected def singleNativeData(formats: Array[MemoryData]): Array[MemoryData] = {
    require(formats.length == 1, "Only accept one tensor as input")
    nativeData(formats)
  }

  protected def nativeData(formats: Array[MemoryData]): Array[MemoryData] = {
    formats.map(
      f => {
        f match {
          case i: NativeData => i
          case i: HeapData => i.toNative()
          case _ => throw new UnsupportedOperationException("Not support memory format")
        }
      }
    )
  }
}

trait MklDnnLayer extends AbstractModule[Activity, Activity, Float] with MklDnnModule {
  /**
   * MKL-DNN primitives of the module. Note you should only initialize this field by calling
   * initPrimitives method. This field will be erased when sending model to remote worker. So you
   * need to reinitialize it after sending the model.
   */
  @transient
  protected var updateOutputPrimitives: Array[Long] = _
  @transient
  protected var updateGradInputPrimitives: Array[Long] = _
  @transient
  protected var accGradientPrimitives: Array[Long] = _

  protected var _inputFormats: Array[MemoryData] = _
  protected var _gradInputFormats: Array[MemoryData] = _
  protected var _outputFormats: Array[MemoryData] = _
  protected var _gradOutputFormats: Array[MemoryData] = _
  protected var _gradOutputFormatsForWeight: Array[MemoryData] = _

  @transient
  private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient
  private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient
  private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient
  private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient
  private var cachedInput: Activity = _
  @transient
  private var cachedGradOutput: Activity = _

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
    phase: Phase): Array[MemoryData] = {
    _gradOutputFormatsForWeight = grad
    grad
  }

  def getUpdateOutputMemoryPrimitives(): Array[Long] = {
    inputFormats().map(_.getPrimitive(runtime)) ++
      outputFormats().map(_.getPrimitive(runtime))
  }
  def getUpdateGradInputMemoryPrimitives(): Array[Long] = {
    gradOutputFormats().map(_.getPrimitive(runtime)) ++
      gradInputFormats().map(_.getPrimitive(runtime))
  }
  override def updateOutput(input: Activity): Activity = {
    if (updateOutputMemoryPrimitives == null) {
      updateOutputMemoryPrimitives = getUpdateOutputMemoryPrimitives()
    }
    if (updateOutputTensors == null || cachedInput == null || !cachedInput.eq(input)) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      if (input.isTensor) {
        buffer.append(input.asInstanceOf[Tensor[Float]])
      } else {
        val table = input.toTable
        var i = 1
        while (i <= table.length()) {
          buffer.append(table(i))
          i += 1
        }
      }
      if (output.isTensor) {
        buffer.append(output.asInstanceOf[Tensor[Float]])
      } else {
        val table = output.toTable
        var i = 1
        while (i <= table.length()) {
          buffer.append(table(i))
          i += 1
        }
      }
      updateOutputTensors = buffer.toArray
      cachedInput = input
    }
    MklDnnOps.streamSubmit(
      runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives,
      updateOutputTensors
    )
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputMemoryPrimitives == null) {
      updateGradInputMemoryPrimitives = getUpdateGradInputMemoryPrimitives()
    }
    if (updateGradInputTensors == null || cachedInput == null || !cachedInput.eq(input) ||
      cachedGradOutput == null || !cachedGradOutput.eq(gradOutput)) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      if (gradOutput.isTensor) {
        buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      } else {
        val table = gradOutput.toTable
        var i = 1
        while (i <= table.length()) {
          buffer.append(table(i))
          i += 1
        }
      }
      if (gradInput.isTensor) {
        buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      } else {
        val table = gradInput.toTable
        var i = 1
        while (i <= table.length()) {
          buffer.append(table(i))
          i += 1
        }
      }
      updateGradInputTensors = buffer.toArray
      cachedInput = input
      cachedGradOutput = gradOutput
    }
    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length,
      updateGradInputMemoryPrimitives, updateGradInputTensors)
    gradInput
  }


  override private[bigdl] def inputFormats() = {
    require(_inputFormats != null, "You should call initFwdPrimitives first")
    _inputFormats
  }

  override private[bigdl] def gradInputFormats() = {
    require(_gradInputFormats != null, "You should call initBwdPrimitives first")
    _gradInputFormats
  }

  override private[bigdl] def outputFormats() = {
    require(_outputFormats != null, "You should call initFwdPrimitives first")
    _outputFormats
  }

  override private[bigdl] def gradOutputFormats() = {
    require(_gradOutputFormats != null, "You should call initBwdPrimitives first")
    _gradOutputFormats
  }

  override private[mkldnn] def gradOutputWeightFormats() = {
    _gradOutputFormatsForWeight
  }

  def updateWithNewTensor(from: Array[Tensor[Float]], index: Int,
    value: Activity): Unit = {
    from(index).getTensorType match {
      case DenseType => from(index) = value.toTensor[Float]
      case _ =>
    }
  }

  override def release(): Unit = {
    this.releaseResources()
  }

  override def setQuantize(value: Boolean): MklDnnLayer.this.type = this

  def paramsMMap(): (Array[TensorMMap], Array[TensorMMap]) = {
    // return null for weight and gradWeight by default
    (Array.empty[TensorMMap], Array.empty[TensorMMap])
  }
}

/**
 * Helper utilities when integrating containers with MKL-DNN
 */
trait MklDnnContainer extends DynamicContainer[Activity, Activity, Float] with MklDnnModule {
  @transient protected lazy val reorderManager = new ReorderManager()
  protected var mklDnnModules : Array[MklDnnModule] = _

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, Float]): this.type = {
    require(mklDnnModules == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  private def checkInputs: Boolean = {
    def getAllInputs(
      module: AbstractModule[_ <: Activity, _ <: Activity, Float]): Boolean = {
      module match {
        case seq: Sequential => getAllInputs(seq.modules.head)
        case concat: ConcatTable => concat.modules.map(x => getAllInputs(x)).reduce(_ && _)
        case _: Input => true
        case _ => false
      }
    }

    getAllInputs(this)
  }

  final def compile(phase: Phase): Unit = {
    require(checkInputs, s"You should add Input for the container.")
    compile(phase, new MklDnnRuntime, Array[MemoryData]())
  }

  /**
   * Create MklDnnRuntime and compile the model
   * @param phase
   */
  private[mkldnn] final def compile(phase: Phase, formats: Array[MemoryData]): Unit = {
    compile(phase, new MklDnnRuntime(), formats)
  }

  /**
   * Compile the model, which includes infer memory shapes, allocate memory, optimize computing
   * path and create MKL-DNN primitives
   * @param phase
   * @param runtime
   */
  private[mkldnn] final def compile(phase: Phase, runtime: MklDnnRuntime,
    formats: Array[MemoryData]): Unit = {
    freeze()
    fusion(phase)
    initPrimitives(phase, runtime, formats)
  }

  final def initPrimitives(phase: Phase, runtime: MklDnnRuntime, formats: Array[MemoryData])
  : Unit = {
    setRuntime(runtime)
    val outputFormats = initFwdPrimitives(formats, phase)._2
    if (phase == Phase.TrainingPhase) {
      initBwdPrimitives(outputFormats, phase)
      initGradWPrimitives(outputFormats, phase)
    }
  }

  override def setRuntime(runtime: MklDnnRuntime): Unit = {
    super.setRuntime(runtime)
    reorderManager.setRuntime(runtime)
    modules.foreach { case m: MklDnnModule => m.setRuntime(runtime) }
  }

  /**
   * Modify the computing path by fuse some layers into one to improve the performance
   */
  private[mkldnn] def fusion(phase: Phase): Unit = {
    modules.filter(_.isInstanceOf[MklDnnContainer])
      .map { case mc: MklDnnContainer => mc.fusion(phase) }
  }

  private def freeze(): Unit = {
    if (mklDnnModules == null) {
      mklDnnModules = modules.map(_.asInstanceOf[MklDnnModule]).toArray
    }
    modules.filter(_.isInstanceOf[MklDnnContainer])
      .map { case mc: MklDnnContainer => mc.freeze() }
  }

  override def release(): Unit = {
    super.release()
    this.releaseResources()
  }

  override def setQuantize(value: Boolean): this.type = {
    this.modules.foreach {
      case mkldnnModule: MklDnnModule => mkldnnModule.setQuantize(value)
      case _ =>
    }
    this
  }
}
