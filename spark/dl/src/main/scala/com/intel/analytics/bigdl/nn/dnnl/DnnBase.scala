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

import com.intel.analytics.bigdl.dnnl.{DataType, ArgType}
import com.intel.analytics.bigdl.nn.DynamicContainer
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{DenseType, DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable

/**
 * Helper utilities when integrating Module with MKL-DNN
 */
trait MklDnnModule extends MemoryOwner {

  protected var _inputFormats: Array[MemoryData] = _
  protected var _gradInputFormats: Array[MemoryData] = _
  protected var _outputFormats: Array[MemoryData] = _
  protected var _gradOutputFormats: Array[MemoryData] = _
  protected var _gradOutputFormatsForWeight: Array[MemoryData] = _

  @transient
  protected var updateOutputTensors: mutable.Map[Int, Tensor[Float]] = _
  @transient
  protected var updateGradInputTensors: mutable.Map[Int, Tensor[Float]] = _
  @transient
  protected var updateGradInputPrimitives: Array[Long] = _

  @transient
  protected var fwdExecArgs: mutable.Map[Int, Long] = _
  @transient
  protected var bwdExecArgs: mutable.Map[Int, Long] = _
  @transient
  protected var weightUpdateExecArgs: mutable.Map[Int, Long] = _

  @transient protected implicit lazy val _this : MemoryOwner = this

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
  private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase = null)
  : (Array[MemoryData], Array[MemoryData])

  private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase = null)
  : (Array[MemoryData], Array[MemoryData])

  private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData], phase: Phase = null)
  : Array[MemoryData]
  = grad

  private[mkldnn] def inputFormats(): Array[MemoryData]

  private[mkldnn] def gradInputFormats(): Array[MemoryData]

  private[mkldnn] def outputFormats(): Array[MemoryData]

  private[mkldnn] def gradOutputFormats(): Array[MemoryData]

  private[mkldnn] def gradOutputWeightFormats(): Array[MemoryData]

  def setQuantize(value: Boolean): this.type

  // Merge from helper
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
  protected var accGradientPrimitives: Array[Long] = _
  @transient
  private var cachedInput: Activity = _
  @transient
  private var cachedGradOutput: Activity = _


  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
                                                 phase: Phase): Array[MemoryData] = {
    _gradOutputFormatsForWeight = grad
    grad
  }


  // TODO: rename
  def initFwdExecArgs(): Unit = {
    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC ->
        inputFormats().map(_.getMemoryObject(runtime)).head,
      ArgType.DNNL_ARG_DST ->
        outputFormats().map(_.getMemoryObject(runtime)).head
    )
  }

  // TODO: rename
  def initBwdExecArgs(): Unit = {
    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_DIFF_SRC ->
        gradInputFormats().map(_.getMemoryObject(runtime)).head,
      ArgType.DNNL_ARG_DIFF_DST ->
        gradOutputFormats().map(_.getMemoryObject(runtime)).head
    )
  }

  // TODO:
  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null || cachedInput == null || !cachedInput.eq(input)) {
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
      )
      cachedInput = input
    }
    MklDnnOps.streamSubmit(updateOutputPrimitives,
      runtime.stream, fwdExecArgs,
      updateOutputTensors
    )
    output
  }

  // TODO:
  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null || cachedInput == null || !cachedInput.eq(input) ||
      cachedGradOutput == null || !cachedGradOutput.eq(gradOutput)) {
      updateGradInputTensors = mutable.Map(
        ArgType.DNNL_ARG_DIFF_DST -> gradOutput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DIFF_SRC -> gradInput.asInstanceOf[Tensor[Float]]
      )
      cachedInput = input
      cachedGradOutput = gradOutput
    }

    // TODO:
    MklDnnOps.streamSubmit(updateGradInputPrimitives,
      runtime.stream, bwdExecArgs,
      updateGradInputTensors
    )
    gradInput
  }

  override private[mkldnn] def inputFormats() = {
    require(_inputFormats != null, "You should call initFwdPrimitives first")
    _inputFormats
  }

  override private[mkldnn] def gradInputFormats() = {
    require(_gradInputFormats != null, "You should call initBwdPrimitives first")
    _gradInputFormats
  }

  override private[bigdl] def outputFormats() = {
    require(_outputFormats != null, "You should call initFwdPrimitives first")
    _outputFormats
  }

  override private[mkldnn] def gradOutputFormats() = {
    require(_gradOutputFormats != null, "You should call initBwdPrimitives first")
    _gradOutputFormats
  }

  override private[mkldnn] def gradOutputWeightFormats() = {
    _gradOutputFormatsForWeight
  }

  def updateWithNewTensor(from: mutable.Map[Int, Tensor[Float]], index: Int,
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
  // protected var subModuleFwdPrimitives: Array[Array[Long]] = _
  protected var subModuleBwdPrimitives: Array[Array[Long]] = _

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
    initFwdPrimitives(formats, phase)
    val outFormats = outputFormats()
    if (phase == Phase.TrainingPhase) {
      initBwdPrimitives(outFormats, phase)
      initGradWPrimitives(outFormats, phase)
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
