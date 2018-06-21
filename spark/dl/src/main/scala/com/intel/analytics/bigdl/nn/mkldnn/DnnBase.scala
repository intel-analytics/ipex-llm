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

import com.intel.analytics.bigdl.mkl.{DataType, MklDnn}
import com.intel.analytics.bigdl.nn.DynamicContainer
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable.ArrayBuffer

/**
 * Helper utilities when integrating Module with MKL-DNN
 */
trait MklDnnModule {
  /**
   * MklDnn runtime, which includes a MKL-DNN engine and a MKL-DNN stream.
   * Note that this instance will be erased when send to remote worker, so you
   * should recreate a MklDnnRuntime.
   */
  @transient
  protected var runtime : MklDnnRuntime = _

  /**
   * Compute the output shape based on the input shape
   */
  private[mkldnn] def inferShape(shapes: Array[Array[Int]]): Array[Array[Int]]

  /**
   * Init the MKL-DNN primitives for the layer. Note that these primitives will be erased when
   * sent to a remote worker.
   * @param runtime
   */
  private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit
  private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit
  private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit = {}

  /**
   * Allocate memory. Note that these primitives will be erased when sent to a remote worker.
   */
  private[mkldnn] def initMemory(): Unit

  /**
   * Memory formats
   * @return
   */
  private[mkldnn] def inputFormats(): Array[MemoryData]
  private[mkldnn] def gradInputFormats(): Array[MemoryData]
  private[mkldnn] def outputFormats(): Array[MemoryData]
  private[mkldnn] def gradOutputFormats(): (Array[MemoryData], Array[MemoryData])
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
  @transient
  protected var fwdMemPrims: Array[Long] = _
  @transient
  private var cachedInput: Activity = _
  @transient
  private var fwdTensors: Array[Tensor[Float]] = _
  @transient
  protected var bwdMemPrims: Array[Long] = _
  @transient
  private var cachedGradOutput: Activity = _
  @transient
  private var bwdTensors: Array[Tensor[Float]] = _
  @transient
  protected var gradOutputPrimitivesWeight: Array[Long] = _

  protected var _inputFormats: Array[MemoryData] = _
  protected var _gradInputFormats: Array[MemoryData] = _
  protected var _outputFormats: Array[MemoryData] = _
  protected var _gradOutputFormats: Array[MemoryData] = _
  protected var _gradOutputFormatsForWeight: Array[MemoryData] = _

  protected def initMemPrimFromFormat(formats: Array[MemoryData]): Array[Long] = {
    formats.map(format => {
      val memDesc = MklDnn.MemoryDescInit(format.shape.length, format.shape,
        DataType.F32, format.layout)
      val primDesc = MklDnn.MemoryPrimitiveDescCreate(memDesc, runtime.engine)
      MklDnn.PrimitiveCreate0(primDesc)
    })
  }

  protected def initMemPrimFromPrimDesc(primDescs: Array[Long]): Array[Long] = {
    primDescs.map(primDesc => {
      MklDnn.PrimitiveCreate0(primDesc)
    })
  }

  protected def initMemPrimDescFromFormat(formats: Array[MemoryData]): Array[Long] = {
    formats.map(format => {
      val memDesc = MklDnn.MemoryDescInit(format.shape.length, format.shape,
        DataType.F32, format.layout)
      MklDnn.MemoryPrimitiveDescCreate(memDesc, runtime.engine)
    })
  }

  protected def initMemDescFromFormat(formats: Array[MemoryData]): Array[Long] = {
    formats.map(format => {
      MklDnn.MemoryDescInit(format.shape.length, format.shape,
        DataType.F32, format.layout)
    })
  }

  protected def initActivity(formats: Array[MemoryData]): Activity = {
    if (formats.length == 1) {
      initTensor(formats(0))
    } else {
      T.array(formats.map(initTensor(_)))
    }
  }

  protected def initTensor(format: MemoryData): Tensor[Float] = {
    format match {
      case d: NativeData =>
        DnnTensor[Float](d.shape)
      case d: HeapData =>
        Tensor[Float](d.shape)
      case _ => throw new UnsupportedOperationException("memory format is not supported")
    }
  }

  override private[mkldnn] def inputFormats() = {
    require(_inputFormats != null, "You must call infershape first")
    _inputFormats
  }

  override private[mkldnn] def gradInputFormats() = {
    require(_gradInputFormats != null, "You must call infershape first")
    _gradInputFormats
  }

  override private[mkldnn] def outputFormats() = {
    require(_outputFormats != null, "You must call infershape first")
    _outputFormats
  }

  override private[mkldnn] def gradOutputFormats() = {
    require(_gradOutputFormats != null, "You must call infershape first")
    require(_gradOutputFormatsForWeight != null, "You must call infershape first")
    (_gradOutputFormats, _gradOutputFormatsForWeight)
  }

  override def updateOutput(input: Activity): Activity = {
    if (fwdTensors == null || cachedInput == null || !cachedInput.eq(input)) {
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
      fwdTensors = buffer.toArray
      cachedInput = input
    }
    MklDnnOps.streamSubmit(
      runtime.stream, 1, updateOutputPrimitives, 1, fwdMemPrims, fwdTensors
    )
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (bwdTensors == null || cachedInput == null || !cachedInput.eq(input) ||
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
      bwdTensors = buffer.toArray
      cachedInput = input
      cachedGradOutput = gradOutput
    }
    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives, 1, bwdMemPrims, bwdTensors)
    gradInput
  }

  override def initMemory(): Unit = {
    gradInput = initActivity(gradInputFormats())
    output = initActivity(outputFormats())
  }

}

/**
 * Helper utilities when integrating containers with MKL-DNN
 */
trait MklDnnContainer extends DynamicContainer[Activity, Activity, Float] with MklDnnModule {
  protected val reorderManager = new ReorderManager()
  protected var mklDnnModules : Array[MklDnnModule] = _

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, Float]): this.type = {
    require(mklDnnModules == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  /**
   * Create MklDnnRuntime and compile the model
   * @param phase
   */
  final def compile(phase: Phase): Unit = {
    compile(phase, new MklDnnRuntime())
  }

  /**
   * Compile the model, which includes infer memory shapes, allocate memory, optimize computing
   * path and create MKL-DNN primitives
   * @param phase
   * @param runtime
   */
  final def compile(phase: Phase, runtime: MklDnnRuntime): Unit = {
    freeze()
    inputFormats().foreach(f => require(f.isLayoutFixed(), "Model input layout should be fixed"))
    fusion(phase)
    inferShape(inputFormats.map(_.shape))
    initFwdPrimitives(runtime, phase)
    if (phase == Phase.TrainingPhase) {
      initBwdPrimitives(runtime, phase)
      initGradWPrimitives(runtime, phase)
    }
    initMemory()
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

  override private[mkldnn] def initMemory() = {
    modules.foreach { case m: MklDnnModule => m.initMemory() }
  }
}
