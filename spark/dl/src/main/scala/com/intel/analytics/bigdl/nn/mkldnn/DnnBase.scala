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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable

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
   * Compute the output formats based on the input formats
   */
  private[mkldnn] def inferShape(shapes: Array[Array[Int]]): Array[Array[Int]]

  /**
   * Init the MKL-DNN primitives for the model
   * @param runtime
   */
  private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit
  private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit
  private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit

  private[mkldnn] def initMemory(): Unit

  private[mkldnn] def inputFormats(): Array[MemoryData]

  private[mkldnn] def gradInputFormats(): Array[MemoryData]

  private[mkldnn] def outputFormats(): Array[MemoryData]

  private[mkldnn] def gradOutputFormats(): (Array[MemoryData], Array[MemoryData])
}

trait MklDnnLayer extends MklDnnModule {
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
  protected var inputPrimitives: Array[Long] = _
  @transient
  protected var gradInputPrimitives: Array[Long] = _
  @transient
  protected var outputPrimitives: Array[Long] = _
  @transient
  protected var gradOutputPrimitives: Array[Long] = _
  @transient
  protected var gradOutputPrimitivesWeight: Array[Long] = _
}

/**
 * Helper utilities when integrating containers with MKL-DNN
 */
trait MklDnnContainer extends MklDnnModule {

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
  private[mkldnn] def fusion(phase: Phase): Unit
}

private[mkldnn] class ReorderManager() {

  val reorders = mutable.HashMap[(MemoryData, MemoryData), ReorderMemory]()
  val tensorCaches = mutable.HashMap[(Long, MemoryData), Tensor[Float]]()

  def register(from: MemoryData, to: MemoryData, runtime: MklDnnRuntime, phase: Phase): Unit = {
    if (needReorder(from, to) && !reorders.contains((from, to))) {
      val reorder = new ReorderMemory(from, to)
      reorder.initFwdPrimitives(runtime, phase)
      reorder.initMemory()
      reorders((from, to)) = reorder
    }
  }

  def infer(from: Array[MemoryData], to: Array[MemoryData], output: Activity)
  : Activity = {
    if (from.length == 1) {
      require(output.isTensor, "output activity should be a tensor")
      inferTensor(from(0), to(0), output.asInstanceOf[Tensor[Float]])
    } else {
      require(output.toTable.length() == from.length,
        "output activity length doesn't match")
      val outputTable = T()
      var i = 0
      while(i < from.length) {
        outputTable(i + 1) = inferTensor(from(i), to(i), output.toTable(i + 1))
        i += 1
      }
      output
    }
  }

  private def inferTensor(from: MemoryData, to : MemoryData, output: Tensor[Float])
  : Tensor[Float] = {
    //tensorCaches.getOrElse((System.identityHashCode(output), to), {
    if (reorders.contains((from, to))) {
      reorders((from, to)).forward(output)
    } else {
      output
    }
    //})
  }

  private def needReorder(from: MemoryData, to: MemoryData): Boolean = {
    from match {
      case h: HeapData =>
        to match {
          case hh: HeapData =>
            require(h.layout == hh.layout, "Heap data layout should be same")
            false
          case nn: NativeData => true
          case _ => throw new UnsupportedOperationException("Not support such memory format")
        }
      case n: NativeData =>
        to match {
          case hh: HeapData => true
          case nn: NativeData =>
            nn.layout != n.layout
          case _ => throw new UnsupportedOperationException("Not support such memory format")
        }
      case _ => throw new UnsupportedOperationException("Not support such memory format")
    }
  }
}
