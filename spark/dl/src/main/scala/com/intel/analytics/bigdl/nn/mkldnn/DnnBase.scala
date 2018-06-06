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
import com.intel.analytics.bigdl.nn.{Sequential => Seq}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.DnnTensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

sealed class Phase
case object TrainingPhase extends Phase
case object InferencePhase extends Phase

sealed class MemoryFormat(val shape: Array[Int], val layout: Int)
case class HeapData(_shape: Array[Int], _layout: Int) extends MemoryFormat(_shape, _layout)
case class NativeData(_shape: Array[Int], _layout: Int) extends MemoryFormat(_shape, _layout)

private[mkldnn] object MemoryFormat {
  def isCompatible(actuals: Array[MemoryFormat], expects: Array[MemoryFormat]): Boolean = {
    if (actuals.length != expects.length) return false
    actuals.zip(expects).foreach { case (actual, expect) =>
      if (!isSizeCompatible(actual, expect)) return false
      actual match {
        case h: HeapData =>
          expect match {
            case hh: HeapData =>
              if (hh.layout != MklDnn.MemoryFormat.any && hh.layout != h.layout) return false
            case nn: NativeData => return false
            case _ => throw new UnsupportedOperationException("Not support such memory format")
          }
        case n: NativeData =>
          expect match {
            case hh: HeapData => return false
            case nn: NativeData =>
              if (nn.layout != MklDnn.MemoryFormat.any && nn.layout != n.layout) return false
            case _ => throw new UnsupportedOperationException("Not support such memory format")
          }
        case _ => throw new UnsupportedOperationException("Not support such memory format")
      }
    }
    return true
  }

  def isSizeCompatible(actual: MemoryFormat, expect: MemoryFormat): Boolean = {
    if (actual == null || expect == null) return true
    if (actual.shape.length != expect.shape.length) return false
    actual.shape.zip(expect.shape).foreach {case (a, e) => if (a != e) return false}
    return true
  }
}

class MklDnnRuntime {
  val engine : Long = MklDnn.EngineCreate(MklDnn.EngineType.cpu, 0)
  val stream : Long = MklDnn.StreamCreate(MklDnn.StreamType.eager)
}

/**
 * Helper utilities when integrating containers with MKL-DNN
 */
trait MklDnnContainer extends MklDnnModule {

  /**
   * Execution path
   */
  protected var executions: Array[MklDnnModule] = _

  /**
   * Create MklDnnRuntime and compile the model
   * @param phase
   * @param inputFormats
   */
  def compile(phase: Phase, inputFormats: Array[MemoryFormat]): Unit = {
    compile(phase, inputFormats, new MklDnnRuntime())
  }

  /**
   * Compile the model, which includes infer memory shapes, allocate memory, optimize computing
   * path and create MKL-DNN primitives
   * @param phase
   * @param inputFormats
   * @param runtime
   */
  def compile(phase: Phase, inputFormats: Array[MemoryFormat], runtime: MklDnnRuntime): Unit = {
    this.phase = phase
    inferOutputFormats()
    fusion()
    initPrimitives(runtime)
    allocateMemory()
  }

  /**
   * Modify the computing path by fuse some layers into one to improve the performance
   */
  protected def fusion(): Unit = {}

  override private[mkldnn] def allocateMemory() = {
    executions.foreach(m => {
      m.allocateMemory()
    })
  }
}

class Sequential[T: ClassTag](implicit ev: TensorNumeric[T]) extends Seq[T] with MklDnnContainer {
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(executions == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  override private[mkldnn] def initPrimitives(runtime: MklDnnRuntime) = {
    executions.foreach(m => {
      m.phase = phase
      m.initPrimitives(runtime)
    })
    modules.clear()
    modules.appendAll(executions.map(_.asInstanceOf[AbstractModule[Activity, Activity, T]]))
  }

  override private[mkldnn] def inferOutputFormats(): Array[MemoryFormat] = {
    var lastOutputFormats = expectInputFormats
    val executionsBuffer = new ArrayBuffer[MklDnnModule]()
    modules.foreach(m => {
      require(m.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
      val _m = m.asInstanceOf[MklDnnModule]

      _m.setInputFormats(lastOutputFormats)
      require(MemoryFormat.isCompatible(lastOutputFormats, _m.expectInputFormats),
        "memory format is not compatible with expected one")
      lastOutputFormats = _m.inferOutputFormats()
      executionsBuffer.append(_m)
    })
    executions = executionsBuffer.toArray
    lastOutputFormats
  }

  override private[mkldnn] def setInputFormats(formats: Array[MemoryFormat]): Unit = {
    modules(0).asInstanceOf[MklDnnModule].setInputFormats(formats)
  }

  override private[mkldnn] def expectInputFormats = {
    modules(0).asInstanceOf[MklDnnModule].expectInputFormats
  }
}

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
   * Which phase the current model is working on
   */
  private[mkldnn] var phase: Phase = _

  /**
   * Module input formats
   */
  private[mkldnn] def expectInputFormats: Array[MemoryFormat]
  private[mkldnn] def setInputFormats(formats: Array[MemoryFormat]): Unit = {
    // Override this method for layer which allow to modify the input formats
    // The default behavior is skip
  }

  /**
   * Compute the output formats based on the input formats
   */
  private[mkldnn] def inferOutputFormats(): Array[MemoryFormat]

  /**
   * MKL-DNN primitives of the module. Note you should only initialize this field by calling
   * initPrimitives method. This field will be erased when sending model to remote worker. So you
   * need to reinitialize it after sending the model.
   */
  @transient
  protected var forwardPrimitives: Array[Long] = _
  @transient
  protected var backwardPrimitives: Array[Long] = _
  @transient
  protected var inputPrimitives: Array[Long] = _
  @transient
  protected var outputPrimitives: Array[Long] = _

  /**
   * Init the MKL-DNN primitives for the model
   * @param runtime
   */
  private[mkldnn] def initPrimitives(runtime: MklDnnRuntime): Unit

  private[mkldnn] def allocateMemory(): Unit
}
