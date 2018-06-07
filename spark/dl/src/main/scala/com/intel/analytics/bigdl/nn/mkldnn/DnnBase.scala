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
  private[mkldnn] def expectInputFormats: Array[MemoryData]
  private[mkldnn] def setInputFormats(formats: Array[MemoryData]): Unit = {
    // Override this method for layer which allow to modify the input formats
    // The default behavior is skip
  }

  /**
   * Compute the output formats based on the input formats
   */
  private[mkldnn] def inferOutputFormats(): Array[MemoryData]

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
  def compile(phase: Phase, inputFormats: Array[MemoryData]): Unit = {
    compile(phase, inputFormats, new MklDnnRuntime())
  }

  /**
   * Compile the model, which includes infer memory shapes, allocate memory, optimize computing
   * path and create MKL-DNN primitives
   * @param phase
   * @param inputFormats
   * @param runtime
   */
  def compile(phase: Phase, inputFormats: Array[MemoryData], runtime: MklDnnRuntime): Unit = {
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
