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

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn}
import com.intel.analytics.bigdl.nn.Dropout
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Dropout masks(set to zero) parts of input using a bernoulli distribution.
 * Each input element has a probability initP of being dropped. If `scale` is
 * true(true by default), the outputs are scaled by a factor of `1/(1-initP)`
 * during training.
 * During evaluating, output is the same as input.
 *
 * It has been proven an effective approach for regularization and preventing
 * co-adaptation of feature detectors. For more details, plese see
 * [Improving neural networks by preventing co-adaptation of feature detectors]
 * (https://arxiv.org/abs/1207.0580)
 *
 * @param initP the probability p
 * @param inplace whether to make `input` and `output` share the same storage
 * @param scale whether to scale the output by a factor of `1 / (1 - p)`
 */
@SerialVersionUID(- 4636332259181125718L)
class DropoutDnn(
  val initP: Double = 0.5,
  val inplace: Boolean = false,
  var scale: Boolean = true)(
                  implicit ev: TensorNumeric[Float]) extends TensorModule[Float] {
  private var p = initP
  var noise = Tensor[Float]()
  var isResampling = true

  @transient
  protected var results: Array[Future[Unit]] = null

  private var gradBuffer = Tensor[Float]()
  /**
   * Get current probability to be dropped.
   * @return p
   */
  def getP(): Float = {
    return p.toFloat
  }

  @transient
  private var inputElement : Int = 0
  @transient
  private var reorder_dst_memory : Long = 0L
  @transient
  private var reorder_src_memory : Long = 0L
  @transient
  private var update_primitive: Boolean = true
  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L
  val dataType = DataType.F32
  val stream_fwd = new ArrayBuffer[Long]

  def reorderTwoTensor(input: Tensor[Float], inputFormat: Int,
                       output: Tensor[Float], outputFormat: Int): Unit = {
    if (update_primitive) {
      val sizes = input.size()
      val dim = input.dim()
      output.resizeAs(input)

      val src_md = MklDnnOps.memoryDescInit(dim, sizes, dataType, inputFormat)
      val src_pd = MklDnnOps.memoryPrimitiveDescCreate(src_md, engine)

      val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, outputFormat)
      val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
      reorder_dst_memory = MklDnn.PrimitiveCreate0(user_pd)
      output.setPrimitiveDesc(user_pd)

      val res = MklDnnOps.prepareReorder(reorder_dst_memory, src_pd, false)
      reorder_src_memory = res._2

      stream_fwd.clear()
      stream_fwd.append(res._1)
    }
    /* build a simple net */
    val memoryPrimitives = Array(reorder_src_memory, reorder_dst_memory)
    val buffer = Array(input, output)
    MklDnnOps.streamSubmit(stream, 1, stream_fwd.toArray, 1, memoryPrimitives, buffer)
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    if (inplace) {
      this.output = input
    } else {
      this.output.resizeAs(input).copy(input)
    }

    if (inputElement != input.nElement()) {
      update_primitive = true
      inputElement = input.nElement()
    } else {
      update_primitive = false
    }

    if (results == null) {
      results = new Array[Future[Unit]](Engine.model.getPoolSize)
    }
    if (train) {
      noise.resizeAs(input)
      if (input.isContiguous()) {
        if (isResampling) {
          val noiseData = noise.storage().array()
          var taskSize = noise.nElement() / Engine.model.getPoolSize
          var extraTask = noise.nElement() % Engine.model.getPoolSize
          var allocated = 0
          val offset = this.output.storageOffset() - 1
          val data = this.output.storage.array()
          var i = 0
          while (allocated < noise.nElement()) {
            val start = allocated
            allocated += taskSize
            if (extraTask > 0) {
              allocated += 1
              extraTask -= 1
            }
            val end = allocated
            results(i) = Engine.model.invoke(() => {
              var k = start
              while (k < end) {
                noiseData(k) = if (RNG.bernoulli(1 - p)) {
                  if (scale) {
                    data(offset + k) = ev.divide(data(offset + k), ev.fromType[Double](1 - p))
                    ev.fromType[Double](1.0 / (1 - p))
                  } else {
                    ev.fromType[Int](1)
                  }
                } else {
                  data(offset + k) = ev.fromType[Int](0)
                  ev.fromType[Int](0)
                }

                k += 1
              }
            })
            i += 1

          }

          Engine.model.sync(results)
        } else {
          this.output.cmul(noise)
        }
        this.output
      } else {
        if (isResampling) {
          noise.bernoulli(1 - p)

          if (scale) {
            noise.div(ev.fromType[Double](1 - p))
          }
        }

        this.output.cmul(noise)
      }
    } else if (!scale) {
      this.output.mul(ev.fromType[Double](1 - p))
    } else {
      output
    }

    if (input.getPrimitiveDesc() != 0L) {
      output.setPrimitiveDesc(input.getPrimitiveDesc())
    }
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    if (results == null) {
      results = new Array[Future[Unit]](Engine.model.getPoolSize)
    }
    // refactor all to input format
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    val default_format = input.dim() match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.nc
      case 4 => Memory.Format.nchw
    }
    val user_format = input.getFormat() match {
      case -1 => default_format
      case _ => input.getFormat()
    }

    // refactor gradOutput to user_format
    if (gradOutput.getFormat() != input.getFormat() && gradOutput.getFormat() != user_format) {
      gradBuffer.resizeAs(gradOutput)
      reorderTwoTensor(gradOutput, gradOutput.getFormat(), gradBuffer, user_format)
    } else {
      gradBuffer = gradOutput
    }

    if (train) {
      if (inplace) {
        this.gradInput = gradBuffer
      } else {
        this.gradInput.resizeAs(gradBuffer).copy(gradBuffer)
      }

      if (gradInput.isContiguous()) {
        val noiseData = noise.storage().array()
        var taskSize = noise.nElement() / Engine.model.getPoolSize
        var extraTask = noise.nElement() % Engine.model.getPoolSize
        val gradInputData = gradInput.storage().array()
        val gradInputOffset = gradInput.storageOffset() - 1
        var allocated = 0
        var i = 0
        while (allocated < noise.nElement()) {
          val start = allocated
          allocated += taskSize
          if (extraTask > 0) {
            allocated += 1
            extraTask -= 1
          }
          val end = allocated
          results(i) = Engine.model.invoke(() => {
            var k = start
            while (k < end) {
              gradInputData(gradInputOffset + k) =
                ev.times(gradInputData(gradInputOffset + k), noiseData(k))
              k += 1
            }
          })
          i += 1
        }

        Engine.model.sync(results)

        this.gradInput
      } else {
        this.gradInput.cmul(noise)
      }
    } else {
      throw new IllegalArgumentException("backprop only defined while training")
    }

    if (user_format != default_format && update_primitive) {
      val md = MklDnnOps.memoryDescInit(gradInput.dim(), gradInput.size(), dataType, user_format)
      val pd = MklDnnOps.memoryPrimitiveDescCreate(md, engine)
      gradInput.setPrimitiveDesc(pd)
    }
    this.gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    noise.set()
    gradBuffer.set()
    this
  }

  /**
   * Set current probability to be dropped.
   * @param p new probability
   * @return
   */
  def setP(p: Double): this.type = {
    this.p = p
    this
  }

  override def toString(): String = {
    s"${getPrintName}($p)"
  }
}

object DropoutDnn {
  def apply(
    initP: Double = 0.5,
    inplace: Boolean = false,
    scale: Boolean = true) : DropoutDnn = {
    new DropoutDnn(initP, inplace, scale)
  }
}
