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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.log4j.Logger

import scala.math._
import scala.reflect.ClassTag

/**
 * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
 * @param learningRate learning rate
 * @param learningRateDecay learning rate decay
 * @param beta1 first moment coefficient
 * @param beta2 second moment coefficient
 * @param Epsilon for numerical stability
 * @tparam T
 */
class EmbeddingAdam[@specialized(Float, Double) T: ClassTag](
      var learningRate: Double = 1e-3,
      var learningRateDecay: Double = 0.0,
      var beta1: Double = 0.9,
      var beta2: Double = 0.999,
      var Epsilon: Double = 1e-8)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  @transient
  private var ones: Tensor[T] = null

  /**
   * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
                        parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.Epsilon

    val (fx, dfdx) = feval(parameter)

    var timestep = state.getOrElse[Int]("evalCounter", 0)

    val clr = lr / (1 + timestep*lrd)

    timestep = timestep + 1

    val parallelNum = Engine.coreNumber()
    val gradLength = parameter.nElement()
    val taskSize = gradLength / parallelNum
    val extraTask = gradLength % parallelNum
    if (ones == null || ones.nElement() < taskSize + 1) {
      ones = Tensor[T]().resize(taskSize + 1).fill(ev.one)
    }

    val times = new Array[Long](parallelNum)

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      val start = System.nanoTime()
      val offset = tid * taskSize + math.min(tid, extraTask)
      val length = taskSize + (if (tid < extraTask) 1 else 0)
      val currentDfdx = dfdx.narrow(1, offset + 1, length)
      val currentParameter = parameter.narrow(1, offset + 1, length)
      val currentOnes = ones.narrow(1, 1, length)
      val (_s, _r, _denom) =
        if (state.get[Tensor[T]](s"s$tid").isDefined && state.get[Tensor[T]](s"r$tid").isDefined
          && state.get[Tensor[T]](s"denom$tid").isDefined) {
          (state.get[Tensor[T]](s"s$tid").get, state.get[Tensor[T]](s"r$tid").get,
            state.get[Tensor[T]](s"denom$tid").get)
        } else {
          (Tensor[T]().resizeAs(currentParameter).zero(),
            Tensor[T]().resizeAs(currentParameter).zero(),
            Tensor[T]().resizeAs(currentParameter).zero())
        }
      ParallelAdam.updateFrame(_s, _r, _denom, clr, currentDfdx, currentParameter,
        beta1, beta2, timestep, currentOnes, eps)

      state(s"s$tid") = _s // 1st moment variables
      state(s"r$tid") = _r // 2nd moment variables
      state(s"denom$tid") = _denom // 3nd moment variables
      times(tid) = (System.nanoTime() - start) / 1000000L
    }))

    ParallelAdam.logger
      .info(s"update ${parameter.nElement()} parameters, maximum time is ${times.max} ms")
    ParallelAdam.logger.info(s"Time is ${times.mkString("\t")} ms")


    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon

    (parameter, Array(fx))
  }


  var embeddingNoutput = 0
  var embeddingNIndex = 0
  var lastUpdated = Array.tabulate(embeddingNIndex)(_ => 1)
//  var lastUpdated = collection.mutable.HashMap(
//    Array.tabulate(embeddingNoutput)(i => (i + 1, 0)): _*)

  // TODO: clear before saving
  var s: Array[Tensor[T]] = _
  var r: Array[Tensor[T]] = _
  var denom: Array[Tensor[T]] = _
  var buffer: Array[Tensor[T]] = _
  def setNOutput(nIndex: Int, nOutput: Int): Unit = {
    embeddingNoutput = nOutput
    embeddingNIndex = nIndex
//    lastUpdated = collection.mutable.HashMap(
//      Array.tabulate(nIndex)(i => (i + 1, 1)): _*)
    lastUpdated = Array.tabulate(nIndex)(_ => 1)
    ones = Tensor(embeddingNoutput).fill(ev.one)
    s = Array.tabulate(nIndex)(_ => Tensor[T](nOutput))
    r = Array.tabulate(nIndex)(_ => Tensor[T](nOutput))
    denom = Array.tabulate(nIndex)(_ => Tensor[T](nOutput))
    buffer = Array.tabulate(nIndex)(_ => Tensor[T](nOutput))
    (1 to nIndex).foreach{i =>
      state(s"s$i") = Tensor[Float]() // 1st moment variables
      state(s"r$i") = Tensor[Float]() // 2nd moment variables
      state(s"denom$i") = Tensor[Float]() // 3nd moment variables
    }
  }

  def updateNograd(indices: Tensor[T], parameter: Tensor[T]): Unit = {
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.Epsilon

    val uniqueStart = System.nanoTime()
    val uniqueIndices = Tensor[T](Storage(indices.storage().array().distinct))
    println(s"unique indices ${System.nanoTime() - uniqueStart}")

    var timestep = state.getOrElse[Int]("evalCounter", 1)

    val clr = lr / (1 + (timestep - 1) *lrd)

    val parallelNum = Engine.coreNumber()
    val gradLength = uniqueIndices.nElement()
    val taskSize = gradLength / parallelNum
    val extraTask = gradLength % parallelNum

//    val times = new Array[Long](parallelNum)

    var updateTime = System.nanoTime()
    (0 until parallelNum).map(tid => {
//      Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      val offset = tid * taskSize + math.min(tid, extraTask)
      val length = taskSize + (if (tid < extraTask) 1 else 0)
      val currentIndex = uniqueIndices.narrow(1, offset + 1, length)
      var i = 1
      while(i <= currentIndex.nElement()) {
        val index = ev.toType[Int](currentIndex.valueAt(i))
        if (timestep > lastUpdated(index - 1)) {
          val (_s, _r, _denom, _buffer) =
            (s(index - 1), r(index - 1), denom(index - 1), buffer(index - 1))
          val indexThParameter = parameter.narrow(1,
            (index - 1) * embeddingNoutput + 1, embeddingNoutput)
//          println(s"update index ${index}")
          ParallelAdam.updateFrameZeroGrad(
            timestep, lastUpdated(index - 1),
            _s, _r, _denom, _buffer, clr, indexThParameter,
            beta1, beta2, ones, eps)
          lastUpdated(index - 1) = timestep
        }
//        println(index)
        i += 1
      }
//      Adam.logger.info(s"zero grad${tid} $i ${ev.toType[Int](currentIndex.valueAt(i - 1))}")
//      times(tid) = (System.nanoTime() - start) / 1000000L
    })
//  Adam.logger.info(s"update ${parameter.nElement()} parameters, maximum time is ${times.max} ms")
//    Adam.logger.info(s"Time is ${times.mkString("\t")} ms")
    println(s"${parallelNum}nograd update frame time cost ${System.nanoTime() - updateTime}")


    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon
  }

//  def updateNograd(indices: Tensor[T], parameter: Array[Tensor[T]]): Unit = {
//    val lr = this.learningRate
//    val lrd = this.learningRateDecay
//    val beta1 = this.beta1
//    val beta2 = this.beta2
//    val eps = this.Epsilon
//
//    val uniqueStart = System.nanoTime()
//    val uniqueIndices = Tensor[T](Storage(indices.storage().array().distinct))
//    println(s"unique indices ${System.nanoTime() - uniqueStart}")
//
//    var timestep = state.getOrElse[Int]("evalCounter", 1)
//
//    val clr = lr / (1 + (timestep - 1) *lrd)
//
//    val parallelNum = Engine.coreNumber()
//    val gradLength = uniqueIndices.nElement()
//    val taskSize = gradLength / parallelNum
//    val extraTask = gradLength % parallelNum
//
//    //    val times = new Array[Long](parallelNum)
//
//    var updateTime = System.nanoTime()
//    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
//      val offset = tid * taskSize + math.min(tid, extraTask)
//      val length = taskSize + (if (tid < extraTask) 1 else 0)
//      val currentIndex = uniqueIndices.narrow(1, offset + 1, length)
//      var i = 1
//      while(i <= currentIndex.nElement()) {
//        val index = ev.toType[Int](currentIndex.valueAt(i))
//        if (timestep > lastUpdated(index - 1)) {
//          val (_s, _r, _denom, _buffer) =
//            (s(index - 1), r(index - 1), denom(index - 1), buffer(index - 1))
//          val indexThParameter = parameter(index - 1)
//          //          println(s"update index ${index}")
//          ParallelAdam.updateFrameZeroGrad(
//            timestep, lastUpdated(index - 1),
//            _s, _r, _denom, _buffer, clr, indexThParameter,
//            beta1, beta2, ones, eps)
//          lastUpdated(index - 1) = timestep
//        }
//        //        println(index)
//        i += 1
//      }
//    }))
//    println(s"${parallelNum}nograd update frame time cost ${System.nanoTime() - updateTime}")
//
//
//    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon
//  }

  /**
   * update embedding gradient
   * @param dfdx input -> gradOutput
   * @param parameter
   * @return
   */
  def optimizeEmbedding(
                                 dfdx: Array[(Tensor[T], Tensor[T])],
                                 parameter: Tensor[T]): Unit = {
    // TODO: assert input is unique
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.Epsilon

    var timestep = state.getOrElse[Int]("evalCounter", 1)

    val clr = lr / (1 + timestep*lrd)

//    timestep = timestep + 1

    val parallelNum = Engine.coreNumber()
//    val gradLength = parameter.nElement()
//    val taskSize = gradLength / parallelNum
//    val extraTask = gradLength % parallelNum
//    if (ones == null || ones.nElement() < taskSize + 1) {
//      ones = Tensor[T]().resize(taskSize + 1).fill(ev.one)
//    }

    val times = new Array[Long](parallelNum)

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
//      val start = System.nanoTime()
      val currentGradient = dfdx(tid)
      val currentIndex = currentGradient._1
      val currentDfdx = currentGradient._2
      var i = 1
      while(i <= currentIndex.nElement()) {
        val index = ev.toType[Int](currentIndex.valueAt(i))
        val (_s, _r, _denom) = (s(index - 1), r(index - 1), denom(index - 1))
        val indexThParameter = parameter.narrow(1,
          (index - 1) * embeddingNoutput + 1, embeddingNoutput)
        val iThGradient = currentDfdx.select(1, i)
        ParallelAdam.updateFrame(
          _s, _r, _denom, clr, iThGradient, indexThParameter,
          beta1, beta2, timestep, ones, eps)
        lastUpdated(index - 1) = timestep + 1
        i += 1
      }
//      Adam.logger.info(s"update grad${tid} $i ${ev.toType[Int](currentIndex.valueAt(i - 1))}")
    }))


    // TODO: timestep
    timestep = timestep + 1
    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("s")
    state.delete("r")
  }

  override def getLearningRate(): Double = this.learningRate
}

