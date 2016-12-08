/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import java.util.concurrent.{LinkedBlockingQueue, TimeUnit, ThreadPoolExecutor}

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet}
import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._

object LocalOptimizer {
  // Todo: Move this into engine
  var recordsArray: Array[Int] = null
}

class LocalOptimizer[T](
  dataset: LocalDataSet[(Tensor[T], Tensor[T])],
  model: Module[Activities, Activities, T],
  criterion: Criterion[Tensor[T], T],
  optimMethod: OptimMethod[T],
  coreNumber: Int,
  state: Table,
  endWhen: Trigger
)(implicit ev : TensorNumeric[T]) extends Optimizer[T](model, endWhen) {

  import LocalOptimizer._

  val (weights, grad) = model.getParameters()

  private val subModelNumber = Engine.backend match {
    case MKL_BLAS() => coreNumber
    case MKL_DNN() => 1
  }

  private val workingModels = (1 to subModelNumber).map(_ => model.cloneModule()).toArray

  private val workingModelWAndG = workingModels.map(_.getParameters())

  workingModelWAndG.foreach(_._1.storage().set(weights.storage()))

  private val workingCriterions =
    (1 to subModelNumber).map(_ => criterion.cloneCriterion()).toArray

  override def optimize(): Module[Activities, Activities, T] = {
    var wallClockTime = 0L
    var count = 0

    optimMethod.clearHistory(state)
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    dataset.shuffle()
    var iter = dataset.data()
    while (!endWhen(state)) {
      val start = System.nanoTime()

      // Fetch data and prepare tensors
      val batch = iter.next()
      var b = 0
      require(batch._1.size(1) == batch._2.size(1))
      // require(batch._1.size(1) % subModelNumber == 0)
      val stackSize = batch._1.size(1) / subModelNumber
      val extraSize = batch._1.size(1) % subModelNumber
      val parallelism = if(stackSize == 0) extraSize else subModelNumber
      val tensorBuffer = new Array[(Tensor[T], Tensor[T])](parallelism)
      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize)
        val length = stackSize + (if(b < extraSize) 1 else 0)
        tensorBuffer(b) = (batch._1.narrow(1, offset + 1, length),
          batch._2.narrow(1, offset + 1, length))
        b += 1
      }

      val dataFetchTime = System.nanoTime()

      if (recordsArray == null || recordsArray.length < subModelNumber) {
        recordsArray = new Array[Int](subModelNumber)
      }

      val loss = Engine.invokeAndWait(
        (0 until parallelism).map(i =>
          () => {
            val localModel = workingModels(i)
            localModel.zeroGradParameters()
            localModel.training()
            val localCriterion = workingCriterions(i)
            val (input, target) = tensorBuffer(i)
            val output = localModel.forward(input).asInstanceOf[Tensor[T]]
            val _loss = ev.toType[Double](localCriterion.forward(output, target))
            val errors = localCriterion.backward(output, target)
            localModel.backward(input, errors)
            recordsArray(i) = target.size(1)
            _loss
          })
      ).sum

      // copy multi-model gradient to the buffer
      val gradLength = grad.nElement()
      val taskSize = gradLength / subModelNumber
      val extraTask = gradLength % subModelNumber
      val parallelNum = if (taskSize == 0) extraTask else subModelNumber

      Engine.invokeAndWait(
        (0 until parallelNum).map(tid =>
          () => {
            val offset = tid * taskSize + math.min(tid, extraTask)
            val length = taskSize + (if (tid < extraTask) 1 else 0)
            var i = 0
            while (i < parallelism) {
              if (i == 0) {
                grad.narrow(1, offset + 1, length)
                  .copy(workingModelWAndG(i)._2.narrow(1, offset + 1, length))
              } else {
                grad.narrow(1, offset + 1, length)
                  .add(workingModelWAndG(i)._2.narrow(1, offset + 1, length))
              }
              i += 1
            }
          })
      )

      optimMethod.optimize(_ => (ev.fromType(loss), grad), weights, state)
      val end = System.nanoTime()
      wallClockTime += end - start
      count += batch._1.size(1)
      println(s"[Epoch ${state[Int]("epoch")} $count/${dataset.size()}][Iteration ${
        state[Int]("neval")}][Wall Clock ${wallClockTime / 1e9
      }s] loss is $loss, iteration time is ${(end - start) / 1e9}s data " +
        s"fetch time is " +
        s"${(dataFetchTime - start) / 1e9}s, train time ${(end - dataFetchTime) / 1e9}s." +
        s" Throughput is ${batch._1.size(1).toDouble / (end - start) * 1e9} img / second")
      state("neval") = state[Int]("neval") + 1

      if (count >= dataset.size()) {
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        iter = dataset.data()
        count = 0
      }

      validate(wallClockTime)
      checkpoint(wallClockTime)
    }

    model
  }

  private def checkpoint(wallClockTime: Long): Unit = {
    if(cacheTrigger.isEmpty || cachePath.isEmpty) {
      return
    }

    val trigger = cacheTrigger.get
    val path = cachePath.get
    if (trigger(state) && cachePath.isDefined) {
      println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to path")
      saveModel(this.model, s".${state[Int]("neval")}")
      saveState(state, s".${state[Int]("neval")}")
    }
  }

  private def validate(wallClockTime: Long): Unit = {
    if(validationTrigger.isEmpty || validationDataSet.isEmpty) {
      return
    }
    val trigger = validationTrigger.get
    if(!trigger(state)) {
      return
    }
    val vMethods = validationMethods.get
    val dataIter = validationDataSet.get.asInstanceOf[LocalDataSet[(Tensor[T], Tensor[T])]].data()
    println(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")

    workingModels.foreach(_.evaluate())

    var count = 0
    dataIter.map(batch => {
      require(batch._1.size(1) == batch._2.size(1))
      val stackSize = batch._1.size(1) / subModelNumber
      val extraSize = batch._1.size(1) % subModelNumber
      val parallelism = if(stackSize == 0) extraSize else subModelNumber
      val result = Engine.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize)
            val length = stackSize + (if(b < extraSize) 1 else 0)
            val input = batch._1.narrow(1, offset + 1, length)
            val target = batch._2.narrow(1, offset + 1, length)
            val output = workingModels(b).forward(input)
            vMethods.map(validation => {
              validation(output.asInstanceOf[Tensor[T]], target)
            })
          }
        )
      ).reduce((left, right) => {
        left.zip(right).map { case (l, r) =>
          l + r
        }
      })
      count += batch._1.size(1)
      println(s"[Validation] $count/${validationDataSet.get.size()}")
      result
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods).foreach(r => {
      println(s"${r._2} is ${r._1}")
    })
  }
}

