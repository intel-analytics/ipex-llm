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

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet}
import com.intel.analytics.bigdl.nn.{Criterion, Module, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Activities, Table}

class LocalOptimizer[T](
  data: LocalDataSet[(Tensor[T], Tensor[T])],
  model: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T],
  optimMethod: OptimMethod[T],
  state: Table,
  endWhen: Trigger
) extends Optimizer[T](endWhen) {

  override def optimize(): Module[Activities, Activities, T] = {
    val (weights, grad) = model.getParameters()
    var wallClockTime = 0L
    var count = 0

    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    val iter = data.data()
    data.reset()
    data.shuffle()
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val (input, target) = iter.next()
      val dataFetchTime = System.nanoTime()
      model.zeroGradParameters()
      val output = model.forward(input).toTensor[T]()
      val loss = criterion.forward(output, target)
      val gradOutput = criterion.backward(output, target)
      model.backward(input, gradOutput)
      optimMethod.optimize(_ => (loss, grad), weights, state)
      val end = System.nanoTime()
      wallClockTime += end - start
      count += input.size(1)
      println(s"[Epoch ${state[Int]("epoch")} $count/${data.size()}][Iteration ${
        state[Int]("neval")}][Wall Clock ${wallClockTime / 1e9
      }s] loss is $loss, iteration time is ${(end - start) / 1e9}s data " +
        s"fetch time is " +
        s"${(dataFetchTime - start) / 1e9}s, train time ${(end - dataFetchTime) / 1e9}s." +
        s" Throughput is ${input.size(1).toDouble / (end - start) * 1e9} img / second")
      state("neval") = state[Int]("neval") + 1

      if (count >= data.size()) {
        state("epoch") = state[Int]("epoch") + 1
        data.reset()
        data.shuffle()
        count = 0
      }

      validate(wallClockTime)
      cache(wallClockTime)
    }
    validate(wallClockTime)
    cache(wallClockTime)

    model.asInstanceOf[Module[Activities, Activities, T]]
  }

  private def cache(wallClockTime: Long): Unit = {
    cacheTrigger.foreach(trigger => {
      if (trigger(state) && cachePath.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
        saveModel(this.model.asInstanceOf[Module[Activities, Activities, T]],
          s".${state[Int]("neval")}")
        saveState(state, s".${state[Int]("neval")}")
      }
    })
  }

  private def validate(wallClockTime: Long): Unit = {
    validationTrigger.foreach(trigger => {
      if (trigger(state) && validator.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
        val results = validator.get.validate(
          this.model.asInstanceOf[Module[Activities, Activities, T]])
        results.foreach(r => {
          println(s"${r._1} is ${r._2}")
        })
        model.training()
      }
    })
  }
}

