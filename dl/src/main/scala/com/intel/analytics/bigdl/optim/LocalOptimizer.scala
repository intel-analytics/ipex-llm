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

import com.intel.analytics.bigdl.dataset.DataSource
import com.intel.analytics.bigdl.nn.{Criterion, Module, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_FLOAT_TENSOR
import com.intel.analytics.bigdl.utils.{Activities, File, Table}

class LocalOptimizer[T](
  data: DataSource[(Tensor[T], Tensor[T])],
  validationData: DataSource[(Tensor[T], Tensor[T])],
  model: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T],
  optimMethod: OptimMethod[T],
  state: Table,
  endWhen: Trigger
) extends Optimizer[T](model, endWhen) {

  def this(
    data: DataSource[(Tensor[T], Tensor[T])],
    model: Module[Tensor[T], Tensor[T], T],
    criterion: Criterion[Tensor[T], T],
    optimMethod: OptimMethod[T],
    state: Table,
    endWhen: Trigger) = this(data, null, model, criterion, optimMethod, state, endWhen)

  override def optimize(): Module[Tensor[T], Tensor[T], T] = {
    val (weights, grad) = model.getParameters()
    var wallClockTime = 0L
    var count = 0
    var iBatch = 1

    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    data.reset()
    data.shuffle()

//
//
//
//    count = 0
//    val suffix = ".t7"
//    val testDataInput = Tensor[Float](10000, 3, 32, 32)
//    val testDataLabel = Tensor[Float](10000, 1)
//    while (count < validationData.total()) {
//      val (input, target) = validationData.next()
//      testDataInput.narrow(1, count+1, input.size(1)).copy(input.asInstanceOf[Tensor[Float]])
//      testDataLabel.narrow(1, count+1, input.size(1)).copy(target.asInstanceOf[Tensor[Float]])
//      count += input.size(1)
//    }
//
//    val tmpTestInput = java.io.File.createTempFile("torchdata/data_test_batch_input", suffix)
//    val tmpTestLabel = java.io.File.createTempFile("torchdata/data_test_batch_label", suffix)
//    val tmpTestPathInput = tmpTestInput.getAbsolutePath
//    val tmpTestPathLabel = tmpTestLabel.getAbsolutePath
//    File.saveTorch(testDataInput, tmpTestPathInput, TYPE_FLOAT_TENSOR)
//    File.saveTorch(testDataLabel, tmpTestPathLabel, TYPE_FLOAT_TENSOR)
//    println(s"save test to file ${tmpTestPathInput}")
//
//
//    val trainDataInput = Tensor[Float](50000, 3, 32, 32)
//    val trainDataLabel = Tensor[Float](50000, 1)
    count = 0
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val (input, target) = data.next()
//      trainDataInput.narrow(1, count+1, input.size(1)).copy(input.asInstanceOf[Tensor[Float]])
//      trainDataLabel.narrow(1, count+1, input.size(1)).copy(target.asInstanceOf[Tensor[Float]])

      val dataFetchTime = System.nanoTime()
      model.zeroGradParameters()
      val output = model.forward(input)
      val loss = criterion.forward(output, target)
      val gradOutput = criterion.backward(output, target)
      model.backward(input, gradOutput)
      optimMethod.optimize(_ => (loss, grad), weights, state)
      val end = System.nanoTime()
      wallClockTime += end - start
      count += input.size(1)
      println(s"[Epoch ${state[Int]("epoch")} $count/${data.total()}][Iteration ${
        state[Int]("neval")}][Wall Clock ${wallClockTime / 1e9
      }s] loss is $loss, iteration time is ${(end - start) / 1e9}s data " +
        s"fetch time is " +
        s"${(dataFetchTime - start) / 1e9}s, train time ${(end - dataFetchTime) / 1e9}s." +
        s" Throughput is ${input.size(1).toDouble / (end - start) * 1e9} img / second")
      state("neval") = state[Int]("neval") + 1

      if(count >= data.total()) {
//        val tmpInput = java.io.File.createTempFile("torchdata/data_batch_input_" + state("epoch").toString + "_", suffix)
//        val tmpLabel = java.io.File.createTempFile("torchdata/data_batch_label_" + state("epoch").toString + "_", suffix)
//        val tmpPathInput = tmpInput.getAbsolutePath
//        val tmpPathLabel = tmpLabel.getAbsolutePath
//        File.saveTorch(trainDataInput, tmpPathInput, TYPE_FLOAT_TENSOR)
//        File.saveTorch(trainDataLabel, tmpPathLabel, TYPE_FLOAT_TENSOR)
//        println(s"save train to file ${tmpPathInput}")
        state("epoch") = state[Int]("epoch") + 1
        data.reset()
        data.shuffle()
        count = 0
      }

      validate(wallClockTime)

      cacheTrigger.foreach(trigger => {
        if (trigger(state) && cachePath.isDefined) {
          println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
          saveModel(s".${state[Int]("neval")}")
          saveState(state, s".${state[Int]("neval")}")
        }
      })
    }
    validate(wallClockTime)

    model
  }

  private def validate(wallClockTime: Long): Unit = {
    validationTrigger.foreach(trigger => {
      if (trigger(state) && validationMethods.length > 0) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
        model.evaluate()
        validationData.reset()
        var count = 0
        val results = validationData.map { case (input, target) =>
          val output = model.forward(input)
          println(s"[Validation][Epoch ${state[Int]("epoch")}][Iteration ${state[Int]("neval")}] " +
            s"$count/${validationData.total()}")
          count += input.size(1)
          validationMethods.map(validation => {
            validation(output.asInstanceOf[Tensor[T]], target)
          }).toArray
        }.reduce((left, right) => {
          left.zip(right).map { case (l, r) =>
            l ++ r
          }
        })
        validationMethods.zip(results).foreach {
          case (validation, result) =>
            println(s"[Wall Clock ${wallClockTime / 1e9}s] $validation is $result")
        }
        model.training()
      }
    })
  }
}

