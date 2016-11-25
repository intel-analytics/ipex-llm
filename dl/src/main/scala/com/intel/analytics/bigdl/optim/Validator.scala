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

import com.intel.analytics.bigdl.dataset.{LocalDataSet, RDDDataSet}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Activities

abstract class Validator[T](vMethods: Array[ValidationMethod[T]]) {
  def validate(model: Module[Activities, Activities, T])
  : Array[(ValidationMethod[T], ValidationResult)]
}

class LocalValidator[T](vMethods: Array[ValidationMethod[T]],
  validationData: LocalDataSet[(Tensor[T], Tensor[T])]) extends Validator[T](vMethods) {

  override def validate(model: Module[Activities, Activities, T])
  : Array[(ValidationMethod[T], ValidationResult)] = {
    model.evaluate()
    validationData.reset()
    val iter = validationData.data()
    var count = 0
    val results = iter.map { case (input, target) =>
      val output = model.forward(input)
      println(s"[Validation] $count/${validationData.size()}")
      count += input.size(1)
      vMethods.map(validation => {
        validation(output.asInstanceOf[Tensor[T]], target)
      })
    }.reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    })
    vMethods.zip(results)
  }
}

class RDDValidator[T](vMethods: Array[ValidationMethod[T]],
  validationData: RDDDataSet[(Tensor[T], Tensor[T])]) extends Validator[T](vMethods) {

  private val broadcastVMethods = validationData.partitions().sparkContext.broadcast(vMethods)

  override def validate(model: Module[Activities, Activities, T])
  : Array[(ValidationMethod[T], ValidationResult)] = {
    model.clearState()
    model.evaluate()
    val broadcastModel = validationData.partitions().sparkContext.broadcast(model)
    val results = validationData.data().mapPartitions(dataIter => {
      val localModel = broadcastModel.value
      val localVMethods = broadcastVMethods.value
      val result = dataIter.map { case (input, target) =>
        val output = localModel.forward(input)
        localVMethods.map(validation => {
          validation(output.asInstanceOf[Tensor[T]], target)
        })
      }.reduce((left, right) => {
        left.zip(right).map { case (l, r) =>
          l + r
        }
      })
      Iterator.single(result)
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    })
    vMethods.zip(results)
  }
}
