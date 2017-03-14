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

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.utils.{Engine, File, T, Table}
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class OptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()

  before {
    Engine.setNodeAndCore(1, 4)
  }

  "Optimizer" should "end with maxEpoch" in {
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        val state = T("epoch" -> 9)
        endWhen(state) should be(false)
        state("epoch") = 10
        endWhen(state) should be(false)
        state("epoch") = 11
        endWhen(state) should be(true)
        model
      }
    }
    dummyOptimizer.setEndWhen(Trigger.maxEpoch(10)).optimize()
  }

  it should "end with iteration" in {
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        val state = T("neval" -> 999)
        endWhen(state) should be(false)
        state("neval") = 1000
        endWhen(state) should be(false)
        state("neval") = 1001
        endWhen(state) should be(true)
        model
      }
    }
    dummyOptimizer.setEndWhen(Trigger.maxIteration(1000)).optimize()
  }

  it should "be triggered every epoch" in {
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        val state = T("epoch" -> 9)
        validationTrigger.get(state) should be(false)
        checkpointTrigger.get(state) should be(false)
        state("epoch") = 10
        validationTrigger.get(state) should be(true)
        checkpointTrigger.get(state) should be(true)
        validationTrigger.get(state) should be(false)
        checkpointTrigger.get(state) should be(false)
        state("epoch") = 11
        validationTrigger.get(state) should be(true)
        checkpointTrigger.get(state) should be(true)
        checkpointPath.isDefined should be(true)
        model
      }
    }
    dummyOptimizer.setValidation(Trigger.everyEpoch, null, null)
    dummyOptimizer.setCheckpoint("", Trigger.everyEpoch)
    dummyOptimizer.optimize()
  }

  it should "be triggered every 5 iterations" in {
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        val state = T("neval" -> 1)
        validationTrigger.get(state) should be(false)
        checkpointTrigger.get(state) should be(false)
        state("neval") = 4
        validationTrigger.get(state) should be(false)
        checkpointTrigger.get(state) should be(false)
        state("neval") = 5
        validationTrigger.get(state) should be(true)
        checkpointTrigger.get(state) should be(true)
        model
      }
    }
    dummyOptimizer.setValidation(Trigger.severalIteration(5), null, null)
    dummyOptimizer.setCheckpoint("", Trigger.severalIteration(5))
    dummyOptimizer.optimize()
  }

  it should "save model to given path" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val model = AlexNet(1000)
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        Optimizer.saveModel(model, this.checkpointPath, this.isOverWrite)
        model
      }
    }
    dummyOptimizer.setCheckpoint(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    model.clearState()
    val loadedModel = File.load[Module[Double]] (filePath + "/model")
    loadedModel should be(model)
  }

  it should "save model and state to given path with postfix" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val model = AlexNet(1000)
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        Optimizer.saveModel(model, this.checkpointPath, this.isOverWrite, ".test")
        model
      }
    }
    dummyOptimizer.setCheckpoint(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    model.clearState()
    val loadedModel =
      File.load[Module[Double]](filePath + "/model.test")
    loadedModel should be(model)
  }

  it should "save state to given path" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "state").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val state = T("test" -> 123)
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        Optimizer.saveState(state, this.checkpointPath, this.isOverWrite)
        model
      }
    }.setState(state)
    dummyOptimizer.setCheckpoint(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    val loadedState = File.load[Table](filePath + "/state")
    loadedState should be(state)
  }

  it should "save state to given path with post fix" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "state").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val state = T("test" -> 123)
    val dummyOptimizer = new Optimizer[Float, Float](model, null, null) {
      override def optimize(): Module[Float] = {
        Optimizer.saveState(state, this.checkpointPath, this.isOverWrite, ".post")
        model
      }
    }.setState(state)
    dummyOptimizer.setCheckpoint(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    val loadedState = File.load[Table](filePath + "/state.post")
    loadedState should be(state)
  }

  "A Distributed dataset" should "spawn a distributed optimizer" in {
    val ds = new DistributedDataSet[Float] {
      override def originRDD(): RDD[_] = null
      override def data(train: Boolean): RDD[Float] = null
      override def size(): Long = 0
      override def shuffle(): Unit = {}
    }

    val model = Linear[Float](4, 3)
    val criterion = ClassNLLCriterion[Float]()
    val res = Optimizer(model, ds, criterion)
    res.isInstanceOf[DistriOptimizer[Float]] should be(true)
    res.isInstanceOf[LocalOptimizer[Float]] should be(false)
  }

  "A Local dataset" should "spawn a local optimizer" in {
    val ds = new LocalDataSet[Float] {
      override def data(train: Boolean): Iterator[Float] = null
      override def size(): Long = 0
      override def shuffle(): Unit = {}
    }

    val model = Linear[Float](4, 3)
    val criterion = ClassNLLCriterion[Float]()

    val res = Optimizer(model, ds, criterion)
    res.isInstanceOf[DistriOptimizer[Float]] should be(false)
    res.isInstanceOf[LocalOptimizer[Float]] should be(true)
  }
}
