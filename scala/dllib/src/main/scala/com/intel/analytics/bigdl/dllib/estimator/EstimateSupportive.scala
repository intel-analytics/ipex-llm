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

package com.intel.analytics.bigdl.dllib.estimator
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import org.slf4j.LoggerFactory

import scala.reflect.ClassTag

/**
 * EstimateSupportive trait to provide some AOP methods as utils
 *
 */
 trait EstimateSupportive {

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    EstimateSupportive.logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  /**
   * calculate and log the time and throughput
   *
   * @param name    the name of the process
   * @param batch   the number of the batch
   * @param f       the process function
   * @tparam T      the return of the process function
   * @return        the result of the process function
   */
  def throughputing[T](name: String, batch: Int)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    val throughput = batch.toDouble / cost * 1000
    EstimateSupportive.logger.info(
      s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms], " +
        s"throughput: ${throughput} records/second.")
    result
  }

  /**
   * calculate and log the time and throughput and loss
   *
   * @param name  the name of the process
   * @param batch the number of the batch
   * @param f     the process function
   * @tparam T    the return of the process function
   * @return      the result of the process function
   */
  def throughputingWithLoss[T](name: String, batch: Int)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    val throughput = batch.toDouble / cost * 1000
    EstimateSupportive.logger.info(
      s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms], " +
        s"throughput: ${throughput} records/second, loss: ${result}.")
    result
  }

  def clearWeightBias(model: Module[Float]): Unit = {
    model.reset()
    val weightBias = model.parameters()._1
    val clonedWeightBias = model.parameters()._1.map(tensor => {
      val newTensor = Tensor[Float]().resizeAs(tensor)
      newTensor.copy(tensor)
    })
    val localWeightBias = model.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(clonedWeightBias(i))
      }
      i += 1
    }
    releaseTensors(model.parameters()._1)
    releaseTensors(model.parameters()._2)
  }

  def makeMetaModel(original: AbstractModule[Activity, Activity, Float]):
  AbstractModule[Activity, Activity, Float] = {
    val metaModel = original.cloneModule()
    releaseWeightBias(metaModel)
    metaModel
  }

  def releaseWeightBias(model: Module[Float]): Unit = {
    model.reset()
    releaseTensors(model.parameters()._1)
    releaseTensors(model.parameters()._2)
  }

  private def releaseTensors[T: ClassTag](tensors: Array[Tensor[T]])
                                         (implicit ev: TensorNumeric[T]) = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        tensors(i).set()
      }
      i += 1
    }
  }

  def makeUpModel(clonedModel: Module[Float], weightBias: Array[Tensor[Float]]):
  AbstractModule[Activity, Activity, Float] = {
    putWeightBias(clonedModel, weightBias)
    clonedModel.evaluate()
    clonedModel
  }

  private def putWeightBias(target: Module[Float], weightBias: Array[Tensor[Float]]):
  Module[Float] = {
    val localWeightBias = target.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(weightBias(i))
      }
      i += 1
    }
    target
  }
 }

 object EstimateSupportive {
  val logger = LoggerFactory.getLogger(getClass)
 }
