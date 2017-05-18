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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File, T, Table}
import org.apache.commons.lang3.SerializationUtils

import scala.reflect.ClassTag

/**
 * Similar to torch Optim method, which is used to update the parameter
 */
trait OptimMethod[@specialized(Float, Double) T] extends Serializable {

  /**
   * Optimize the model parameter
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation,
   *                  and returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list, evaluated before the update
   */
  def optimize(feval: (Tensor[T]) => (T, Tensor[T]), parameter: Tensor[T])
  : (Tensor[T], Array[T])

  /**
   * Clear the history information in the OptimMethod state
   *
   * @return
   */
  def clearHistory(): Unit

  /**
   * Update hyper parameter.
   * We have updated hyper parameter in method optimize(). But in DistriOptimizer, the method
   * optimize() is only called on the executor side, the driver's hyper parameter is unchanged.
   * So this method is using to update hyper parameter on the driver side.
   *
   * @return A string.
   */
  def updateHyperParameter(): Unit = {}


  /**
   * Get hyper parameter from config table.
   *
   */
  def getHyperParameter(): String = ""

  /**
   * clone OptimMethod
   *
   * @return
   */
  override def clone(): OptimMethod[T] = SerializationUtils.clone(this)

  /**
   * get learning rate
   *
   * @return
   */
  def getLearningRate(): Double

  // a table describing the state of the optimizer; after each call the state is modified
  private[bigdl] var state: Table = T(
    "epoch" -> 1,
    "neval" -> 1
  )

  /**
   * save OptimMethod
   *
   * @param path      path
   * @param overWrite whether to overwrite
   * @return
   */
  def save(path: String, overWrite: Boolean = false): this.type = {
    this.clearHistory()
    File.save(this, path, overWrite)
    this
  }

  /**
   * load optimMethod parameters from Table
   * @param config
   * @return
   */
  def loadFromTable(config: Table): this.type

  /**
   * Optimize the model parameter
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation,
   *                  and returns f(X) and df/dX
   * @param parameter the initial point
   * @param config    a table with configuration parameters for the optimizer
   * @param state     a table describing the state of the optimizer; after each call the state
   *                  is modified
   * @return the new x vector and the function list, evaluated before the update
   */
  @deprecated(
    "Please initialize OptimMethod with parameters when creating it instead of importing table",
    "0.2.0")
  def optimize(feval: (Tensor[T]) => (T, Tensor[T]), parameter: Tensor[T], config: Table,
               state: Table = null): (Tensor[T], Array[T]) = {
    val _config = if (config == null) T() else config
    this.state = if (state == null) _config else state
    this.loadFromTable(config)
    val res = optimize(feval, parameter)
    config.update("clr", this.getLearningRate())
    res
  }

  /**
   * Clear the history information in the state
   *
   * @param state
   * @return
   */
  @deprecated(
    "Please use clearHistory() instead",
    "0.2.0")
  def clearHistory(state: Table): Table = {
    this.state = state
    this.clearHistory()
    this.state
  }

  /**
   * Update hyper parameter.
   * We have updated hyper parameter in method optimize(). But in DistriOptimizer, the method
   * optimize() is only called on the executor side, the driver's hyper parameter is unchanged.
   * So this method is using to update hyper parameter on the driver side.
   *
   * @param config config table.
   * @param state  state Table.
   * @return A string.
   */
  @deprecated(
    "Please use updateHyperParameter() instead",
    "0.2.0")
  def updateHyperParameter(config: Table, state: Table): Unit = {}

  /**
   * Get hyper parameter from config table.
   *
   * @param config a table contains the hyper parameter.
   */
  @deprecated(
    "Please use getHyperParameter() instead",
    "0.2.0")
  def getHyperParameter(config: Table): String = ""

}

object OptimMethod {
  /**
   * load optim method
   * @param path file path
   * @return
   */
  def load[T: ClassTag](path : String) : OptimMethod[T] = {
    File.load[OptimMethod[T]](path)
  }
}
