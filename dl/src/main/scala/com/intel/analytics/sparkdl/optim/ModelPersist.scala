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

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{Activities, File, Table}

trait ModelPersist[@specialized(Float, Double) T] {

  private var modelSaveInterval: Option[Int] = None

  private var path: Option[String] = None

  private var isOverWrite = true

  def setModelSaveInterval(modelSaveInterval: Int): this.type = {
    require(modelSaveInterval > 0)
    this.modelSaveInterval = Some(modelSaveInterval)
    this
  }

  def setPath(path: String): this.type = {
    if (path != null) {
      this.path = Some(path)
    }
    this
  }

  def setOverWrite(isOverWrite: Boolean): this.type = {
    this.isOverWrite = isOverWrite
    this
  }


  def saveModel(
    model: Module[_ <: Activities, _ <: Activities, T],
    iter: Int,
    force: Boolean = false): this.type = {
    if (this.path.isDefined) {
      require(model != null)

      if (iter == 0) {
        File.save(model, path.get, isOverWrite)
      } else if (modelSaveInterval.isDefined && iter % modelSaveInterval.get == 0) {
        File.save(model, s"${path.get}.$iter", isOverWrite)
      }
    }

    this
  }

  def saveModel(model: Module[_ <: Activities, _ <: Activities, T]): this.type = {
    saveModel(model, 0, true)
  }

  def saveState(state: Table, iter: Int, force: Boolean = false): this.type = {
    if (this.path.isDefined) {
      require(state != null)

      if (iter == 0) {
        File.save(state, s"${path.get}.state", isOverWrite)
      } else if (modelSaveInterval.isDefined && iter % modelSaveInterval.get == 0) {
        File.save(state, s"${path.get}.state.$iter", isOverWrite)
      }
    }

    this
  }

  def saveState(state: Table): this.type = {
    saveState(state, 0, true)
  }

}
