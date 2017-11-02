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

package com.intel.analytics.bigdl.utils.keras

import java.nio.file.{Files, Paths}

import com.amazonaws.services.cloudfront.model.InvalidArgumentException
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import ncsa.hdf.hdf5lib.HDF5Constants._
import org.apache.log4j.Logger
import play.api.libs.functional.syntax._
import play.api.libs.json._

import scala.reflect.ClassTag
import ncsa.hdf.hdf5lib.{H5, HDF5Constants}

import scala.util.control.Exception


class KerasLoader[T: ClassTag](kerasJsonPath: String)(implicit ev: TensorNumeric[T]) {
  private val logger = Logger.getLogger(getClass)
  private val nameToBigDLNode = Map[String, ModuleNode[T]]()
  private val nameToBigDLLayer = Map[String, ModuleNode[T]]()
  private val kerasJson = this.loadKerasJsonFromPath(kerasJsonPath)
  private val converter = new Keras1LayerConverter[T](kerasJson)


  def loadModule(): AbstractModule[Activity, Activity, T] = {
    loadModule(loadKerasJsonFromPath(kerasJsonPath))
  }

  def loadModule(kerasJson: KModel): AbstractModule[Activity, Activity, T] = {
    converter.createGraph(kerasJson)
  }


  def loadKerasJsonFromPath(path: String): KModel = {
    val jsonStr = readFileToString(path)
    val kerasJson = new Keras1DefinitionParser[KModel]().parseModel(jsonStr)
    kerasJson
  }

  private def readFileToString(path: String): String = {
    new String(Files.readAllBytes(Paths.get(path)))
  }

}
