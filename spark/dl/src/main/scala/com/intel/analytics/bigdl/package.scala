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
package com.intel.analytics

import java.util.Properties

import com.intel.analytics.bigdl.dataset.AbstractDataSet
import com.intel.analytics.bigdl.nn.abstractnn.Activity

import scala.language.implicitConversions

package object bigdl {
  type Module[T] =
    com.intel.analytics.bigdl.nn.abstractnn.AbstractModule[Activity, Activity, T]
  type Criterion[T] =
    com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion[Activity, Activity, T]

  implicit def convModule[T](
    module: com.intel.analytics.bigdl.nn.abstractnn.AbstractModule[_, _, T]
  ): Module[T] = module.asInstanceOf[Module[T]]

  implicit def convCriterion[T](
    criterion: com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion[_, _, T]
  ): Criterion[T] = criterion.asInstanceOf[Criterion[T]]

  val numeric = com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

  type DataSet[D] = AbstractDataSet[D, _]

  private object BigDLBuildInfo {

      val version: String = {

      val resourceStream = Thread.currentThread().getContextClassLoader.
        getResourceAsStream("bigdl-version-info.properties")

      try {
        val unknownProp = "<unknown>"
        val props = new Properties()
        props.load(resourceStream)
        props.getProperty("version", unknownProp)
      } catch {
        case npe: NullPointerException =>
          throw new IllegalArgumentException
          ("Error while locating file bigdl-version-info.properties")
        case e: Exception =>
          throw new IllegalArgumentException
          ("Error loading properties from bigdl-version-info.propertiess")
      } finally {
        if (resourceStream != null) {
          try {
            resourceStream.close()
          } catch {
            case e: Exception =>
              throw new IllegalArgumentException
              ("Error closing bigdl build info resource stream", e)
          }
        }
      }
    }
  }

  val BIGDL_VERSION = BigDLBuildInfo.version
}
