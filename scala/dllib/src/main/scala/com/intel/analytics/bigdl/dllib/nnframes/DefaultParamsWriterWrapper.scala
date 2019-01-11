/*
 * Copyright 2018 Analytics Zoo Authors.
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
package org.apache.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter}
import org.json4s.jackson.JsonMethods.{compact, render}
import org.json4s.{DefaultFormats, JObject, JValue}


object DefaultParamsWriterWrapper {
  def saveMetadata(
                    instance: Params,
                    path: String,
                    sc: SparkContext,
                    extraMetadata: Option[JObject] = None,
                    paramMap: Option[JValue] = None): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, paramMap)
  }

  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
  }

  def getAndSetParams(instance: Params, metadata: Metadata): Unit = {
    implicit val format = DefaultFormats
    metadata.params match {
      case JObject(pairs) =>
        pairs.foreach { case (paramName, jsonValue) =>
          val param = instance.getParam(paramName)
          val value = param.jsonDecode(compact(render(jsonValue)))
          instance.set(param, value)
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata: ${metadata.metadataJson}.")
    }
  }
}
