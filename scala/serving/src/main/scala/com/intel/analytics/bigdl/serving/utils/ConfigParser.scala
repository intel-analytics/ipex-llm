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

package com.intel.analytics.bigdl.serving.utils

import java.io.{File, FileInputStream}

import com.fasterxml.jackson.databind.{DeserializationFeature, MapperFeature, ObjectMapper, SerializationFeature}
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor


class ConfigParser(configPath: String) {

  def loadConfig(): ClusterServingHelper = {
    try {
      val configStr = scala.io.Source.fromFile(configPath).mkString
      val mapper = new ObjectMapper(new YAMLFactory())
      mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true)
      mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
      mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false)
      val helper = mapper.readValue[ClusterServingHelper](configStr, classOf[ClusterServingHelper])
      helper.parseConfigStrings()
      try {
        // if no modelPath is set, just skip, this would be ok in some tests
        // if not set in runtime, error would be raised in model loading process
        helper.parseModelType(helper.modelPath)
      } catch {
        case e: Error =>
      }

      helper
    }
    catch {
      case e: Exception =>
        println(s"Invalid configuration, please check type regulations in config file")
        e.printStackTrace()
        throw new Error("Configuration parsing error")
    }
  }


}

