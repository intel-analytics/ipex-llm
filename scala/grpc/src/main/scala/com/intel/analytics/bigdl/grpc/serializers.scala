/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.grpc

import com.fasterxml.jackson.databind.{DeserializationFeature, MapperFeature, ObjectMapper, SerializationFeature}
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

trait SerializeSuported {
  def serialize(src: Object): String

  def deSerialize[T](clazz: Class[T], data: String): T
}

class JacksonYamlSerializer extends SerializeSuported {
  val mapper = new ObjectMapper(new YAMLFactory()) with ScalaObjectMapper
  mapper.registerModule(DefaultScalaModule)
  mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true)
  mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false)
  mapper.configure(SerializationFeature.INDENT_OUTPUT, true)
  mapper

  override def serialize(src: Object): String = {
    mapper.writeValueAsString(src)
  }

  override def deSerialize[T](clazz: Class[T], data: String): T = {
    mapper.readValue[T](data, clazz)
  }
}

class JacksonJsonSerializer extends SerializeSuported {
  val mapper = new ObjectMapper() with ScalaObjectMapper
  mapper.registerModule(DefaultScalaModule)
  mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true)
  mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false)
  mapper.configure(SerializationFeature.INDENT_OUTPUT, true)
  mapper

  override def serialize(src: Object): String = {
    mapper.writeValueAsString(src)
  }

  override def deSerialize[T](clazz: Class[T], dest: String): T = {
    mapper.readValue[T](dest, clazz)
  }
}

object YamlUtil {
  val jacksonYamlSerializer = new JacksonYamlSerializer()

  def fromYaml[T](clazz: Class[T], data: String)(implicit m: Manifest[T]): T =
    jacksonYamlSerializer.deSerialize[T](clazz, data)

  def toYaml(value: Object): String = jacksonYamlSerializer.serialize(value)
}

object JsonUtil {
  val jacksonJsonSerializer = new JacksonJsonSerializer()

  def fromJson[T](clazz: Class[T], dest: String): T = jacksonJsonSerializer.deSerialize(clazz, dest)

  def toJson(value: Object): String = jacksonJsonSerializer.serialize(value)
}
