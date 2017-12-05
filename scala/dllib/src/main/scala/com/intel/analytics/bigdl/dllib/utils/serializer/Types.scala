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

package com.intel.analytics.bigdl.utils.serializer

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import serialization.Bigdl.BigDLModule

import scala.collection.mutable
import scala.reflect.ClassTag


trait StorageType
object ProtoStorageType extends StorageType
object BigDLStorage extends StorageType

case class SerializeContext[T: ClassTag](moduleData: ModuleData[T],
                                         storages: mutable.HashMap[Int, Any],
                                         storageType: StorageType)
case class DeserializeContext(bigdlModule : BigDLModule,
                              storages: mutable.HashMap[Int, Any],
                              storageType: StorageType)

case class SerializeResult(bigDLModule: BigDLModule.Builder, storages: mutable.HashMap[Int, Any])

case class ModuleData[T: ClassTag](module : AbstractModule[Activity, Activity, T],
                                   pre : Seq[String], next : Seq[String])

object BigDLDataType extends Enumeration{
  type BigDLDataType = Value
  val FLOAT, DOUBLE, CHAR, BOOL, STRING, INT, SHORT, LONG, BYTESTRING, BYTE = Value
}

object SerConst {
  val MAGIC_NO = 3721
  val DIGEST_TYPE = "MD5"
  val GLOBAL_STORAGE = "global_storage"
}
