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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericChar, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericString}
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule

import scala.collection.mutable
import scala.reflect.ClassTag


trait StorageType
object ProtoStorageType extends StorageType
object BigDLStorage extends StorageType

case class SerializeContext[T: ClassTag](moduleData: ModuleData[T],
                                         storages: mutable.HashMap[Int, Any],
                                         storageType: StorageType,
                                         copyWeightAndBias : Boolean = true,
                                         groupType : String = null)
case class DeserializeContext(bigdlModule : BigDLModule,
                              storages: mutable.HashMap[Int, Any],
                              storageType: StorageType,
                              copyWeightAndBias : Boolean = true)

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
  val MODULE_TAGES = "module_tags"
  val MODULE_NUMERICS = "module_numerics"
  val GROUP_TYPE = "group_type"
}

object ClassTagMapper {
  def apply(tpe : String): ClassTag[_] = {
    tpe match {
      case "Float" => scala.reflect.classTag[Float]
      case "Double" => scala.reflect.classTag[Double]
      case "Char" => scala.reflect.classTag[Char]
      case "Boolean" => scala.reflect.classTag[Boolean]
      case "String" => scala.reflect.classTag[String]
      case "Int" => scala.reflect.classTag[Int]
      case "Long" => scala.reflect.classTag[Long]
      case "com.google.protobuf.ByteString" => scala.reflect.classTag[ByteString]
    }
  }

  def apply(classTag: ClassTag[_]): String = classTag.toString
}
object TensorNumericMapper {
  def apply(tpe : String): TensorNumeric[_] = {
    tpe match {
      case "Float" => NumericFloat
      case "Double" => NumericDouble
      case "Char" => NumericChar
      case "Boolean" => NumericBoolean
      case "String" => NumericString
      case "Int" => NumericInt
      case "Long" => NumericLong
      case "ByteString" => NumericByteString
    }
  }

  def apply(tensorNumeric: TensorNumeric[_]): String = {
    tensorNumeric match {
      case NumericFloat => "Float"
      case NumericDouble => "Double"
      case NumericChar => "Char"
      case NumericBoolean => "Boolean"
      case NumericString => "String"
      case NumericInt => "Int"
      case NumericLong => "Long"
      case NumericByteString => "ByteString"
    }
  }
}
