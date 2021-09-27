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

package com.intel.analytics.bigdl.serving.serialization

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

class StreamSerializer {

}

object StreamSerializer {
  def objToBytes(obj: Object): Array[Byte] = {
    val bos = new ByteArrayOutputStream()
    val out = new ObjectOutputStream(bos)
    try {
      out.writeObject(obj)
      out.flush()
      bos.toByteArray()
    } catch {
      case e: Exception => e.printStackTrace()
        throw new Error("Ser error")
    } finally {
      try {
        bos.close()
      } catch {
        case e: Exception => // ignore close exception
      }
    }
  }
  def bytesToObj(bytes: Array[Byte]): Object = {
    val bis = new ByteArrayInputStream(bytes)
    val in = new ObjectInputStream(bis)
    try {
      in.readObject()
    } catch {
      case e: Exception => e.printStackTrace()
        throw new Error("Deser error")
    } finally {
      try {
        bis.close()
      } catch {
        case e: Exception => // ignore close exception
      }
    }
  }
}
