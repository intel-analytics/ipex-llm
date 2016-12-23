/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.utils

import java.io._
import java.nio.file._

object File {

  /**
   * Load torch object from a torch binary file
   *
   * @param fileName file name.
   * @return An instance of T
   */
  def loadTorch[T](fileName: String): T = {
    TorchFile.load[T](fileName)
  }

  /**
   * Save scala object into a torch binary file
   *
   * @param source  The object to be saved.
   * @param fileName file name to saving.
   * @param objectType The object type.
   * @param overWrite If over write.
   */
  def saveTorch(
      source: Any,
      fileName: String,
      objectType: TorchObject,
      overWrite: Boolean = false): Unit = {
    TorchFile.save(source, fileName, objectType, overWrite)
  }

  /**
   * Save scala object into a Java object file
   *
   * @param obj obj to be saved.
   * @param fileName file name saved to.
   * @param isOverwrite if overwrite.
   */
  def save(obj: Serializable, fileName: String, isOverwrite: Boolean = false): Unit = {
    if (Files.exists(Paths.get(fileName)) && !isOverwrite) {
      throw new RuntimeException("file exists!")
    }

    val objFile = new ObjectOutputStream(new FileOutputStream(fileName))
    objFile.writeObject(obj)
  }

  /**
   * Load object from a Java object file
   *
   * @param fileName file name.
   */
  def load[T](fileName: String): T = {
    val objFile = new ObjectInputStream(new FileInputStream(fileName))
    objFile.readObject().asInstanceOf[T]
  }
}
