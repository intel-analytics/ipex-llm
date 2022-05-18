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

package com.intel.analytics.bigdl.ppml.utils
import java.io.PrintWriter
import scala.io.Source

class KeyReaderWriter {

  def writeKeyToFile(encryptedKeyPath: String, encryptedKeyContent: String): PrintWriter = {
    new PrintWriter(encryptedKeyPath) {
      write(encryptedKeyContent); close
    }
  }

  def readKeyFromFile(encryptedKeyPath: String): String = {
    val encryptedKeyCiphertext: String = Source.fromFile(encryptedKeyPath).getLines.next()
    encryptedKeyCiphertext
  }

}

