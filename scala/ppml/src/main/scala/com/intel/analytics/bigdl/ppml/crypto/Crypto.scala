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

package com.intel.analytics.bigdl.ppml.crypto

import com.intel.analytics.bigdl.ppml.utils.Supportive

trait Crypto extends Supportive with Serializable {
  def encryptFile(sourceFilePath:String, saveFilePath:String, dataKeyPlaintext:String)
  def decryptFile(sourceFilePath:String, saveFilePath:String, dataKeyPlaintext:String)
  def encryptBytes(sourceBytes:Array[Byte], dataKeyPlaintext:String): Array[Byte]
  def decryptBytes(sourceBytes:Array[Byte], dataKeyPlaintext:String): Array[Byte]
}

object CryptoMode extends Enumeration {
  type CryptoMode = Value
  val PLAIN_TEXT = Value("plain_text", "plain_text")
  val AES_CBC_PKCS5PADDING = Value("AES/CBC/PKCS5Padding", "AES/CBC/PKCS5Padding")
  val UNKNOWN = Value("UNKNOWN", "UNKNOWN")
  class EncryptModeEnumVal(name: String, val value: String) extends Val(nextId, name)
  protected final def Value(name: String, value: String): EncryptModeEnumVal = {
    new EncryptModeEnumVal(name, value)
  }
  def parse(s: String): Value = {
    values.find(_.toString.toLowerCase() == s.toLowerCase).getOrElse(CryptoMode.UNKNOWN)
  }
}
