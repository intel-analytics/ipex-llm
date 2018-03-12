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

import java.io._

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.utils.serializer.converters._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DataReaderWriterSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var inputStream : DataInputStream = null
  var outputStream : DataOutputStream = null
  var tmpFile : File = null

  before {
    tmpFile = File.createTempFile("testWeight", "bin")
    inputStream = new DataInputStream(new FileInputStream(tmpFile))
    outputStream = new DataOutputStream(new FileOutputStream(tmpFile))
  }

  "Float read/write" should "work properly" in {
    val flts = Array[Float](1.0f, 2.0f)
    FloatReaderWriter.write(outputStream, flts)
    outputStream.flush
    val readFloats = FloatReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Float]].array
    flts should be (readFloats)
  }

  "Double read/write" should "work properly" in {
    val dbs = Array[Double](1.0, 2.0)
    DoubleReaderWriter.write(outputStream, dbs)
    outputStream.flush
    val readDoubles = DoubleReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Double]].array
    dbs should be (readDoubles)
  }

  "Char read/write" should "work properly" in {
    val chs = Array[Char]('a', 'b')
    CharReaderWriter.write(outputStream, chs)
    outputStream.flush
    val readChars = CharReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Char]].array
    chs should be (readChars)
  }

  "Bool read/write" should "work properly" in {
    val bools = Array[Boolean](true, false)
    BoolReaderWriter.write(outputStream, bools)
    outputStream.flush
    val readBools = BoolReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Boolean]].array
    bools should be (readBools)
  }

  "String read/write" should "work properly" in {
    val strs = Array[String]("abc", "123")
    StringReaderWriter.write(outputStream, strs)
    outputStream.flush
    val readStrs = StringReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[String]].array
    strs should be (readStrs)
  }

  "Int read/write" should "work properly" in {
    val ints = Array[Int](1, 2)
    IntReaderWriter.write(outputStream, ints)
    outputStream.flush
    val readInts = IntReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Int]].array
    ints should be (readInts)
  }

  "Short read/write" should "work properly" in {
    val shorts = Array[Short](1, 2)
    ShortReaderWriter.write(outputStream, shorts)
    outputStream.flush
    val readShorts = ShortReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Short]].array
    shorts should be (readShorts)
  }

  "Long read/write" should "work properly" in {
    val longs = Array[Long](1, 2)
    LongReaderWriter.write(outputStream, longs)
    outputStream.flush
    val readLongs = LongReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Long]].array
    longs should be (readLongs)
  }

  "ByteString read/write" should "work properly" in {
    val bytStrs = Array[ByteString](ByteString.copyFromUtf8("abc"))
    ByteStringReaderWriter.write(outputStream, bytStrs)
    outputStream.flush
    val readBytStrs = ByteStringReaderWriter.read(inputStream, 1).
      asInstanceOf[Storage[ByteString]].array
    bytStrs should be (readBytStrs)
  }

  "Byte read/write" should "work properly" in {
    val byts = Array[Byte](1, 2)
    ByteReaderWriter.write(outputStream, byts)
    outputStream.flush
    val readBytes = ByteReaderWriter.read(inputStream, 2).
      asInstanceOf[Storage[Byte]].array
    byts should be (readBytes)
  }

  after {
    if (tmpFile.exists) {
      tmpFile.delete
    }
    if (inputStream != null) {
      inputStream.close
    }
    if (outputStream != null) {
      outputStream.close
    }
  }
}
