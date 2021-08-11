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

package com.intel.analytics.bigdl.dataset

import java.nio.file.Path

/**
 * Represent an image
 */
abstract class Image extends Serializable {
  def width(): Int

  def height(): Int

  def content: Array[Float]
}

 /**
  * Represent a sentence
  */
abstract class Sentence[T] extends Serializable {
  def dataLength(): Int

  def data(): Array[T]
}

/**
 * Represent a local file path of an image file
 *
 * @param path
 */
class LocalImagePath(val path : Path)

/**
 * Represent a local file path of a hadoop sequence file
 *
 * @param path
 */
case class LocalSeqFilePath(val path: Path)

/**
 * Represent a label
 *
 * @tparam T
 */
trait Label[T] {
  def setLabel(label: T): this.type
  def label(): T
}

/**
 * A byte array and a label. It can contain anything.
 * @param data
 * @param label
 */
case class ByteRecord(data: Array[Byte], label: Float)
