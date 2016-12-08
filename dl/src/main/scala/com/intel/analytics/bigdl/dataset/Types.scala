/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
 * Represent a local file path of an image file
 * @param path
 */
class ImageLocalPath(val path : Path)

/**
 * Represent a local file path of a hadoop sequence file
 * @param path
 */
case class SeqFileLocalPath(val path: Path)

/**
 * Represent a label
 * @tparam T
 */
trait Label[T] {
  def setLabel(label: T): this.type
  def label(): T
}
