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

package com.intel.analytics.bigdl.tensor

import scala.reflect.ClassTag

/**
 * Storage defines a simple storage interface that controls the underlying storage for
 * any tensor object.
 *
 */
trait Storage[T] extends Iterable[T] with Serializable {

  /**
   * Returns the number of elements in the storage. The original method name in torch is size,
   * which is conflict with
   * Iterable
   *
   * @return
   */
  def length(): Int

  /**
   * Returns the number of elements in the storage. Override `size` in Iterable, which will cause
   * some full gc and performance issues.
   *
   * @return
   */
  override def size: Int = length()

  /**
   * Returns the element at position index in the storage. Valid range of index is 0 to length() -1
   *
   * @param index
   * @return
   */
  // ToDo: make the index range from 1 to length() like torch
  def apply(index: Int): T

  /**
   * Set the element at position index in the storage. Valid range of index is 1 to length()
   *
   * @param index
   * @param value
   */
  def update(index: Int, value: T): Unit

  /**
   * Copy another storage. The types of the two storages might be different: in that case a
   * conversion of types occur
   * (which might result, of course, in loss of precision or rounding). This method returns itself.
   *
   * @param source
   * @return
   */
  def copy(source: Storage[T], offset: Int, sourceOffset: Int, length: Int): this.type

  def copy(source: Storage[T]): this.type = copy(source, 0, 0, length())

  /**
   * Fill the Storage with the given value. This method returns itself.
   *
   * @param value
   * @param offset offset start from 1
   * @param length length of fill part
   * @return
   */
  def fill(value: T, offset: Int, length: Int): this.type

  /**
   * Resize the storage to the provided size. The new contents are undetermined.
   * This function returns itself
   *
   * @param size
   * @return
   */
  def resize(size: Long): this.type

  /**
   * Convert the storage to an array
   *
   * @return
   */
  def array(): Array[T]

  /**
   * Get the element type in the storage
   *
   * @return
   */
  //  def getType() : DataType

  /**
   * Share the same underlying storage
   *
   * @param other
   * @return
   */
  def set(other: Storage[T]): this.type
}

object Storage {
  def apply[T: ClassTag](): Storage[T] = new ArrayStorage[T](new Array[T](0))

  def apply[T: ClassTag](size: Int): Storage[T] = new ArrayStorage[T](new Array[T](size))

  def apply[@specialized(Float, Double) T: ClassTag](data: Array[T]): Storage[T] =
    new ArrayStorage[T](data)
}
