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

package com.intel.analytics.bigdl.utils

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import scala.reflect.ClassTag
import scala.language.implicitConversions
import scala.collection.mutable.{IndexedSeq, WrappedArray}


class SerializableIndexedSeq[T: ClassTag](@transient private var impl: IndexedSeq[T])
  extends IndexedSeq[T] with Serializable {

  override def length: Int = impl.length

  override def apply(idx: Int): T = impl.apply(idx)

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream) = {
    out.defaultWriteObject()
    out.writeObject(impl.toArray)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream) = {
    in.defaultReadObject()
    val array = in.readObject().asInstanceOf[Array[T]]
    require(array != null)
    impl = array
  }

  override def update(idx: Int, elem: T): Unit = impl.update(idx, elem)
}

object SerializableIndexedSeq {

  def apply[T: ClassTag](impl: IndexedSeq[T]): SerializableIndexedSeq[T] =
    new SerializableIndexedSeq(impl)

  implicit def indexedSeq2Serializable[T: ClassTag](impl: IndexedSeq[T])
  : SerializableIndexedSeq[T] = apply(impl)

}
