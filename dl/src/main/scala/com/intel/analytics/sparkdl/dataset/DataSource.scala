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

package com.intel.analytics.sparkdl.dataset

import scala.collection.Iterator

trait DataSource[T] extends Iterator[T] {
  def reset(): Unit

  def shuffle(): Unit

  def finished(): Boolean

  // scalastyle:off methodName
  def ++[C](transformer: Transformer[T, C]): DataSource[C] = {
    val curDataSource = this
    new DataSource[C] {
      private val iterator = transformer.transform(curDataSource)

      override def reset(): Unit = curDataSource.reset

      override def shuffle(): Unit = curDataSource.shuffle

      override def next(): C = iterator.next

      override def hasNext: Boolean = iterator.hasNext

      override def total(): Long = curDataSource.total()

      override def finished(): Boolean = curDataSource.finished()
    }
  }

  // scalastyle:on methodName

  def total(): Long
}

trait Transformer[A, B] extends Serializable {
  def transform(prev: Iterator[A]): Iterator[B]
}
