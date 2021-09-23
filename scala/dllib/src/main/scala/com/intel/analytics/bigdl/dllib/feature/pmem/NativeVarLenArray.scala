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
//
// package com.intel.analytics.bigdl.dllib.feature.pmem
//
// import org.apache.spark.unsafe.Platform
//
// import scala.reflect.ClassTag
//
// private[bigdl] abstract class NativeVarLenArray[T: ClassTag](val recordNum: Int,
//    totalSizeByBytes: Long,
//    memoryType: MemoryType, protected val moveStep: Int) extends
//  NativeArray[Array[T]](totalSizeByBytes, memoryType) {
//
//  // TODO: maybe this can be changed to native long array
//  val indexer = Array.fill[Long](recordNum + 1)(-1L)
//
//  indexer(0) = startAddr
//
//  protected def isValidIndex(i: Int): Boolean = {
//    i < recordNum && indexer(i + 1) != -1
//  }
//
//  protected def getRecordLength(i: Int): Int = {
//    assert(isValidIndex(i), s"Invalid index: ${i}")
//    ((indexer(i + 1) - indexer(i)) >> moveStep).toInt
//  }
//
//  protected def getTypeOffSet(): Int
//
//  def get(i: Int): Array[T] = {
//    assert(isValidIndex(i), s"Invalid index ${i}")
//    val recordLen = getRecordLength(i)
//    val result = new Array[T](recordLen)
//    Platform.copyMemory(null, indexOf(i), result,
//      getTypeOffSet(), (recordLen << moveStep))
//    return result
//  }
//
//  // TODO: would be slow if we put item one by one.
//  def set(i: Int, ts: Array[T]): Unit = {
//    assert(!deleted)
//    val curOffSet = indexer(i)
//    assert(curOffSet != -1, s"Invalid index: ${i}")
//    var j = 0
//    while (j < ts.length) {
//      putSingle(curOffSet + (j << moveStep), ts(j))
//      j += 1
//    }
//    indexer(i + 1) = curOffSet + (ts.length << moveStep)
//  }
//
//  def putSingle(offset: Long, s: T): Unit
//
//  def indexOf(i: Int): Long = {
//    assert(isValidIndex(i), s"Invalid index: ${i}")
//    return indexer(i)
//  }
// }
