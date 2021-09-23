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
// class VarLenBytesArray(recordNum: Int, totalSizeByBytes: Long,
//    memoryType: MemoryType = PMEM) extends NativeVarLenArray[Byte](recordNum,
//  totalSizeByBytes, memoryType, 0) {
//
//  def putSingle(offset: Long, s: Byte): Unit = {
//    Platform.putByte(null, offset, s.asInstanceOf[Byte])
//  }
//
//  override def getTypeOffSet(): Int = Platform.BYTE_ARRAY_OFFSET
// }
//
// class FixLenBytesArray(val numOfRecord: Long, val sizeOfRecordByByte: Int,
//    memoryType: MemoryType = PMEM) extends
//  NativeArray[Array[Byte]](numOfRecord * sizeOfRecordByByte, memoryType) {
//
//  override def get(i: Int): Array[Byte] = {
//    val result = new Array[Byte](sizeOfRecordByByte)
//    Platform.copyMemory(null, indexOf(i), result, Platform.BYTE_ARRAY_OFFSET, sizeOfRecordByByte)
//    return result
//  }
//
//  // TODO: would be slow if we put byte one by one.
//  def set(i: Int, bytes: Array[Byte]): Unit = {
//    assert(!deleted)
//    val startOffset = indexOf(i)
//    var j = 0
//    while(j < bytes.length) {
//      Platform.putByte(null, startOffset + j, bytes(j))
//      j += 1
//    }
//  }
//
//  def indexOf(i: Int): Long = {
//    val index = startAddr + (i * sizeOfRecordByByte)
//    assert(index <= lastOffSet)
//    index
//  }
// }
//
