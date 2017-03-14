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

package org.apache.spark.storage

import java.nio.ByteBuffer

import org.apache.spark.SparkEnv
import org.apache.spark.util.io.ChunkedByteBuffer

object BlockManagerWrapper {

  def putBytes( blockId: BlockId,
                bytes: ByteBuffer,
                level: StorageLevel): Unit = {
    require(bytes != null, "Bytes is null")
    val blockManager = SparkEnv.get.blockManager
    blockManager.removeBlock(blockId)
    blockManager.putBytes(blockId, new ChunkedByteBuffer(bytes), level)
  }

  def getLocal(blockId: BlockId): Option[BlockResult] = {
    SparkEnv.get.blockManager.getLocalValues(blockId)
  }

  def putSingle(blockId: BlockId,
    value: Any,
    level: StorageLevel,
    tellMaster: Boolean = true): Unit = {
    SparkEnv.get.blockManager.putSingle(blockId, value, level, tellMaster)
  }

  def removeBlock(blockId: BlockId): Unit = {
    SparkEnv.get.blockManager.removeBlock(blockId)
  }

  def byteBufferConvert(chunkedByteBuffer: ChunkedByteBuffer): ByteBuffer = {
    ByteBuffer.wrap(chunkedByteBuffer.toArray)
  }

  def unlock(blockId : BlockId): Unit = {
    val blockInfoManager = SparkEnv.get.blockManager.blockInfoManager
    if(blockInfoManager.get(blockId).isDefined) {
      blockInfoManager.unlock(blockId)
    }
  }
}
