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

object BlockManagerWrapper {

  def putBytes( blockId: BlockId,
    bytes: ByteBuffer,
    level: StorageLevel): Unit = {
    require(bytes != null, "Bytes is null")
    SparkEnv.get.blockManager.putBytes(blockId, bytes, level)
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

  def getLocal(blockId: BlockId): Option[BlockResult] = {
    SparkEnv.get.blockManager.getLocal(blockId)
  }


  def getLocalBytes(blockId: BlockId): Option[ByteBuffer] = {
    SparkEnv.get.blockManager.getLocalBytes(blockId)
  }

  def getLocalOrRemoteBytes(blockId: BlockId): Option[ByteBuffer] = {
    val bm = SparkEnv.get.blockManager
    val maybeLocalBytes = bm.getLocalBytes(blockId)
    if (maybeLocalBytes.isDefined) {
      maybeLocalBytes
    } else {
      bm.getRemoteBytes(blockId)
    }
  }

  def unlock(blockId : BlockId): Unit = {}
}
