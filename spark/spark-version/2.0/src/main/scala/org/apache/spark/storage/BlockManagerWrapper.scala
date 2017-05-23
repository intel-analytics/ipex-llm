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

import java.lang.{Boolean => JBoolean}
import java.nio.ByteBuffer

import org.apache.spark.SparkEnv
import org.apache.spark.util.io.ChunkedByteBuffer

import scala.reflect.ClassTag

object BlockManagerWrapper {

  def putBytes( blockId: BlockId,
                bytes: ByteBuffer,
                level: StorageLevel): Unit = {
    require(bytes != null, "Bytes is null")
    val blockManager = SparkEnv.get.blockManager
    blockManager.removeBlock(blockId)
    putBytesFn(blockId, new ChunkedByteBuffer(bytes), level)
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

  private val putBytesFn: (BlockId, ChunkedByteBuffer, StorageLevel) => Unit = {
    val bmClass = classOf[BlockManager]
    // Spark 2.0.0 - 2.1.0, and 2.2.0+ (as of this writing), declare the method:
    // def putBytes[T: ClassTag](
    //   blockId: BlockId,
    //   bytes: ChunkedByteBuffer,
    //   level: StorageLevel,
    //   tellMaster: Boolean = true): Boolean
    val putBytesMethod =
      try {
        bmClass.getMethod("putBytes",
          classOf[BlockId], classOf[ChunkedByteBuffer], classOf[StorageLevel],
          classOf[Boolean], classOf[ClassTag[_]])
      } catch {
        case _: NoSuchMethodException =>
          // But Spark 2.1.1 and distros like Cloudera 2.0.0 / 2.1.0 had an extra boolean
          // param:
          //   def putBytes[T: ClassTag](
          //     blockId: BlockId,
          //     bytes: ChunkedByteBuffer,
          //     level: StorageLevel,
          //     tellMaster: Boolean = true,
          //     encrypt: Boolean = false): Boolean
          bmClass.getMethod("putBytes",
            classOf[BlockId], classOf[ChunkedByteBuffer], classOf[StorageLevel],
            classOf[Boolean], classOf[Boolean], classOf[ClassTag[_]])
      }
    putBytesMethod.getParameterTypes.length match {
      case 5 =>
        (blockId: BlockId, bytes: ChunkedByteBuffer, level: StorageLevel) =>
          putBytesMethod.invoke(SparkEnv.get.blockManager,
            blockId, bytes, level, JBoolean.TRUE, null)
      case 6 =>
        (blockId: BlockId, bytes: ChunkedByteBuffer, level: StorageLevel) =>
          putBytesMethod.invoke(SparkEnv.get.blockManager,
            blockId, bytes, level, JBoolean.TRUE, JBoolean.FALSE, null)
    }
  }

}
