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

  def getLocalBytes(blockId: BlockId): Option[ByteBuffer] = {
    getLocalBytesFn(blockId)
  }

  def getLocalOrRemoteBytes(blockId: BlockId): Option[ByteBuffer] = {
    val maybeLocalBytes = getLocalBytesFn(blockId)
    if (maybeLocalBytes.isDefined) {
      maybeLocalBytes
    } else {
      SparkEnv.get.blockManager.getRemoteBytes(blockId).map(_.toByteBuffer)
    }
  }

  def unlock(blockId : BlockId): Unit = {
    val blockInfoManager = SparkEnv.get.blockManager.blockInfoManager
    if (blockInfoManager.get(blockId).isDefined) {
      unlockFn(blockId)
    }
  }

  private val getLocalBytesFn: (BlockId) => Option[ByteBuffer] = {
    val bmClass = classOf[BlockManager]
    val getLocalBytesMethod = bmClass.getMethod("getLocalBytes", classOf[BlockId])

    // Spark versions before 2.2.0 declare:
    // def getLocalBytes(blockId: BlockId): Option[ChunkedByteBuffer]
    // Spark 2.2.0+ declares:
    // def getLocalBytes(blockId: BlockId): Option[BlockData]
    // Because the latter change happened in the commit that introduced BlockData,
    // and because you can't discover the generic type of the return type by reflection,
    // distinguish the cases by seeing if BlockData exists.
    try {
      val blockDataClass = Class.forName("org.apache.spark.storage.BlockData")
      // newer method, apply reflection to transform BlockData after invoking
      val toByteBufferMethod = blockDataClass.getMethod("toByteBuffer")
      (blockId: BlockId) =>
        getLocalBytesMethod.invoke(SparkEnv.get.blockManager, blockId)
          .asInstanceOf[Option[_]]
          .map(blockData => toByteBufferMethod.invoke(blockData).asInstanceOf[ByteBuffer])
    } catch {
      case _: ClassNotFoundException =>
        // older method, can be invoked directly
        (blockId: BlockId) =>
          getLocalBytesMethod.invoke(SparkEnv.get.blockManager, blockId)
            .asInstanceOf[Option[ChunkedByteBuffer]]
            .map(_.toByteBuffer)
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

  private val unlockFn: (BlockId) => Unit = {
    val bimClass = classOf[BlockInfoManager]
    // Spark 2.0.0-2.0.2, 2.1.0-2.1.1 declare:
    // def unlock(blockId: BlockId): Unit
    val unlockMethod =
      try {
        bimClass.getMethod("unlock", classOf[BlockId])
      } catch {
        case _: NoSuchMethodException =>
          // But 2.0.3+, 2.1.2+, 2.2.0+ declare:
          // def unlock(blockId: BlockId, taskAttemptId: Option[TaskAttemptId] = None): Unit
          bimClass.getMethod("unlock", classOf[BlockId], classOf[Option[_]])
      }
    unlockMethod.getParameterTypes.length match {
      case 1 =>
        (blockId: BlockId) =>
          unlockMethod.invoke(SparkEnv.get.blockManager.blockInfoManager, blockId)
      case 2 =>
        (blockId: BlockId) =>
          unlockMethod.invoke(SparkEnv.get.blockManager.blockInfoManager, blockId, None)
    }
  }

}
