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

package com.intel.analytics.bigdl.ppml.utils

import org.apache.parquet.io.DelegatingSeekableInputStream
import org.apache.parquet.io.InputFile
import org.apache.parquet.io.SeekableInputStream
import java.io.ByteArrayInputStream
import java.io.IOException

class ParquetStream(stream: Array[Byte]) extends InputFile{
    val data: Array[Byte] = stream

    class SeekableByteArrayInputStream(buf: Array[Byte]) extends ByteArrayInputStream(buf) {
        def setPos(pos: Int): Unit = {
            this.pos = pos
        }
        def getPos(): Int = {
            this.pos
        }
    }

    @throws[IOException]
    override def getLength: Long = {
        this.data.length
    }

    @throws[IOException]
    override def newStream(): SeekableInputStream = {
        new DelegatingSeekableInputStream(new SeekableByteArrayInputStream(this.data)) {
            @throws[IOException]
            override def seek(newPos: Long): Unit = {
                this.getStream.asInstanceOf[SeekableByteArrayInputStream].setPos(newPos.toInt)
            }

            @throws[IOException]
            override def getPos(): Long = {
                this.getStream.asInstanceOf[SeekableByteArrayInputStream].getPos
            }
        }
    }
}
