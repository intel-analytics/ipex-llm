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

import com.intel.analytics.bigdl.tensor.Tensor
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import org.scalatest.{FlatSpec, Matchers}

class SerializableIndexSeqSpec extends FlatSpec with Matchers {

  private def serializeAndCheck(s2: SerializableIndexedSeq[Int], cmp: Seq[Int]): Unit = {
    val buffered = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(buffered)
    oos.writeObject(s2)
    val buf = buffered.toByteArray
    val binputStream = new ByteArrayInputStream(buf)
    val ois = new ObjectInputStream(binputStream)
    val t2 = ois.readObject().asInstanceOf[IndexedSeq[Int]]
    t2 should be(cmp)
  }

  "SerializableIndexSeqSpec" should "work well" in {
    val t = (0 to 100).toArray
    val s: SerializableIndexedSeq[Int] = t.view(1, 20)

    s should be (1 until 20)
    serializeAndCheck(s, 1 until 20)

    val s2: SerializableIndexedSeq[Int] = s.view(15, 19)
    s2 should be (16 until 20)
    serializeAndCheck(s2, 16 until 20)

    val s3: SerializableIndexedSeq[Int] = s2.view(1, 3)
    s3 should be (17 until 19)
    serializeAndCheck(s3, 17 until 19)
  }


  "SerializableIndexSeqSpec" should "work well for write after read" in {

    def writeAndRead(s2: IndexedSeq[String]): IndexedSeq[String] = {
      val buffered = new ByteArrayOutputStream()
      val oos = new ObjectOutputStream(buffered)
      oos.writeObject(s2)
      val buf = buffered.toByteArray
      val binputStream = new ByteArrayInputStream(buf)
      val ois = new ObjectInputStream(binputStream)
      val ret = ois.readObject().asInstanceOf[IndexedSeq[String]]
      oos.close()
      ois.close()
      ret
    }
    val s2: SerializableIndexedSeq[String] =
      Array("1", "2", "3", "4").view
    val t2 = writeAndRead(s2)
    writeAndRead(t2)

  }

}
