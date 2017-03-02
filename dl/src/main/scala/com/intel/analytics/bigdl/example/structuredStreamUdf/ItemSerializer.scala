
package com.intel.analytics.bigdl.example.udf

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import kafka.serializer.{Decoder, Encoder}
import kafka.utils.VerifiableProperties

class ItemEncoder[T](props: VerifiableProperties = null) extends Encoder[T] {

  override def toBytes(t: T): Array[Byte] = {
    if (t == null) {
      null
    }
    else {
      var bo: ByteArrayOutputStream = null
      var oo: ObjectOutputStream = null
      var byte: Array[Byte] = null
      try {
        bo = new ByteArrayOutputStream()
        oo = new ObjectOutputStream(bo)
        oo.writeObject(t)
        byte = bo.toByteArray
      } catch {
        case ex: Exception => return byte
      } finally {
        bo.close()
        oo.close()
      }
      byte
    }
  }
}

class ItemDecoder[T](props: VerifiableProperties = null) extends Decoder[T] {

  def fromBytes(bytes: Array[Byte]): T = {
    var t: T = null.asInstanceOf[T]
    var bi: ByteArrayInputStream = null
    var oi: ObjectInputStream = null
    try {
      bi = new ByteArrayInputStream(bytes)
      oi = new ObjectInputStream(bi)
      t = oi.readObject().asInstanceOf[T]
      t
    } catch {
      case e: Exception => null.asInstanceOf[T]
      //        e.printStackTrace()
      //        null.asInstanceOf[T]

    } finally {
      bi.close()
      oi.close()
    }
  }
}
