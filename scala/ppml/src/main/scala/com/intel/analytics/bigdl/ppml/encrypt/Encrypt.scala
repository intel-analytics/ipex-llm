package com.intel.analytics.bigdl.ppml.encrypt

import com.intel.analytics.bigdl.ppml.utils.Supportive

trait Encrypt extends Supportive with Serializable {
  def encryptFile(sourceFilePath:String, saveFilePath:String, dataKeyPlaintext:String)
  def decryptFile(sourceFilePath:String, saveFilePath:String, dataKeyPlaintext:String)
  def encryptBytes(sourceBytes:Array[Byte], dataKeyPlaintext:String): Array[Byte]
  def decryptBytes(sourceBytes:Array[Byte], dataKeyPlaintext:String): Array[Byte]
}

object EncryptMode extends Enumeration {
  type EncryptMode = Value
  val PLAIN_TEXT = Value("plain_text", "plain_text")
  val AES_CBC_PKCS5PADDING = Value("AES/CBC/PKCS5Padding", "AES/CBC/PKCS5Padding")
  val UNKNOWN = Value("UNKNOWN", "UNKNOWN")
  class EncryptModeEnumVal(name: String, val value: String) extends Val(nextId, name)
  protected final def Value(name: String, value: String): EncryptModeEnumVal = new EncryptModeEnumVal(name, value)
  def parse(s: String) = values.find(_.toString.toLowerCase() == s.toLowerCase).getOrElse(EncryptMode.UNKNOWN)
}
