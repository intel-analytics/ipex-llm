package com.intel.analytics.bigdl.ppml.utils
import java.io.PrintWriter
import scala.io.Source

class KeyReaderWriter {

  def writeKeyToFile(encryptedKeyPath: String, encryptedKeyContent: String) = {
    new PrintWriter(encryptedKeyPath) { write(encryptedKeyContent); close }
  }

  def readKeyFromFile(encryptedKeyPath: String):String = {
    val encryptedKeyCiphertext:String = Source.fromFile(encryptedKeyPath).getLines.next()
    encryptedKeyCiphertext
  }

}

