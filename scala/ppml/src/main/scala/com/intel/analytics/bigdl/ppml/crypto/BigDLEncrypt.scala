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

package com.intel.analytics.bigdl.ppml.crypto

import com.intel.analytics.bigdl.dllib.utils.{File, Log4Error}
import com.intel.analytics.bigdl.ppml.utils.ParquetStream
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode
import org.apache.hadoop.fs.Path

import java.io._
import java.security.SecureRandom
import java.time.Instant
import java.util.Arrays
import javax.crypto.spec.{IvParameterSpec, SecretKeySpec}
import javax.crypto.{Cipher, Mac}
import org.apache.spark.input.PortableDataStream
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.example.data.simple.convert.GroupRecordConverter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.io.ColumnIOFactory

import java.nio.ByteBuffer
import scala.util.Random

/**
 * BigDL general crypto for encrypt and decrypt data.
 */
class BigDLEncrypt extends Crypto {
  protected var cipher: Cipher = null
  protected var mac: Mac = null
  protected var ivParameterSpec: IvParameterSpec = null
  protected var encryptionKeySpec: SecretKeySpec = null
  protected var opMode: OperationMode = null
  protected var initializationVector: Array[Byte] = null

  /**
   * Init this crypto with crypto mode, operation mode and keys.
   * @param cryptoMode cryptoMode to en/decrypt data, such as AES_CBC_PKCS5PADDING.
   * @param mode en/decrypt mode, one of ENCRYPT or DECRYPT.
   * @param dataKeyPlaintext signing key and data key.
   */
  override def init(cryptoMode: CryptoMode, mode: OperationMode, dataKeyPlaintext: String): Unit = {
    opMode = mode
    val secret = dataKeyPlaintext.getBytes()
    // key encrypt
    val signingKey = Arrays.copyOfRange(secret, 0, 16)
    val encryptKey = Arrays.copyOfRange(secret, 16, 32)
//    initializationVector = Arrays.copyOfRange(secret, 0, 16)
    val r = new Random(signingKey.sum)
    initializationVector = Array.tabulate(16)(_ => (r.nextInt(256) - 128).toByte)
    ivParameterSpec = new IvParameterSpec(initializationVector)
    encryptionKeySpec = new SecretKeySpec(encryptKey, cryptoMode.secretKeyAlgorithm)
    cipher = Cipher.getInstance(cryptoMode.encryptionAlgorithm)
    cipher.init(mode.opmode, encryptionKeySpec, ivParameterSpec)
    mac = Mac.getInstance(cryptoMode.signingAlgorithm)
    val signingKeySpec = new SecretKeySpec(signingKey, cryptoMode.signingAlgorithm)
    mac.init(signingKeySpec)
  }

  protected var signingDataStream: DataOutputStream = null

  /**
   * If encrypt data, should generate header and put return value to the head.
   * @return header bytes
   */
  override def genHeader(): Array[Byte] = {
    Log4Error.invalidOperationError(cipher != null,
      s"you should init BigDLEncrypt first.")
    val timestamp: Instant = Instant.now()
    val signingByteBuffer = ByteBuffer.allocate(1 + 8 + ivParameterSpec.getIV.length)
    val version: Byte = (0x80).toByte
    signingByteBuffer.put(version)
    signingByteBuffer.putLong(timestamp.getEpochSecond())
    signingByteBuffer.put(ivParameterSpec.getIV())
    signingByteBuffer.array()
  }

  /**
   * Verify the header bytes when decrypt.
   * @param header header bytes
   */
  override def verifyHeader(header: Array[Byte]): Unit = {
    val headerBuffer = ByteBuffer.wrap(header)
    val version: Byte = headerBuffer.get()
    Log4Error.invalidInputError(version.compare((0x80).toByte) == 0,
      "File header version error!")
    val timestampSeconds: Long = headerBuffer.getLong
    val initializationVector: Array[Byte] = header.slice(1 + 8, header.length)
    if (!initializationVector.sameElements(this.initializationVector)) {
      ivParameterSpec = new IvParameterSpec(initializationVector)
      cipher.init(opMode.opmode, encryptionKeySpec, ivParameterSpec)
    }
  }

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * @param content byte to be encrypted or decrypted.
   * @return encrypted or decrypted bytes.
   */
  override def update(content: Array[Byte]): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content)
    mac.update(cipherText)
    cipherText
  }

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return encrypted or decrypted bytes.
   */
  override def update(content: Array[Byte], offset: Int, len: Int): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content, offset, len)
    mac.update(cipherText, offset, len)
    cipherText
  }

  /**
   * Encrypts or decrypts data in a single-part operation,
   * or finishes a multiple-part operation. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param content bytes to be encrypted or decrypted.
   * @return (encrypted or decrypted bytes, Message Authentication Code)
   */
  override def doFinal(content: Array[Byte]): (Array[Byte], Array[Byte]) = {
    val cipherText: Array[Byte] = cipher.doFinal(content)
    val hmac: Array[Byte] = mac.doFinal(cipherText)
    (cipherText, hmac)
  }

  /**
   * Encrypts or decrypts data in a single-part operation,
   * or finishes a multiple-part operation. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return (encrypted or decrypted bytes, Message Authentication Code)
   */
  override def doFinal(content: Array[Byte], offset: Int, len: Int): (Array[Byte], Array[Byte]) = {
    val cipherText: Array[Byte] = cipher.doFinal(content, offset, len)
    val hmac: Array[Byte] = mac.doFinal(cipherText.slice(offset, offset + len))
    (cipherText, hmac)
  }

  /**
   * Encrypts or decrypts a byte stream. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param inputStream input stream
   * @param outputStream output stream
   */
  def doFinal(inputStream: DataInputStream, outputStream: DataOutputStream): Unit = {
    if (opMode == ENCRYPT) {
      encryptStream(inputStream, outputStream)
    } else {
      decryptStream(inputStream, outputStream)
    }
  }

  /**
   * Encrypts or decrypts a file. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param binaryFilePath
   * @param savePath
   */
  def doFinal(binaryFilePath: String, savePath: String): Unit = {
    if (opMode == ENCRYPT) {
      encryptFile(binaryFilePath, savePath)
    } else {
      decryptFile(binaryFilePath, savePath)
    }
  }

  val blockSize = 1024 * 1024 // 1m per update
  val byteBuffer = new Array[Byte](blockSize)
  protected def encryptStream(
        inputStream: DataInputStream,
        outputStream: DataOutputStream): Unit = {
    val header = genHeader()
    outputStream.write(header)
    while (inputStream.available() > blockSize) {
      val readLen = inputStream.read(byteBuffer)
      outputStream.write(update(byteBuffer, 0, readLen))
    }
    val last = inputStream.read(byteBuffer)
    val (lastSlice, hmac) = doFinal(byteBuffer, 0, last)
    outputStream.write(lastSlice)
    outputStream.write(hmac)
    outputStream.flush()
  }

  val hmacSize = 32
  protected def decryptStream(
        inputStream: DataInputStream,
        outputStream: DataOutputStream): Unit = {
    val header = read(inputStream, 25)
    verifyHeader(header)
    while (inputStream.available() > blockSize) {
      val readLen = inputStream.read(byteBuffer)
      outputStream.write(update(byteBuffer, 0, readLen))
    }
    val last = inputStream.read(byteBuffer)
    val inputHmac = byteBuffer.slice(last - hmacSize, last)
    val (lastSlice, streamHmac) = doFinal(byteBuffer, 0, last - hmacSize)
    Log4Error.invalidInputError(!inputHmac.sameElements(streamHmac), "hmac not match")
    outputStream.write(lastSlice)
    outputStream.flush()
  }

  protected def decryptFile(binaryFilePath: String, savePath: String): Unit = {
    Log4Error.invalidInputError(savePath != null && savePath != "",
      "decrypted file save path should be specified")
    val fs = File.getFileSystem(binaryFilePath)
    val bis = fs.open(new Path(binaryFilePath))
    val outs = fs.create(new Path(savePath))
    encryptStream(bis, outs)
    bis.close()
    outs.close()
  }

  protected def encryptFile(binaryFilePath: String, savePath: String): Unit = {
    Log4Error.invalidInputError(savePath != null && savePath != "",
      "decrypted file save path should be specified")
    val fs = File.getFileSystem(binaryFilePath)
    val bis = fs.open(new Path(binaryFilePath))
    val outs = fs.create(new Path(savePath))
    decryptStream(bis, outs)
    bis.close()
    outs.close()
  }

  private def read(stream: DataInputStream, numBytes: Int): Array[Byte] = {
    val retval = new Array[Byte](numBytes)
    val bytesRead: Int = stream.read(retval)
    Log4Error.invalidOperationError(bytesRead == numBytes,
      s"Not enough bits to read!, excepted $numBytes, but got $bytesRead.")
    retval
  }

  /**
   * decrypt big data stream.
   * @param ite stream iterator.
   * @return iterator of String.
   */
  override def decryptBigContent(
        ite: Iterator[(String, PortableDataStream)]): Iterator[String] = {
    var result: Iterator[String] = Iterator[String]()

    while (ite.hasNext == true) {
      val inputStream: DataInputStream = ite.next._2.open()
      verifyHeader(read(inputStream, 25))

      // do first
      var lastString = ""
      while (inputStream.available() > blockSize) {
        val readLen = inputStream.read(byteBuffer)
        Log4Error.unKnowExceptionError(readLen != blockSize)
        val currentSplitDecryptString = new String(byteBuffer, 0, readLen)
        val splitDecryptString = lastString + currentSplitDecryptString
        val splitDecryptStringArray = splitDecryptString.split("\r").flatMap(_.split("\n"))
        lastString = splitDecryptStringArray.last
        result = result ++ splitDecryptStringArray.dropRight(1)
      }
      // do last
      val last = inputStream.read(byteBuffer)
      val inputHmac = byteBuffer.slice(last - hmacSize, last)
      val (lastSlice, streamHmac) = doFinal(byteBuffer, 0, last - hmacSize)
      Log4Error.invalidInputError(!inputHmac.sameElements(streamHmac),
        "hmac not match")
      val lastDecryptString = lastString + new String(lastSlice)
      val splitDecryptStringArray = lastDecryptString.split("\r").flatMap(_.split("\n"))
      result = result ++ splitDecryptStringArray
    }
    result

  }

  /**
   * read parquet from byte array.
   * @param content plaintext of data stream.
   * @return iterator of String.
   */
  override def readParquet(content: Array[Byte]): Iterator[String] = {
    var result: Iterator[String] = Iterator[String]()
    val parquetStream = new ParquetStream(content)
    val reader = ParquetFileReader.open(parquetStream)
    val metaData = reader.getFooter.getFileMetaData.getSchema
    val fieldSize = metaData.getFields.size()
    val headArray: Array[String] = new Array[String](1)
    var headString = ""
    val fields = metaData.getFields
    for(fieldIndex <- 0 until fieldSize){
      headString += fields.get(fieldIndex).getName
      if (fieldIndex != fieldSize - 1){
        headString += ","
      }
    }
    headArray(0) = headString  // store the first line(fields) in a string array
    result = result ++ headArray  // store the first line in the result iterator
    var rowGroup: PageReadStore = reader.readNextRowGroup()
    // read per row group
    while (rowGroup != null) {
      val rowSize = rowGroup.getRowCount.toInt
      val rowArray: Array[String] = new Array[String](rowSize)
      val columnIO = new ColumnIOFactory().getColumnIO(metaData)
      val recordReader = columnIO.getRecordReader(rowGroup, new GroupRecordConverter(metaData))
      // get per row in this row group
      for (i <- 0 until rowSize){
        val simpleGroup = recordReader.read()
        var row = ""
        for (j <- 0 until fieldSize){
          row += simpleGroup.getValueToString(j, 0)
          if (j != fieldSize - 1){
            row += ","
          }
        }
        rowArray(i) = row
      }
      result = result ++ rowArray
      rowGroup = reader.readNextRowGroup()
    }
    result

  }

  /**
   * decrypt big parquet data stream.
   * @param ite stream iterator.
   * @return iterator of String.
   */
  override def decryptParquetContent(ite: Iterator[(String, PortableDataStream)]): Iterator[String] = {
    var result: Iterator[String] = Iterator[String]()

    while (ite.hasNext) {
      val inputStream: DataInputStream = ite.next._2.open()
      verifyHeader(read(inputStream, 25))

      var content = ""
      while (inputStream.available() > blockSize) {
        val readLen = inputStream.read(byteBuffer)
        Log4Error.unKnowExceptionError(readLen != blockSize)
        val currentSplitDecryptString = new String(byteBuffer, 0, readLen)
        content += currentSplitDecryptString
      }

      val last = inputStream.read(byteBuffer)
      val inputHmac = byteBuffer.slice(last - hmacSize, last)
      val (lastSlice, streamHmac) = doFinal(byteBuffer, 0, last - hmacSize)
      Log4Error.invalidInputError(!inputHmac.sameElements(streamHmac),
        "hmac not match")
      val lastCipherText: Array[Byte] = read(inputStream, inputStream.available() - 32)
      content += new String(lastSlice)
      result = result ++ readParquet(content.getBytes())
    }
    result

  }

}
