package com.intel.analytics.bigdl.ppml.utils

import java.nio.charset.StandardCharsets
import java.util.Base64
import java.util
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.crypto.KeyAccessDeniedException
import org.apache.parquet.crypto.ParquetCryptoRuntimeException
import org.apache.parquet.crypto.keytools.KeyToolkit
import org.apache.parquet.crypto.keytools.KmsClient
import org.slf4j.LoggerFactory


object InMemoryKMS {
    val LOG = LoggerFactory.getLogger(classOf[InMemoryKMS])
    val KEY_LIST_PROPERTY_NAME = "parquet.encryption.key.list"
    val NEW_KEY_LIST_PROPERTY_NAME = "parquet.encryption.new.key.list"
    var masterKeyMap: util.HashMap[String, Array[Byte]] = null
    var newMasterKeyMap: util.HashMap[String, Array[Byte]] = null

    def startKeyRotation(hadoopConfiguration: Configuration): Unit = {
        val newMasterKeys = hadoopConfiguration.getTrimmedStrings(NEW_KEY_LIST_PROPERTY_NAME)
        if (null == newMasterKeys || newMasterKeys.length == 0) throw new ParquetCryptoRuntimeException("No encryption key list")
        newMasterKeyMap = parseKeyList(newMasterKeys)
    }

    def finishKeyRotation(): Unit = {
        masterKeyMap = newMasterKeyMap
    }

    private def parseKeyList(masterKeys: Array[String]) = {
        val keyMap = new util.HashMap[String, Array[Byte]]
        val nKeys = masterKeys.length
        for (i <- 0 until nKeys) {
            val parts = masterKeys(i).split(":")
            val keyName = parts(0).trim
            if (parts.length != 2) throw new IllegalArgumentException("Key '" + keyName + "' is not formatted correctly")
            val key = parts(1).trim
            try {
                val keyBytes = Base64.getDecoder.decode(key)
                keyMap.put(keyName, keyBytes)
            } catch {
                case e: IllegalArgumentException =>
                    LOG.warn("Could not decode key '" + keyName + "'!")
                    throw e
            }
        }
        keyMap
    }
}

class InMemoryKMS extends KmsClient {
    override def initialize(configuration: Configuration, kmsInstanceID: String, kmsInstanceURL: String, accessToken: String): Unit = { // Parse master  keys
        val masterKeys = configuration.getTrimmedStrings(InMemoryKMS.KEY_LIST_PROPERTY_NAME)
        if (null == masterKeys || masterKeys.length == 0) throw new ParquetCryptoRuntimeException("No encryption key list")
        InMemoryKMS.masterKeyMap = InMemoryKMS.parseKeyList(masterKeys)
        InMemoryKMS.newMasterKeyMap = InMemoryKMS.masterKeyMap
    }

    @throws[KeyAccessDeniedException]
    @throws[UnsupportedOperationException]
    override def wrapKey(keyBytes: Array[Byte], masterKeyIdentifier: String): String = { // Always use the latest key version for writing
        val masterKey = InMemoryKMS.newMasterKeyMap.get(masterKeyIdentifier)
        if (null == masterKey) throw new ParquetCryptoRuntimeException("Key not found: " + masterKeyIdentifier)
        val AAD = masterKeyIdentifier.getBytes(StandardCharsets.UTF_8)
        KeyToolkit.encryptKeyLocally(keyBytes, masterKey, AAD)
    }

    @throws[KeyAccessDeniedException]
    @throws[UnsupportedOperationException]
    override def unwrapKey(wrappedKey: String, masterKeyIdentifier: String): Array[Byte] = {
        val masterKey = InMemoryKMS.masterKeyMap.get(masterKeyIdentifier)
        if (null == masterKey) throw new ParquetCryptoRuntimeException("Key not found: " + masterKeyIdentifier)
        val AAD = masterKeyIdentifier.getBytes(StandardCharsets.UTF_8)
        KeyToolkit.decryptKeyLocally(wrappedKey, masterKey, AAD)
    }
}

