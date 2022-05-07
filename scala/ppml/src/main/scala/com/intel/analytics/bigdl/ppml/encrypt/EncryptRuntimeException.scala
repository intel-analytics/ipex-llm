package com.intel.analytics.bigdl.ppml.encrypt

class EncryptRuntimeException(message: String, cause: Throwable)
  extends RuntimeException(message) {
  if (cause != null) {
    initCause(cause)
  }
  def this(message: String) = this(message, null)
}
