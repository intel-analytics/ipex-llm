package com.intel.analytics.bigdl.ppml.utils
import java.io.IOException
import java.net.{DatagramSocket, ServerSocket}

object PortUtils {
  def findNextPortAvailable(port: Int): Int = {
    def isAvailable(port: Int): Boolean = {
      var ss: ServerSocket = null
      var ds: DatagramSocket = null
      try {
        ss = new ServerSocket(port)
        ds = new DatagramSocket(port)
        true
      }
        catch {
          case e: IOException =>
            false
        } finally {
          if (ss != null) ss.close()
          if (ds != null) ds.close()
        }
      }

    var portAvailable: Int = port
    while (!isAvailable(portAvailable)) {
      portAvailable += 1
    }
    portAvailable
  }
}
