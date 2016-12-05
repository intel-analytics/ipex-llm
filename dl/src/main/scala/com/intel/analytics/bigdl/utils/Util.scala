package com.intel.analytics.bigdl.utils

object Util {
  def kthLargest(arr: Array[Long], l: Int, r: Int, k: Int): Long = {
    val pos = randomPartition(arr, l, r)
    if (pos-l == k-1)  return arr(pos)

    if (pos-l > k-1) return kthLargest(arr, l, pos-1, k)

    kthLargest(arr, pos + 1, r, k - pos + l - 1)
  }

  def swap(arr: Array[Long], i: Int, j: Int): Unit = {
    val temp = arr(i)
    arr(i) = arr(j)
    arr(j) = temp
  }

  private def partition(arr: Array[Long], l: Int, r: Int): Int = {
    val x = arr(r)
    var i = l
    for (j <- l to (r - 1)) {
      if (arr(j) > x) {
        swap(arr, i, j);
        i += 1
      }
    }
    swap(arr, i, r);
    i
  }

  private def randomPartition(arr: Array[Long], l: Int, r: Int): Int = {
    val n = r - l + 1;
    val pivot = ((Math.random()) % n).toInt;
    swap(arr, l + pivot, r);
    partition(arr, l, r);
  }
}