/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.utils;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.intArray;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.longArray;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.floatArray;
import org.apache.log4j.Logger;

public class IndexHelperHNSW {
    private static final Logger log = Logger.getLogger(IndexHelper.class);

    public static String show(longArray a, int rows, int cols) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.append(i).append('\t').append('|');
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%5d ", a.getitem(i * cols + j)));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static String show(floatArray a, int rows, int cols) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.append(i).append('\t').append('|');
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%7g ", a.getitem(i * cols + j)));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static floatArray makeFloatArray(float[][] vectors) {
        int d = vectors[0].length;
        int nb = vectors.length;
        floatArray fa = new floatArray(d * nb);
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < d; j++) {
                fa.setitem(d * i + j, vectors[i][j]);
            }
        }
        return fa;
    }

    public static longArray makeLongArray(int[] ints) {
        int len = ints.length;
        longArray la = new longArray(len);
        for (int i = 0; i < len; i++) {
            la.setitem(i, ints[i]);
        }
        return la;
    }

    public static long[] toArray(longArray c_array, int length) {
        return toArray(c_array, 0, length);
    }

    public static long[] toArray(longArray c_array, int start, int length) {
        long[] re = new long[length];
        for (int i = start; i < length; i++) {
            re[i] = c_array.getitem(i);
        }
        return re;
    }

    public static int[] toArray(intArray c_array, int length) {
        return toArray(c_array, 0, length);
    }

    public static int[] toArray(intArray c_array, int start, int length) {
        int[] re = new int[length];
        for (int i = start; i < length; i++) {
            re[i] = c_array.getitem(i);
        }
        return re;
    }

    public static float[] toArray(floatArray c_array, int length) {
        return toArray(c_array, 0, length);
    }

    public static float[] toArray(floatArray c_array, int start, int length) {
        float[] re = new float[length];
        for (int i = start; i < length; i++) {
            re[i] = c_array.getitem(i);
        }
        return re;
    }
}
