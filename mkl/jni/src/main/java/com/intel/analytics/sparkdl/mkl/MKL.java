package com.intel.analytics.sparkdl.mkl;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * MKL Library Wrapper for JVM
 */
public class MKL {
    private static boolean isLoaded = false;
    private static File tmpFile = null;

    static {
        isLoaded = true;
        try {
            tmpFile = extract("libjmkl.so");
            System.load(tmpFile.getAbsolutePath());
        } catch (Throwable e) {
            isLoaded = false;
        }
    }

    /**
     * Check if MKL is loaded
     * @return
     */
    public static boolean isMKLLoaded() {
        return isLoaded;
    }

    /**
     * Get the temp path of the .so file
     * @return
     */
    public static String getTmpSoFilePath() {
        if(tmpFile == null)
            return "";
        else
            return tmpFile.getAbsolutePath();
    }

    /**
     * Set MKL worker pool size for current JVM thread. Note different JVM thread has separated MKL worker pool.
     * @param numThreads
     */
    public native static void setNumThreads(int numThreads);

    public native static void vsAdd(int n, float[] a, int aOffset, float[] b, int bOffset,
                                    float[] y, int yOffset);

    public native static void vdAdd(int n, double[] a, int aOffset, double[] b, int bOffset,
                                    double[] y, int yOffset);

    public native static void vsSub(int n, float[] a, int aOffset, float[] b, int bOffset,
                                    float[] y, int yOffset);

    public native static void vdSub(int n, double[] a, int aOffset, double[] b, int bOffset,
                                    double[] y, int yOffset);

    public native static void vsMul(int n, float[] a, int aOffset, float[] b, int bOffset,
                                    float[] y, int yOffset);

    public native static void vdMul(int n, double[] a, int aOffset, double[] b, int bOffset,
                                    double[] y, int yOffset);

    public native static void vsDiv(int n, float[] a, int aOffset, float[] b, int bOffset,
                                    float[] y, int yOffset);

    public native static void vdDiv(int n, double[] a, int aOffset, double[] b, int bOffset,
                                    double[] y, int yOffset);

    public native static void vsPowx(int n, float[] a, int aOffset, float b, float[] y, int yOffset);

    public native static void vdPowx(int n, double[] a, int aOffset, double b, double[] y, int yOffset);

    public native static void vsLn(int n, float[] a, int aOffset, float[] y, int yOffset);

    public native static void vdLn(int n, double[] a, int aOffset, double[] y, int yOffset);

    public native static void vsExp(int n, float[] a, int aOffset, float[] y, int yOffset);

    public native static void vdExp(int n, double[] a, int aOffset, double[] y, int yOffset);

    public native static void vsSqrt(int n, float[] a, int aOffset, float[] y, int yOffset);

    public native static void vdSqrt(int n, double[] a, int aOffset, double[] y, int yOffset);

    public native static void vsLog1p(int n, float[] a, int aOffset, float[] y, int yOffset);

    public native static void vdLog1p(int n, double[] a, int aOffset, double[] y, int yOffset);

    /**
     * Get the worker pool size of current JVM thread. Note different JVM thread has separated MKL worker pool.
     * @return
     */
    public native static int getNumThreads();

    // Extract so file from jar to a temp path
    private static File extract(String path) {
        try {
            URL url = MKL.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find so file in jar, path = " + path);
            }

            InputStream in = MKL.class.getResourceAsStream("/" + path);
            File file = file(path);

            ReadableByteChannel src = newChannel(in);
            FileChannel dest = new FileOutputStream(file).getChannel();
            dest.transferFrom(src, 0, Long.MAX_VALUE);
            return file;
        } catch (Throwable e) {
            throw new Error("Can't extract so file");
        }
    }

    private static File file(String path) throws IOException {
        String name = new File(path).getName();
        return createTempFile("jniloader", name);
    }
}
