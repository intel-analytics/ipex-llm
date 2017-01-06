package com.intel.analytics.bigdl.mkl;

import java.io.*;
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
        try {
            String iomp5FileName = "libiomp5.so";
            String jmklFileName = "libjmkl.so";
            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                iomp5FileName = "libiomp5.dylib";
                jmklFileName = "libjmkl.dylib";
            }
            tmpFile = extract(iomp5FileName);
            System.load(tmpFile.getAbsolutePath());
            tmpFile.delete(); // delete so temp file after loaded
            tmpFile = extract(jmklFileName);
            System.load(tmpFile.getAbsolutePath());
            tmpFile.delete(); // delete so temp file after loaded
            isLoaded = true;

        } catch (Exception e) {
            isLoaded = false;
            e.printStackTrace();
            // TODO: Add an argument for user, continuing to run even if MKL load failed.
            throw new RuntimeException("Failed to load MKL");
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

    public native static void vsAbs(int n, float[] a, int aOffset, float[] y, int yOffset);

    public native static void vdAbs(int n, double[] a, int aOffset, double[] y, int yOffset);

    public native static void vsgemm(char transa, char transb, int m, int n, int k, float alpha,
                                     float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb,
                                     float beta, float[] c,int cOffset, int ldc);

    public native static void vdgemm(char transa, char transb, int m, int n, int k, double alpha,
                                     double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb,
                                     double beta, double[] c,int cOffset, int ldc);

    public native static void vsgemv(char trans, int m, int n, float alpha, float[] a, int aOffset,
                                     int lda, float[] x, int xOffest, int incx, float beta, float[] y,
                                     int yOffest,int incy);

    public native static void vdgemv(char trans, int m, int n, double alpha, double[] a, int aOffset,
                                     int lda, double[] x, int xOffest, int incx, double beta, double[] y,
                                     int yOffest,int incy);

    public native static void vsaxpy(int n, float da, float[] dx, int dxOffest, int incx, float[] dy,
                                     int dyOffset, int incy);

    public native static void vdaxpy(int n, double da, double[] dx, int dxOffest, int incx, double[] dy,
                                     int dyOffset, int incy);

    public native static float vsdot(int n, float[] dx, int dxOffset, int incx, float[]dy, int dyOffset,
                                     int incy);

    public native static double vddot(int n, double[] dx, int dxOffset, int incx, double[]dy, int dyOffset,
                                      int incy);

    public native static void vsger(int m, int n, float alpha, float[] x, int xOffset, int incx,
                                    float[] y, int yOffset, int incy, float[] a, int aOffset, int lda);

    public native static void vdger(int m, int n, double alpha, double[] x, int xOffset, int incx,
                                    double[] y, int yOffset, int incy, double[] a, int aOffset, int lda);

    public native static void vsscal(int n, float sa, float[] sx, int offset, int incx);

    public native static void vdscal(int n, double sa, double[] sx, int offset, int incx);
    
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
            File file = createTempFile("dlNativeLoader", path);

            ReadableByteChannel src = newChannel(in);
            FileChannel dest = new FileOutputStream(file).getChannel();
            dest.transferFrom(src, 0, Long.MAX_VALUE);
            return file;
        } catch (Throwable e) {
            throw new Error("Can't extract so file to /tmp dir");
        }
    }

}
