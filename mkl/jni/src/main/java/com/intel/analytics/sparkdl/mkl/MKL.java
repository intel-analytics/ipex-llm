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
            System.out.println(tmpFile.getAbsolutePath());
            System.load(tmpFile.getAbsolutePath());
        } catch (Exception e) {
            System.out.println("Can't load the library" + tmpFile.getAbsolutePath());
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

    /**
     * Get the worker pool size of current JVM thread. Note different JVM thread has separated MKL worker pool.
     * @return
     */
    public native static int getNumThreads();

    // Extract so file from jar to a temp path
    private static File extract(String path) {
        System.out.println(path);
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

    /* Convolution API */
    public native static long ConvolutionInitFloat(
            int inputNumber, int inputChannel, int inputHeight, int inputWidth,
            int kernelNumber, int kernelChannel, int kernelHeight, int kernelWidth,
            int strideHeight, int strideWidth, int padHeight, int padWidth,
            int dimension, int groups, String name);
    public native static void ConvolutionForwardFloat(
            float[] input, int inputOffset, float[] output, int outputOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void ConvolutionBackwardDataFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradInput, int gradInputOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void ConvolutionBackwardKernelFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradKernel, int gradKernelOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void ConvolutionBackwardBiasFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradBias, int gradBiasOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);

    public native static long ConvolutionInitDouble(
            int inputNumber, int inputChannel, int inputHeight, int inputWidth,
            int kernelNumber, int kernelChannel, int kernelHeight, int kernelWidth,
            int strideHeight, int strideWidth, int padHeight, int padWidth,
            int dimension, int groups, String name);
    public native static void ConvolutionForwardDouble(
            double[] input, int inputOffset, double[] output, int outputOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void ConvolutionBackwardDataDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradInput, int gradInputOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void ConvolutionBackwardKernelDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradKernel, int gradKernelOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void ConvolutionBackwardBiasDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradBias, int gradBiasOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);

    /* ReLU API */
    public native static long ReLUInitFloat(
            int inputNumber, int inputChannel, int inputHeight, int inputWidth, int dimension, String name);
    public native static void ReLUForwardFloat(
            float[] input, int inputOffset, float[] output, int outputOffset, long classPtr);
    public native static void ReLUBackwardFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradInput, int gradInputOffset, long classPtr);

    public native static long ReLUInitDouble(
            int inputNumber, int inputChannel, int inputHeight, int inputWidth, int dimension, String name);
    public native static void ReLUForwardDouble(
            double[] input, int inputOffset, double[] output, int outputOffset, long classPtr);
    public native static void ReLUBackwardDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradInput, int gradInputOffset, long classPtr);

    /* Pooling API */
    public native static long PoolingInitFloat(
        int inputNumber, int inputChannel, int inputHeight, int inputWidth,
        int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
        int padHeight, int padWidth, int dimension, int ceilMode,
        int algorithm, String name);
    public native static void PoolingForwardFloat(
        float[] input, int inputOffset, float[] output, int outputOffset,
        long classPtr);
    public native static void PoolingBackwardFloat(
        float[] input, int inputOffset, float[] outputDiff,
        int outputDiffOffset, float[] inputDiff, int inputDiffOffset,
        long classPtr);

    public native static long PoolingInitDouble(
        int inputNumber, int inputChannel, int inputHeight, int inputWidth,
        int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
        int padHeight, int padWidth, int dimension, int ceilMode,
        int algorithm, String name);
    public native static void PoolingForwardDouble(
        double[] input, int inputOffset, double[] output, int outputOffset,
        long classPtr);
    public native static void PoolingBackwardDouble(
        double[] input, int inputOffset, double[] outputDiff,
        int outputDiffOffset, double[] inputDiff, int inputDiffOffset,
        long classPtr);

    /* Batch Normalization */
    public native static long BatchNormInitFloat(
            int inputNumber, int inputChannel, int inputHeight, int inputWidth,
            double eps, int useKernel, int useBias,
            int dimension, String name);
    public native static void BatchNormForwardFloat(
            float[] input, int inputOffset, float[] output, int outputOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void BatchNormBackwardFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradInput, int gradInputOffset,
            float[] kernelDiff, int kernelDiffOffset, float[] biasDiff, int biasDiffOffset, long classPtr);

    public native static long BatchNormInitDouble(
            int inputNumber, int inputChannel, int inputHeight, int inputWidth,
            double eps, int useKernel, int useBias,
            int dimension, String name);
    public native static void BatchNormForwardDouble(
            double[] input, int inputOffset, double[] output, int outputOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void BatchNormBackwardDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradInput, int gradInputOffset,
            double[] kernelDiff, int kernelDiffOffset, double[] biasDiff, int biasDiffOffset, long classPtr);

    /* LRN API*/
    public native static long LRNInitFloat(int inputNumber, int inputChannel, int inputHeight, int inputWidth,
                                           int size, float alpha, float beta, float k, int dimension);
    public native static void LRNForwardFloat(float[] input, int inputOffset, float[] output, int outputOffset, long classPtr);
    public native static void LRNBackwardFloat(float[] input, int inputOffset,
                                               float[] outputDiff, int outputOffsetDiff,
                                               float[] inputDiff, int inputDiffOffset,
                                               long classPtr);
    public native static long LRNInitDouble(int inputNumber, int inputChannel, int inputHeight, int inputWidth,
                                           int size, double alpha, double beta, double k, int dimension);
    public native static void LRNForwardDouble(double[] input, int inputOffset, double[] output, int outputOffset, long classPtr);
    public native static void LRNBackwardDouble(double[] input, int inputOffset,
                                               double[] outputDiff, int outputOffsetDiff,
                                               double[] inputDiff, int inputDiffOffset,
                                               long classPtr);


    /* Init MKL Model */
    public native static void SetPrevFloat(long prev, long current);
    public native static void SetPrevDouble(long prev, long current);

    public native static void SetConcatPrevFloat(long prev, int index, long current);
    public native static void SetConcatPrevDouble(long prev, int index, long current);
    public native static void SetConcatNextFloat(long prev, int index, long current);
    public native static void SetConcatNextDouble(long prev, int index, long current);

    public native static void SetSumNextFloat(long prev, int index, long current);
    public native static void SetSumNextDouble(long prev, int index, long current);

    public native static void SetNextFloat(long prev, long current);
    public native static void SetNextDouble(long prev, long current);

    public native static void SetIPrevFloat(long prev, int index, long current);
    public native static void SetIPrevDouble(long prev, int index, long current);

    /* Delete all memmory allocated */
    public native static void ReleaseAllMemFloat(long classPtr);
    public native static void ReleaseAllMemDouble(long classPtr);


    // TODO
    /* Linear API */
    public native static long LinearInitFloat(
            int inputHeight, int inputWidth, int outputChannel,
            int kernelHeight, int kernelWidth, String name);
    public native static void LinearForwardFloat(
            float[] input, int inputOffset, float[] output, int outputOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void LinearBackwardDataFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradInput, int gradInputOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void LinearBackwardKernelFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradKernel, int gradKernelOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);
    public native static void LinearBackwardBiasFloat(
            float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset,
            float[] gradBias, int gradBiasOffset,
            float[] kernel, int kernelOffset, float[] bias, int biasOffset, long classPtr);

    public native static long LinearInitDouble(
            int inputHeight, int inputWidth, int outputChannel,
            int kernelHeight, int kernelWidth, String name);
    public native static void LinearForwardDouble(
            double[] input, int inputOffset, double[] output, int outputOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void LinearBackwardDataDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradInput, int gradInputOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void LinearBackwardKernelDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradKernel, int gradKernelOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);
    public native static void LinearBackwardBiasDouble(
            double[] input, int inputOffset, double[] gradOutput, int gradOutputOffset,
            double[] gradBias, int gradBiasOffset,
            double[] kernel, int kernelOffset, double[] bias, int biasOffset, long classPtr);

    /* Concat API */
    public native static long ConcatInitFloat(int numChannels, int dimension, int[] size);
    public native static void ConcatForwardFloat(float[][] input, int[] inputOffset, float[] output, int outputOffset, long classPtr);
    public native static void ConcatBackwardFloat(float[][] gradInput, int[] gradInputOffset, float[] output, int outputOffset, long classPtr);
    public native static long ConcatInitDouble(int numChannels, int dimension, int[] size);
    public native static void ConcatForwardDouble(double[][] input, int[] inputOffset, double[] output, int outputOffset, long classPtr);
    public native static void ConcatBackwardDouble(double[][] gradInput, int[] gradInputOffset, double[] output, int outputOffset, long classPtr);

    /* Sum API */
    public native static long SumInitFloat(int numChannels, int dimension, int[] size);
    public native static void SumForwardFloat(float[] input, int inputOffset, float[][] output, int[] outputOffset, long classPtr);
    public native static void SumBackwardFloat(float[] inputDiff, int inputOffset, float[][] outputDiff, int[] outputDiffOffset, long classPtr);
    public native static long SumInitDouble(int numChannels, int dimension, int[] size);
    public native static void SumForwardDouble(double[] input, int inputOffset, double[][] output, int[] outputOffset, long classPtr);
    public native static void SumBackwardDouble(double[] inputDiff, int inputOffset, double[][] outputDiff, int[] outputDiffOffset, long classPtr);
}
