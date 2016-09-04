package com.intel.analytics.dllib.mkl;

import com.github.fommil.jni.JniLoader;

public class MKL {
    private static boolean isLoaded = false;
    static {
        isLoaded = true;
        try {
            JniLoader.load("libjmkl.so");
        } catch (ExceptionInInitializerError ex) {
            isLoaded = false;
        }
    }
    public native static void setNumThreads(int numThreads);

    public native static int getNumThreads();

    public static boolean isMKLLoaded() {
        return isLoaded;
    }
}
