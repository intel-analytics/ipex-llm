/*
 * Copyright 2019 Analytics Zoo Authors.
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

package com.intel.analytics.bigdl.orca.tf;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;

public class TFNetNative {

    private static final boolean DEBUG =
            System.getProperty("com.intel.analytics.bigdl.orca.tf.TFNetNative") != null;

    private static boolean isLoaded = false;

    private static final String[] LINUX_LIBS = new String[]{
            "iomp5",
            "mklml_intel",
            "tensorflow_framework-zoo",
            "tensorflow_jni"
    };

    private static final String[] OSX_LIBS = new String[]{
            "iomp5",
            "mklml",
            "tensorflow_framework",
            "tensorflow_jni"
    };

    private static final String[] WINDOWS_LIBS = new String[]{
            "tensorflow_jni"
    };

    private static String libSharePath() throws URISyntaxException {
        String jarPath = new File(TFNetNative.class.getProtectionDomain().getCodeSource().getLocation()
    .toURI()).getParent();
        String libPath = new File(jarPath, "../../tflibs/").getPath();
        File libPathDir = new File(libPath);
        if (libPathDir.exists()) { // Orca is installed in conda env.
            return libPath;
        } else { // Orca is installed in a local directory.
            return handleCondaLibPath();
        }
    }

    // For example, aiming to lead pyspark_python path(e.g. "/home/arda/anaconda3/envs/py37/bin/python")
    // to where libraries lay(e.g. "/home/arda/anaconda3/envs/py37/lib/python3.7/site-packages/bigdl/share/tflibs")
    private static String handleCondaLibPath() {
        String pyspark_python = System.getenv("PYSPARK_PYTHON"); 
        String pyLibBasePath = pyspark_python.replace("bin/python", "lib");
        File pyLibBase = new File(pyLibBasePath);
        if (!pyLibBase.exists()) {
            return null;
        }
        String pyVerSpecName = null;
        for (File f:pyLibBase.listFiles()) {
            if(f.isDirectory() && f.getName().startsWith("python3.")){
                pyVerSpecName = f.getName();
                break;
            }
        }
        String libPath = String.join("/", pyLibBasePath, pyVerSpecName, "site-packages/bigdl/share/tflibs");
        System.out.println("Try to load libraries from conda path: "+libPath);
        
        return libPath;
    }

    static {
        String[] LIBS;
        String osStr = os();
        if (osStr.equals("linux")) {
            LIBS = LINUX_LIBS;
        } else if (osStr.equals("darwin")) {
            LIBS = OSX_LIBS;
        } else if (osStr.equals("windows")) {
            LIBS = WINDOWS_LIBS;
        } else {
            throw new RuntimeException("os type: " + osStr + " not supported");
        }

        if (!tryLoadLibrary(LIBS)) {
            log("Could not find native libraries in java.library.path, extracting from jar file");
            try {
                final File tempPath = createTemporaryDirectory();
                tempPath.deleteOnExit();
                final String tempDirectory = tempPath.getCanonicalPath();

                for (int i = 0; i < LIBS.length; i++) {
                    String libName;
                    if (LIBS[i].equals("tensorflow_framework")) {
                        libName = getVersionedLibraryName(System.mapLibraryName(LIBS[i]));
                    } else {
                        libName = System.mapLibraryName(LIBS[i]);
                    }
                    String resourceName = makeResourceName(libName);
                    System.out.println("Try to locate " + resourceName);
                    File shareLibFile = new File(libSharePath(), resourceName);
                    if (shareLibFile.exists()) {
                        System.out.println("Got shared path at " + shareLibFile.getPath());
                        System.load(shareLibFile.getPath());
                    } else {
                        System.load(extractResource(getResource(resourceName), libName, tempDirectory));
                    }
                }
            } catch (Exception e) {
                throw new UnsatisfiedLinkError(
                        String.format(
                                "Unable to extract native library into a temporary file (%s)", e.toString()));
            }

        }
        isLoaded = true;
    }

    private static boolean tryLoadLibrary(String[] libs) {
        log("try loading native libraries from java.library.path ");
        try {
            for (int i = 0; i < libs.length; i++) {
                System.loadLibrary(libs[i]);
            }
            log("native libraries loaded");
            return true;
        } catch (UnsatisfiedLinkError e) {
            log("tryLoadLibraryFailed: " + e.getMessage());
            return false;
        }

    }

    private static void log(String msg) {
        if (DEBUG) {
            System.err.println("com.intel.analytics.zoo.core.TFNetNative: " + msg);
        }
    }

    public static boolean isLoaded() {
        return isLoaded;
    }

    private static InputStream getResource(String resourceName) throws IOException {
        InputStream stream = TFNetNative.class.getClassLoader().getResourceAsStream(resourceName);
        if (stream == null) {
            throw new IOException("Can not find resource " + resourceName);
        }

        return stream;
    }

    private static File createTemporaryDirectory() {
        File baseDirectory = new File(System.getProperty("java.io.tmpdir"));
        String directoryName = "tensorflow_native_libraries-" + System.currentTimeMillis() + "-";
        for (int attempt = 0; attempt < 1000; attempt++) {
            File temporaryDirectory = new File(baseDirectory, directoryName + attempt);
            if (temporaryDirectory.mkdir()) {
                return temporaryDirectory;
            }
        }
        throw new IllegalStateException(
                "Could not create a temporary directory (tried to make "
                        + directoryName
                        + "*) to extract TensorFlow native libraries.");
    }

    private static long copy(InputStream src, File dstFile) throws IOException {
        FileOutputStream dst = new FileOutputStream(dstFile);
        try {
            byte[] buffer = new byte[1 << 20]; // 1MB
            long ret = 0;
            int n = 0;
            while ((n = src.read(buffer)) >= 0) {
                dst.write(buffer, 0, n);
                ret += n;
            }
            return ret;
        } finally {
            dst.close();
            src.close();
        }
    }

    private static String extractResource(
            InputStream resource, String resourceName, String extractToDirectory) throws IOException {
        final File dst = new File(extractToDirectory, resourceName);
        dst.deleteOnExit();
        final String dstPath = dst.toString();

        final long nbytes = copy(resource, dst);
        return dstPath;
    }

    private static boolean resourceExists(String baseName) {
        return TFNetNative.class.getClassLoader().getResource(makeResourceName(baseName)) != null;
    }

    private static String getMajorVersionNumber() {
        String version = TFNetNative.class.getPackage().getImplementationVersion();
        // expecting a string like 1.14.0, we want to get the first '1'.
        int dotIndex;
        if (version == null || (dotIndex = version.indexOf('.')) == -1) {
            return null;
        }
        String majorVersion = version.substring(0, dotIndex);
        try {
            Integer.parseInt(majorVersion);
            return majorVersion;
        } catch (NumberFormatException unused) {
            return null;
        }
    }


    private static String getVersionedLibraryName(String libFilename) {
        // If the resource exists as an unversioned file, return that.
        if (resourceExists(libFilename)) {
            return libFilename;
        }

        final String versionName = getMajorVersionNumber();

        // If we're on darwin, the versioned libraries look like blah.1.dylib.
        final String darwinSuffix = ".dylib";
        if (libFilename.endsWith(darwinSuffix)) {
            final String prefix = libFilename.substring(0, libFilename.length() - darwinSuffix.length());
            if (versionName != null) {
                final String darwinVersionedLibrary = prefix + "." + versionName + darwinSuffix;
                if (resourceExists(darwinVersionedLibrary)) {
                    return darwinVersionedLibrary;
                }
            } else {
                // If we're here, we're on darwin, but we couldn't figure out the major version number. We
                // already tried the library name without any changes, but let's do one final try for the
                // library with a .so suffix.
                final String darwinSoName = prefix + ".so";
                if (resourceExists(darwinSoName)) {
                    return darwinSoName;
                }
            }
        } else if (libFilename.endsWith(".so")) {
            // Libraries ending in ".so" are versioned like "libfoo.so.1", so try that.
            final String versionedSoName = libFilename + "." + versionName;
            if (versionName != null && resourceExists(versionedSoName)) {
                return versionedSoName;
            }
        }

        // Otherwise, we've got no idea.
        return libFilename;
    }

    private static String makeResourceName(String baseName) {
        return String.format("%s-%s/", os(), architecture()) + baseName;
    }

    private static String os() {
        final String p = System.getProperty("os.name").toLowerCase();
        if (p.contains("linux")) {
            return "linux";
        } else if (p.contains("os x") || p.contains("darwin")) {
            return "darwin";
        } else if (p.contains("windows")) {
            return "windows";
        } else {
            return p.replaceAll("\\s", "");
        }
    }

    private static String architecture() {
        final String arch = System.getProperty("os.arch").toLowerCase();
        return (arch.equals("amd64")) ? "x86_64" : arch;
    }
}
