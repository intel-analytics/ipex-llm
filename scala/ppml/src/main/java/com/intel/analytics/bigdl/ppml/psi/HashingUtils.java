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

package com.intel.analytics.bigdl.ppml.psi;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.*;
import java.util.concurrent.*;

public class HashingUtils {
    private static final Logger logger = LoggerFactory.getLogger(HashingUtils.class);
    /***
     * Gen random HashMap<String, String> for test
     * @param size HashMap size, int
     * @return
     */
    public static HashMap<String, String> genRandomHashSet(int size) {
        HashMap<String, String> data = new HashMap<>();
        Random rand = new Random();
        for (int i = 0; i < size; i++) {
            String name = "User_" + rand.nextInt();
            data.put(name, Integer.toString(i));
        }
        logger.info("IDs are: ");
        for (Map.Entry<String, String> element : data.entrySet()) {
            logger.info(element.getKey() + ",  " + element.getValue());
        }
        return data;
    }

    protected static final int nThreads = Integer.parseInt(System.getProperty(
            "PsiThreads", "6"));


    public static List<String> parallelToSHAHexString(
            List<String> ids,
            String salt) throws InterruptedException, ExecutionException {
        return parallelToSHAHexString(ids, salt, 256, 32);
    }

    public static List<String> parallelToSHAHexString(
            List<String> ids,
            String salt,
            int length,
            int paddingSize) throws InterruptedException, ExecutionException {
        String[] idsArray = ids.toArray(new String[ids.size()]);
        String[] hashedIds = parallelToSHAHexString(idsArray, salt, length, paddingSize);
//        return new ArrayList<String>(Arrays.asList(hashedIds));
        return Arrays.asList(hashedIds);
    }

    public static String[] parallelToSHAHexString(
            String[] ids,
            String salt) throws InterruptedException, ExecutionException {
        return parallelToSHAHexString(ids, salt, 256, 32);
    }

    public static String[] parallelToSHAHexString(
            String[] ids,
            String salt,
            int length,
            int paddingSize) throws InterruptedException, ExecutionException {
        String[] output = new String[ids.length];
        ExecutorService pool = Executors.newFixedThreadPool(nThreads);
        int extractLen = ids.length - nThreads * (ids.length / nThreads);
        int average = ids.length / nThreads;
        Future<Integer>[] futures = new Future[nThreads];
        for(int i = 0; i < nThreads - 1; i++) {
            futures[i] = pool.submit(new StringToSHAHex(ids, average * i,
                    average, output, salt, length, paddingSize));
        }
        futures[nThreads - 1] = pool.submit(new StringToSHAHex(ids, average * (nThreads - 1),
                average + extractLen, output, salt, length, paddingSize));

        for(int i = 0; i < nThreads; i++) {
            futures[i].get();
        }
        pool.shutdown();
        return output;
    }

    private static class StringToSHAHex implements Callable<Integer> {
        protected String[] src;
        protected int start;
        protected int length;
        protected String[] dest;
        protected int paddingSize;
        protected String salt;
        protected MessageDigest generator;

        public StringToSHAHex(
                String[] src,
                int start,
                int length,
                String[] dest) {
            this(src, start, length, dest, "", 256, 32);
        }

        public StringToSHAHex(
                String[] src,
                int start,
                int length,
                String[] dest,
                String salt,
                int shaLength,
                int paddingSize) {
            this.src = src;
            this.start = start;
            this.length = length;
            this.dest = dest;
            this.paddingSize = paddingSize;
            this.salt = salt;
            try {
                this.generator = MessageDigest.getInstance("SHA-" + shaLength);
            } catch (NoSuchAlgorithmException nsae) {
                nsae.printStackTrace();
                throw new RuntimeException(nsae);
            }
        }

        @Override
        public Integer call() {
            toSHAHexString();
            return 0;
        }

        protected void toSHAHexString() {
            for(int i = start; i < length + start; i++) {
                dest[i] = toHexString(
                        generator.digest((src[i] + salt).getBytes(StandardCharsets.UTF_8)), paddingSize);
            }
        }
    }



    public static byte[] getSecurityRandomBytes() {
        SecureRandom random = new SecureRandom();
        byte[] randBytes = new byte[20];
        random.nextBytes(randBytes);
        return randBytes;
    }

    public static byte[] getSHA(String input) throws NoSuchAlgorithmException {
        return getSHA(input, 256);
    }

    public static byte[] int2Bytes(int value) {
        byte[] src = new byte[4];
        src[3] =  (byte) ((value>>24) & 0xFF);
        src[2] =  (byte) ((value>>16) & 0xFF);
        src[1] =  (byte) ((value>>8) & 0xFF);
        src[0] =  (byte) (value & 0xFF);
        return src;
    }

    /***
     * Get random HashMap<String, String> for test of random string
     * @param size HashMap size, int
     * @return
     */
    public static HashMap<String, String> getRandomHashSetOfString(int size) {
        HashMap<String, String> data = new HashMap<>();
        for (int i = 0; i < size; i++) {
            String name = toHexString(int2Bytes(i));
            data.put(name, Integer.toString(i));
        }
        logger.info("IDs are: ");
        for (Map.Entry<String, String> element : data.entrySet()) {
            logger.info(element.getKey() + ",  " + element.getValue());
        }
        return data;
    }

    public static HashMap<String, String> getRandomHashSetOfStringForFiveFixed(int size) {
        HashMap<String, String> data = new HashMap<>();
        Random rand = new Random();
        // put several constant for test
        String nameTest = "User_11111111111111111111111111111";//randomBytes;
        data.put(nameTest, Integer.toString(0));
        nameTest = "User_111111111111111111111111122222";//randomBytes;
        data.put(nameTest, Integer.toString(1));
        nameTest = "User_11111111111111111111111133333";//randomBytes;
        data.put(nameTest, Integer.toString(2));
        nameTest = "User_11111111111111111111111144444";//randomBytes;
        data.put(nameTest, Integer.toString(3));
        nameTest = "User_11111111111111111111111155555";//randomBytes;
        data.put(nameTest, Integer.toString(4));
        for (int i = 5; i < size; i++) {
            //String randomBytes = new String(getSecurityRandomBytes());
            String name = toHexString(int2Bytes(i));//randomBytes;
            data.put(name, Integer.toString(i));
        }
        logger.info("IDs are: ");
        for (Map.Entry<String, String> element : data.entrySet()) {
            logger.info(element.getKey() + ",  " + element.getValue());
        }
        return data;
    }


    /**
     * Get SHA hash result of given string input
     *
     * @param input string input
     * @param length bit length, e.g., 128 and 256
     * @return
     * @throws NoSuchAlgorithmException
     */
    public static byte[] getSHA(String input, int length) throws NoSuchAlgorithmException {
        return MessageDigest.getInstance("SHA-" + length).digest(input.getBytes(StandardCharsets.UTF_8));
    }

    public static String toHexString(byte[] hash) {
        return toHexString(hash, 32);
    }

    public static String toHexString(byte[] hash, int paddingSize) {
        // Convert byte array into signum representation
        BigInteger number = new BigInteger(1,  hash);

        // Convert message digest into hex value
        StringBuilder hexString = new StringBuilder(number.toString(16));

        // Pad with leading zeros
        while (hexString.length() < paddingSize) {
            hexString.insert(0,  '0');
        }

        return hexString.toString();
    }

    public static boolean checkHash(byte[] bytes, String hashstr) {
        // transfor bytes to String type
        StringBuffer hexValues = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            int val = ((int) bytes[i]) & 0xff;
            if (val < 16) {
                hexValues.append("0");
            }
            hexValues.append(Integer.toHexString(val));
        }
        String bytestr = hexValues.toString();
        return bytestr.equals(hashstr);
    }
}
