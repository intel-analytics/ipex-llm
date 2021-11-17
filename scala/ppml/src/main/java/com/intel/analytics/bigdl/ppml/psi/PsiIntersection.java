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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

public class PsiIntersection {
    public final int maxCollection;
    public final int shuffleSeed;

    protected final int nThreads = Integer.parseInt(System.getProperty(
            "PsiThreads", "6"));

    protected ExecutorService pool = Executors.newFixedThreadPool(nThreads);

    public PsiIntersection(int maxCollection, int shuffleSeed) {
        this.maxCollection = maxCollection;
        this.shuffleSeed = shuffleSeed;
    }

    protected List<String[]> collections = new ArrayList<String[]>();
    protected List<String> intersection;

    public int numCollection() {
        return collections.size();
    }

    public void addCollection(
            String[] collection) throws InterruptedException, ExecutionException{
        synchronized (this) {
            if (collections.size() == maxCollection) {
                throw new IllegalArgumentException("Collection is full.");
            }
            collections.add(collection);
            if (collections.size() >= maxCollection) {
                // TODO: sort by collections' size
                String[] current = collections.get(0);
                for(int i = 1; i < maxCollection - 1; i++){
                    Arrays.parallelSort(current);
                    current = findIntersection(current, collections.get(i))
                        .toArray(new String[intersection.size()]);
                }
                Arrays.parallelSort(current);
                List<String> result = findIntersection(current, collections.get(maxCollection - 1));
                Utils.shuffle(result, shuffleSeed);
                intersection = result;
                this.notifyAll();
            }
        }
    }

    // Join a with b, a should be sorted.
    private static class FindIntersection implements Callable<List<String>> {
        protected String[] a;
        protected String[] b;
        protected int bStart;
        protected int length;

        public FindIntersection(String[] a,
                                String[] b,
                                int bStart,
                                int length) {
            this.a = a;
            this.b = b;
            this.bStart = bStart;
            this.length = length;
        }

        @Override
        public List<String> call() {
            return findIntersection(a, b, bStart, length);
        }

        protected static List<String> findIntersection(
                String[] a,
                String[] b,
                int start,
                int length){
            ArrayList<String> intersection = new ArrayList<String>();
            for(int i = start; i < length + start; i++) {
                if (Arrays.binarySearch(a, b[i]) >= 0){
                    intersection.add(b[i]);
                }
            }
            return intersection;
        }
    }

    protected List<String> findIntersection(
            String[] a,
            String[] b) throws InterruptedException, ExecutionException{
        int[] splitPoints = new int[nThreads + 1];
        int extractLen = b.length - nThreads * (b.length / nThreads);
        for(int i = 1; i < splitPoints.length; i++) {
            splitPoints[i] = b.length / nThreads * i;
            if (i <= extractLen) {
                splitPoints[i] += i;
            } else {
                splitPoints[i] += extractLen;
            }
        }

        Future<List<String>>[] futures = new Future[nThreads];
        for(int i = 0; i < nThreads; i++) {
            futures[i] = pool.submit(new FindIntersection(a, b, splitPoints[i],
                splitPoints[i + 1] - splitPoints[i]));
        }
        List<String> intersection = futures[0].get();
        for(int i = 1; i < nThreads; i++) {
            intersection.addAll(futures[i].get());
        }
        return intersection;
    }

    public List<String> getIntersection() throws InterruptedException{
        synchronized (this) {
            while (intersection == null) {
                this.wait();
            }
            return intersection;
        }
    }
}
