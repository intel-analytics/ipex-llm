/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pmem;

public class PersistentMemoryAllocator implements BasicMemoryAllocator {
    private static volatile PersistentMemoryAllocator instance;

    private PersistentMemoryAllocator() {
        com.intel.analytics.zoo.core.pmem.PMemNativeLoader.load();
    }

    public static PersistentMemoryAllocator getInstance() {
        if (instance == null) {
            synchronized (PersistentMemoryAllocator.class) {
                if (instance == null) {
                    instance = new PersistentMemoryAllocator();
                }
            }
        }
        return instance;
    }

    public native void initialize(String path, long size);

    public native long allocate(long size);

    public native void free(long address);

    public native void copy(long destAddress, long srcAddress, long size);
}
