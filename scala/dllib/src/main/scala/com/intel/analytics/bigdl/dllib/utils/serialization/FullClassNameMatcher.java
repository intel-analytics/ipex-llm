package com.intel.analytics.bigdl.dllib.utils.serialization;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * A {@link ClassNameMatcher} that matches on full class names.
 * <p>
 * This object is immutable and thread-safe.
 * </p>
 */
final class FullClassNameMatcher implements ClassNameMatcher {

    private final Set<String> classesSet;

    /**
     * Constructs an object based on the specified class names.
     *
     * @param classes a list of class names
     */
    public FullClassNameMatcher(final String... classes) {
        classesSet = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(classes)));
    }

    @Override
    public boolean matches(final String className) {
        return classesSet.contains(className);
    }
}
