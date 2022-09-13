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

/**
 * An object that matches a Class name to a condition.
 */
public interface ClassNameMatcher {

    /**
     * Returns <code>true</code> if the supplied class name matches this object's condition.
     *
     * @param className fully qualified class name
     * @return <code>true</code> if the class name matches this object's condition
     */
    boolean matches(String className);
}
