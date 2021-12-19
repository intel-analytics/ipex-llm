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

package com.intel.analytics.bigdl.grpc;


import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;

/**
 * ConfigParser has static method to read config formatted in JavaBean class from YAML file
 */
public class ConfigParser {
    static ObjectMapper objectMapper;
    static {
        objectMapper = new ObjectMapper(new YAMLFactory());
        objectMapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        objectMapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
    }

    public static <T> T loadConfigFromPath(String configPath, Class<T> valueType)
            throws IOException {
        return objectMapper.readValue(new java.io.File(configPath), valueType);
    }
    public static <T> T loadConfigFromString(String configString, Class<T> valueType)
            throws IOException {

        return objectMapper.readValue(configString, valueType);
    }
}