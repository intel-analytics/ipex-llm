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

import org.apache.commons.cli.*;

import java.io.IOException;

public abstract class AbstractZooGrpc {
    protected String[] args;
    protected Options options = new Options();
    protected String configPath;
    protected CommandLine cmd;

    protected <T> T getConfigFromYaml(Class<T> valueType, String defaultConfigPath)
            throws IOException {
        options.addOption(new Option(
                "c", "config", true, "config path"));
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }
        assert cmd != null;
        configPath = cmd.getOptionValue("config", defaultConfigPath);
        if (configPath != null) {
            System.out.println("Load config from " + configPath);
            // config YAML passed, use config YAML first, command-line could overwrite
            assert valueType != null;
            return ConfigParser.loadConfigFromPath(configPath, valueType);
        }
        else {
            System.out.println("Config is not provided, using default");
            return null;
        }
    }

    public CommandLine getCmd() {
        return cmd;
    }
}
