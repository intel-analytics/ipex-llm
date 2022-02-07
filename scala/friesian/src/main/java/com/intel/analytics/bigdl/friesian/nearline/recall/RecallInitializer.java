package com.intel.analytics.bigdl.friesian.nearline.recall;

import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineHelper;
import com.intel.analytics.bigdl.friesian.serving.recall.IndexService;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.grpc.ConfigParser;
import com.intel.analytics.bigdl.orca.inference.InferenceModel;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.config.Configurator;
import java.io.IOException;

public class RecallInitializer {
    // TODO: add item model inference ?
    private InferenceModel itemModel;

    public void init() {
        IndexService indexService = new IndexService(NearlineUtils.helper().indexDim());
        String dataDir = NearlineUtils.helper().getInitialDataPath();
        RecallNearlineUtils.loadItemData(indexService, dataDir);
        assert(indexService.isTrained());
        System.out.printf("Index service nTotal = %d\n", indexService.getNTotal());
    }

    public static void main(String[] args) throws InterruptedException, IOException {
        Configurator.setLevel("org", Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("c", "The path to the yaml config file.",
                "./config_recall.yaml");

        cmdParser.parseOptions(args);
        String configPath = cmdParser.getOptionValue("c");

        NearlineUtils.helper_$eq(ConfigParser.loadConfigFromPath(configPath, NearlineHelper.class));
        NearlineUtils.helper().parseConfigStrings();
        RecallInitializer initializer = new RecallInitializer();
        initializer.init();
    }
}
