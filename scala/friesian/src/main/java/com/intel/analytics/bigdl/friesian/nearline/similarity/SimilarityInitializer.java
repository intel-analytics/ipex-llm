package com.intel.analytics.bigdl.friesian.nearline.similarity;

import com.intel.analytics.bigdl.friesian.nearline.feature.FeatureNearlineUtils;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineHelper;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.grpc.ConfigParser;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class SimilarityInitializer {
    private RedisUtils redis;

    SimilarityInitializer() {
        redis = RedisUtils.getInstance(256, NearlineUtils.helper().redisHostPort(),
                NearlineUtils.helper().getRedisKeyPrefix(), NearlineUtils.helper().itemSlotType());
    }

    public void init() {
        SparkSession spark = SparkSession.builder().getOrCreate();
        FeatureNearlineUtils.loadItemNeighborRDD(spark, redis);
    }

    public static void main(String[] args) throws InterruptedException, IOException {
        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("c", "The path to the yaml config file.",
                "./config_similarity.yaml");

        cmdParser.parseOptions(args);
        String configPath = cmdParser.getOptionValue("c");

        NearlineUtils.helper_$eq(ConfigParser.loadConfigFromPath(configPath, NearlineHelper.class));
        NearlineUtils.helper().parseConfigStrings();
        SimilarityInitializer initializer = new SimilarityInitializer();
        initializer.init();
    }
}
