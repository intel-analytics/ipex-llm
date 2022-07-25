package com.intel.analytics.bigdl.friesian.serving.feature.utils;

import io.lettuce.core.*;
import io.lettuce.core.api.async.RedisStringAsyncCommands;
import io.lettuce.core.api.sync.RedisStringCommands;
import io.lettuce.core.cluster.ClusterClientOptions;
import io.lettuce.core.cluster.ClusterTopologyRefreshOptions;
import io.lettuce.core.cluster.RedisClusterClient;
import io.lettuce.core.cluster.api.StatefulRedisClusterConnection;
import io.lettuce.core.codec.StringCodec;
import io.lettuce.core.event.DefaultEventPublisherOptions;
import io.lettuce.core.event.EventBus;
import io.lettuce.core.event.metrics.CommandLatencyEvent;
import io.lettuce.core.masterreplica.MasterReplica;
import io.lettuce.core.masterreplica.StatefulRedisMasterReplicaConnection;
import io.lettuce.core.metrics.DefaultCommandLatencyCollector;
import io.lettuce.core.metrics.DefaultCommandLatencyCollectorOptions;
import io.lettuce.core.resource.ClientResources;
import io.lettuce.core.resource.DefaultClientResources;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import scala.Tuple2;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.intel.analytics.bigdl.friesian.serving.utils.Utils;

public class LettuceUtils {
    private static final Logger logger = LogManager.getLogger(LettuceUtils.class.getName());
    private static LettuceUtils instance = null;
    private static StatefulRedisMasterReplicaConnection<String, String> standaloneConn = null;
    private static StatefulRedisClusterConnection<String, String> clusterConn = null;
    private static AbstractRedisClient redisClient;
    private final String redisKeyPrefix;
    private RedisType redisType;
    private int itemSlotType;

    private LettuceUtils(RedisType redisType, ArrayList<Tuple2<String, Integer>> redisHostPort,
                         String redisPrefix, String sentinelMasterUrl, String sentinelMasterName, int itemSlotType) {
        int logInterval = 2;
        this.redisType = redisType;
        this.itemSlotType = itemSlotType;
        if (Utils.helper() != null) {
            logInterval = Utils.helper().logInterval();
        }
        ClientResources res;
        if (logInterval > 0) {
            res = DefaultClientResources.builder()
                    .commandLatencyPublisherOptions(
                            DefaultEventPublisherOptions.builder()
                                    .eventEmitInterval(Duration.ofMinutes(logInterval)).build())
                    .commandLatencyRecorder(
                            new DefaultCommandLatencyCollector(
                                    DefaultCommandLatencyCollectorOptions.builder()
                                            .targetPercentiles(new double[]{50.0, 95.0, 99.0}).build()))
                    .build();
        } else {
            res = DefaultClientResources.builder().build();
        }

        List<RedisURI> nodes = new ArrayList<>();
        if (this.redisType == RedisType.SENTINEL || this.redisType == RedisType.STANDALONE) {
            RedisClient redisStandaloneClient = RedisClient.create(res);
            redisClient = redisStandaloneClient;
            if (this.redisType == RedisType.STANDALONE) {
                for (Tuple2<String, Integer> stringIntegerTuple2 : redisHostPort) {
                    nodes.add(RedisURI.create(stringIntegerTuple2._1, stringIntegerTuple2._2));
                }
            } else {
                String[] sentinelUrl = sentinelMasterUrl.split(":");
                assert sentinelUrl.length == 2;
                nodes.add(RedisURI.Builder.sentinel(sentinelUrl[0].trim(), Integer.parseInt(sentinelUrl[1].trim()),
                        sentinelMasterName).build());
            }
            standaloneConn = MasterReplica.connect(redisStandaloneClient, StringCodec.UTF8, nodes);
            if (nodes.size() == 1) {
                standaloneConn.setReadFrom(ReadFrom.ANY);
            } else {
                standaloneConn.setReadFrom(ReadFrom.ANY_REPLICA);
            }
        } else {
            for (Tuple2<String, Integer> stringIntegerTuple2 : redisHostPort) {
                nodes.add(RedisURI.create(stringIntegerTuple2._1, stringIntegerTuple2._2));
            }
            RedisClusterClient redisClusterClient = RedisClusterClient.create(nodes);
            ClusterTopologyRefreshOptions topologyRefreshOptions =
                    ClusterTopologyRefreshOptions.builder()
                            .enableAdaptiveRefreshTrigger(ClusterTopologyRefreshOptions.RefreshTrigger.MOVED_REDIRECT,
                                    ClusterTopologyRefreshOptions.RefreshTrigger.PERSISTENT_RECONNECTS)
                            .adaptiveRefreshTriggersTimeout(Duration.ofSeconds(30)).build();
            redisClusterClient.setOptions(ClusterClientOptions.builder()
                            .topologyRefreshOptions(topologyRefreshOptions).build());
            redisClient = redisClusterClient;
            clusterConn = redisClusterClient.connect();
        }

        logger.info("Connected to Redis");
        redisKeyPrefix = redisPrefix;

        if (logInterval > 0) {
            EventBus eventBus = redisClient.getResources().eventBus();
            eventBus.get()
                    .filter(redisEvent -> redisEvent instanceof CommandLatencyEvent)
                    .cast(CommandLatencyEvent.class)
                    .subscribe(e -> logger.info(e.getLatencies()));
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (standaloneConn != null) {
                standaloneConn.close();
            }
            if (clusterConn != null) {
                clusterConn.close();
            }
            if (redisClient != null) {
                redisClient.shutdown();
            }
        }));
    }

    public static LettuceUtils getInstance(RedisType redisType, ArrayList<Tuple2<String, Integer>> redisHostPort,
                                           String redisPrefix, String sentinelMasterURL, String sentinelMasterName,
                                           int itemSlotType) {
        if (instance == null) {
            instance = new LettuceUtils(redisType, redisHostPort, redisPrefix, sentinelMasterURL,
                    sentinelMasterName, itemSlotType);
        }
        return instance;
    }

    public RedisStringCommands<String, String> getSync() {
        if (redisType == RedisType.CLUSTER) {
            return clusterConn.sync();
        } else {
            return standaloneConn.sync();
        }
    }

    public RedisStringAsyncCommands<String, String> getAsync() {
        if (redisType == RedisType.CLUSTER) {
            return clusterConn.async();
        } else {
            return standaloneConn.async();
        }
    }

    public String getSchema(String keyPrefix) {
        String hKey = this.redisKeyPrefix + keyPrefix;
        return get(hKey);
    }

    public void setSchema(String keyPrefix, String value) {
        String hKey = this.redisKeyPrefix + keyPrefix;
        set(hKey, value);
    }

    public String get(String key) {
        RedisStringAsyncCommands<String, String> async = getAsync();
        RedisFuture<String> future = async.get(key);
        String result = "";
        try {
            result = future.get(10000, TimeUnit.MILLISECONDS);
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            e.printStackTrace();
        }
        return result;
    }

    public void set(String key, String value) {
        RedisStringAsyncCommands<String, String> async = getAsync();
        async.set(key, value);
        logger.info("1 valid records written to redis.");
    }

    public void MSet(String keyPrefix, String[][] dataArray) {
        Map<String, String> keyValue = new HashMap<>();
        if (redisType == RedisType.CLUSTER && keyPrefix.equals("item") && itemSlotType != 0) {
            if (itemSlotType == 1) {
                for (String[] data : dataArray) {
                    if (data.length != 2) {
                        logger.warn("Data size in dataArray should be 2, but got" + data.length);
                    } else {
                        keyValue.put(this.redisKeyPrefix + "{" + keyPrefix + "}:" + data[0], data[1]);
                    }
                }
            } else {
                for (String[] data: dataArray) {
                    if (data.length != 2) {
                        logger.warn("Data size in dataArray should be 2, but got" + data.length);
                    } else {
                        keyValue.put("{" + this.redisKeyPrefix + keyPrefix + data[0].charAt(data[0].length() - 1) +
                                "}:" + data[0], data[1]);
                    }
                }
            }
        } else {
            for (String[] data : dataArray) {
                if (data.length != 2) {
                    logger.warn("Data size in dataArray should be 2, but got" + data.length);
                } else {
                    keyValue.put(this.redisKeyPrefix + keyPrefix + ":" + data[0], data[1]);
                }
            }
        }
        RedisStringAsyncCommands<String, String> async = getAsync();
        async.mset(keyValue);
        logger.info(keyValue.size() + " valid records written to redis.");
    }

    public List<String> MGet(String keyPrefix, List<Integer> ids) {
        String[] keys = new String[ids.size()];
        for (int i = 0; i < ids.size(); i ++) {
            keys[i] = String.valueOf(ids.get(i));
        }
        return MGet(keyPrefix, keys);
    }

    public List<String> MGet(String keyPrefix, String[] keys) {
        String[] redisKeys = new String[keys.length];
        if (redisType == RedisType.CLUSTER && keyPrefix.equals("item") && itemSlotType != 0) {
            if (itemSlotType == 1) {
                for (int i = 0; i < keys.length; i ++) {
                    redisKeys[i] = this.redisKeyPrefix + "{" + keyPrefix + "}:" + keys[i];
                }
            } else {
                for (int i = 0; i < keys.length; i ++) {
                    // TODO: keys[i] = ""
                    redisKeys[i] = "{" + this.redisKeyPrefix + keyPrefix + keys[i].charAt(keys[i].length() - 1) + "}:" +
                            keys[i];
                }
            }
        } else {
            for (int i = 0; i < keys.length; i ++) {
                redisKeys[i] = this.redisKeyPrefix + keyPrefix + ":" + keys[i];
            }
        }
        return MGet(redisKeys);
    }

    private List<String> MGet(String[] keys) {
        RedisStringAsyncCommands<String, String> async = getAsync();
        RedisFuture<List<KeyValue<String, String>>> future = async.mget(keys);
        List<KeyValue<String, String>> result = null;
        try {
            result = future.get(10000, TimeUnit.MILLISECONDS);
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            e.printStackTrace();
        }
        List<String> values = new ArrayList<>(keys.length);
        if (result != null) {
            for (KeyValue<String, String> kv: result) {
                if (kv.hasValue()) {
                    values.add(kv.getValue());
                } else {
                    values.add("");
                }
            }
        }
        return values;
    }
}
