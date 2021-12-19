package com.intel.analytics.bigdl.friesian.serving.feature.utils;


import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import org.apache.log4j.Logger;
import redis.clients.jedis.*;
import redis.clients.jedis.exceptions.JedisConnectionException;

import java.util.*;


public class RedisUtils {
    private static final Logger logger = Logger.getLogger(RedisUtils.class.getName());
    private static RedisUtils instance = null;
    private static JedisPool jedisPool = null;
    private static JedisCluster cluster = null;

    private RedisUtils(int maxTotal) {
        JedisPoolConfig jedisPoolConfig = new JedisPoolConfig();
        jedisPoolConfig.setMaxTotal(maxTotal);


        if (Utils.helper().redisHostPort().size() == 1) {
            jedisPool = new JedisPool(jedisPoolConfig, Utils.helper().redisHostPort().get(0)._1,
                    (int) Utils.helper().redisHostPort().get(0)._2, 30000);
        } else {
            Set<HostAndPort> hps = new HashSet<HostAndPort>();
            for (int i = 0; i < Utils.helper().redisHostPort().size(); i++) {
                HostAndPort hp = new HostAndPort(Utils.helper().redisHostPort().get(i)._1,
                        (int) Utils.helper().redisHostPort().get(i)._2);
                hps.add(hp);
            }
            // default maxAttempt=5, service likely to down, increase to 20
            cluster = new JedisCluster(hps, 50000, 20, jedisPoolConfig);
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (cluster != null) {
                cluster.close();
            }
            if (jedisPool != null) {
                jedisPool.close();
            }
        }));
    }

    public JedisCluster getCluster() {
        return cluster;
    }

    public static RedisUtils getInstance(int maxTotal) {
        if (instance == null) {
            instance = new RedisUtils(maxTotal);
        }
        return instance;
    }

    public Jedis getRedisClient() {
        Jedis jedis = null;
        int cnt = 10;
        while (jedis == null) {
            try {
                jedis = jedisPool.getResource();
            } catch (JedisConnectionException e) {
                e.printStackTrace();
                cnt--;
                if (cnt <= 0) {
                    throw new Error("Cannot get redis from the pool");
                }
                try {
                    Thread.sleep(100);
                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }
            }
        }
        return jedis;
    }

    public void Mset(String keyPrefix, List<String>[] dataArray) {
        if (cluster == null) {
            jedisMset(keyPrefix, dataArray);
        } else {
            clusterSet(keyPrefix, dataArray);
        }
    }

    public void Hset(String keyPrefix, List<String>[] dataArray) {
        if (cluster == null) {
            piplineHmset(keyPrefix, dataArray);
        } else {
            clusterHset(keyPrefix, dataArray);
        }
    }

    public void clusterHset(String keyPrefix, List<String>[] dataArray) {
        int cnt = 0;
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                String hKey = Utils.helper().getRedisKeyPrefix() + keyPrefix + ":" +
                        data.get(0);
                Map<String, String> hValue = new HashMap<>();
                hValue.put("value", data.get(1));
                getCluster().hset(hKey, hValue);
                cnt += 1;
            }
        }
        logger.info(cnt + " valid records written to redis.");
    }

    public void setSchema(String keyPrefix, String colNames) {
        String hKey = Utils.helper().getRedisKeyPrefix() + keyPrefix;
        if (cluster == null) {
            Jedis jedis = getRedisClient();
            jedis.set(hKey, colNames);
            jedis.close();
        } else {
            getCluster().set(hKey, colNames);
        }
    }

    public void clusterSet(String keyPrefix, List<String>[] dataArray) {
        if (keyPrefix.equals("user") ||
                (keyPrefix.equals("item") && Utils.helper().itemSlotType() == 0)) {
            int cnt = 0;
            for(List<String> data: dataArray) {
                if(data.size() != 2) {
                    logger.warn("Data size in dataArray should be 2, but got" + data.size());
                } else {
                    String key = Utils.helper().getRedisKeyPrefix() + keyPrefix + ":" +
                            data.get(0);
                    getCluster().set(key, data.get(1));
                    cnt += 1;
                }
            }
            logger.info(cnt + " valid records written to redis.");
        } else if (keyPrefix.equals("item")) {
            if (Utils.helper().itemSlotType() == 1) {
                keyPrefix = "{" + Utils.helper().getRedisKeyPrefix() + keyPrefix + "}";
                String[] keyValues = buildKeyValuesArray(keyPrefix, dataArray);
                getCluster().mset(keyValues);
                logger.info(keyValues.length / 2 + " valid records written to redis.");
            } else {
                Collection<ArrayList<String>> keyValueSlots = buildAndDivideKeyValues(keyPrefix,
                        dataArray);
                int cnt = 0;
                for (ArrayList<String> kv: keyValueSlots) {
                    String[] kvs = kv.toArray(new String[0]);
                    getCluster().mset(kvs);
                    cnt += kvs.length;
                }
                logger.info(cnt / 2 + " valid records written to redis.");
            }
        } else {
            logger.error("keyPrefix should be user or item, but got " + keyPrefix);
        }
    }

    public void piplineHmset(String keyPrefix, List<String>[] dataArray) {
        Jedis jedis = getRedisClient();
        Pipeline ppl = jedis.pipelined();
        int cnt = 0;
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                String hKey = Utils.helper().getRedisKeyPrefix() + keyPrefix + ":" +
                        data.get(0);
                Map<String, String> hValue = new HashMap<>();
                hValue.put("value", data.get(1));
                ppl.hmset(hKey, hValue);
                cnt += 1;
            }
        }
        ppl.sync();
        jedis.close();
        logger.info(cnt + " valid records written to redis.");
    }

    public void jedisMset(String keyPrefix, List<String>[] dataArray) {
        Jedis jedis = getRedisClient();
        keyPrefix = Utils.helper().getRedisKeyPrefix() + keyPrefix;
        String[] keyValues = buildKeyValuesArray(keyPrefix, dataArray);
        jedis.mset(keyValues);
        jedis.close();
        logger.info(keyValues.length / 2 + " valid records written to redis.");
    }

    private String[] buildKeyValuesArray(String keyPrefix, List<String>[] dataArray) {
        int cnt = dataArray.length;
        ArrayList<String> keyValues = new ArrayList<>(cnt * 2);
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                String key = keyPrefix + ":" + data.get(0);
                keyValues.add(key);
                keyValues.add(data.get(1));
            }
        }
        return keyValues.toArray(new String[0]);
    }

    private Collection<ArrayList<String>> buildAndDivideKeyValues(String keyPrefix,
                                                                  List<String>[] dataArray) {
        keyPrefix = "{" + Utils.helper().getRedisKeyPrefix() + keyPrefix;
        HashMap<Character, ArrayList<String>> keyValueSlots = new HashMap<>();
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                String id = data.get(0);
                char lastChar = id.charAt(id.length() - 1);
                String key = keyPrefix + lastChar + "}:" + data.get(0);
                if (!keyValueSlots.containsKey(lastChar)) {
                    keyValueSlots.put(lastChar, new ArrayList<>());
                }
                keyValueSlots.get(lastChar).add(key);
                keyValueSlots.get(lastChar).add(data.get(1));
            }
        }
        return keyValueSlots.values();
    }
}
