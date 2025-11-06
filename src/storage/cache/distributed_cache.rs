use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use redis::{Client, Connection, AsyncCommands, Client as RedisClient};
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};

use super::cache_config::{CacheConfig, DistributedCacheConfig};

/// Distributed cache implementation using Redis
#[derive(Debug)]
pub struct DistributedCache {
    config: DistributedCacheConfig,
    client: Option<RedisClient>,
    connection: RwLock<Option<Connection>>,
    stats: RwLock<CacheStats>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub set_operations: u64,
    pub delete_operations: u64,
    pub connection_errors: u64,
    pub last_connection_test: Option<SystemTime>,
    pub is_connected: bool,
}

impl DistributedCache {
    /// Create a new distributed cache instance
    pub async fn new(config: &CacheConfig) -> Result<Self> {
        let distributed_cache = Self {
            config: config.distributed_cache.clone(),
            client: None,
            connection: RwLock::new(None),
            stats: RwLock::new(CacheStats {
                cache_hits: 0,
                cache_misses: 0,
                set_operations: 0,
                delete_operations: 0,
                connection_errors: 0,
                last_connection_test: None,
                is_connected: false,
            }),
        };

        if distributed_cache.config.enable_distributed {
            distributed_cache.initialize_connection().await?;
        }

        info!("Distributed cache initialized: enabled={}, url={}", 
              distributed_cache.config.enable_distributed,
              distributed_cache.config.redis_url);

        Ok(distributed_cache)
    }

    /// Initialize Redis connection
    pub async fn initialize_connection(&self) -> Result<()> {
        match RedisClient::open(&self.config.redis_url) {
            Ok(client) => {
                let mut connection = client.get_async_connection().await
                    .context("Failed to connect to Redis")?;

                // Test the connection
                let _: () = redis::cmd("PING")
                    .query_async(&mut connection)
                    .await
                    .context("Redis ping failed")?;

                {
                    let mut connection_lock = self.connection.write().await;
                    *connection_lock = Some(connection);
                }

                {
                    let mut stats = self.stats.write().await;
                    stats.is_connected = true;
                    stats.last_connection_test = Some(SystemTime::now());
                }

                info!("Successfully connected to Redis: {}", self.config.redis_url);
                Ok(())
            }
            Err(e) => {
                warn!("Failed to connect to Redis: {}", e);
                
                let mut stats = self.stats.write().await;
                stats.connection_errors += 1;
                stats.is_connected = false;

                if !self.config.enable_distributed {
                    // If distributed caching is disabled, this is not an error
                    Ok(())
                } else {
                    Err(anyhow!("Failed to initialize Redis connection: {}", e))
                }
            }
        }
    }

    /// Store data in distributed cache
    pub async fn put(&self, key: &str, data: &[u8], ttl: Option<Duration>, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let redis_key = format!("{}{}", self.config.redis_prefix, key);
        let redis_key_meta = format!("{}meta:{}", self.config.redis_prefix, key);

        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                if !self.config.enable_distributed {
                    debug!("Distributed cache is disabled, skipping put operation");
                    return Ok(());
                }
                return Err(anyhow!("No Redis connection available"));
            }
        };

        // Store main data
        let result: Result<(), redis::RedisError> = redis::pipe()
            .set(&redis_key, data)
            .ignore()
            .cmd("EXPIRE")
            .arg(&redis_key)
            .arg(ttl.map(|d| d.as_secs()).unwrap_or(self.config.ttl))
            .ignore()
            .query_async(&mut *conn)
            .await;

        match result {
            Ok(_) => {
                // Store metadata separately if provided
                if let Some(meta) = metadata {
                    let meta_json = serde_json::to_string(&meta)
                        .context("Failed to serialize metadata")?;
                    
                    let _: () = redis::pipe()
                        .set(&redis_key_meta, meta_json)
                        .ignore()
                        .cmd("EXPIRE")
                        .arg(&redis_key_meta)
                        .arg(ttl.map(|d| d.as_secs()).unwrap_or(self.config.ttl))
                        .ignore()
                        .query_async(&mut *conn)
                        .await
                        .unwrap_or_else(|e| warn!("Failed to store metadata: {}", e));
                }

                {
                    let mut stats = self.stats.write().await;
                    stats.set_operations += 1;
                }

                debug!("Stored distributed cache entry: key={}, size={} bytes", key, data.len());
                Ok(())
            }
            Err(e) => {
                warn!("Failed to store distributed cache entry: {}", e);
                Err(anyhow!("Redis SET failed: {}", e))
            }
        }
    }

    /// Retrieve data from distributed cache
    pub async fn get(&self, key: &str) -> Result<Option<(Vec<u8>, HashMap<String, String>)>> {
        let redis_key = format!("{}{}", self.config.redis_prefix, key);
        let redis_key_meta = format!("{}meta:{}", self.config.redis_prefix, key);

        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                if !self.config.enable_distributed {
                    debug!("Distributed cache is disabled, returning None");
                    return Ok(None);
                }
                return Err(anyhow!("No Redis connection available"));
            }
        };

        // Get main data
        match redis::cmd("GET")
            .arg(&redis_key)
            .query_async::<_, Option<Vec<u8>>>(&mut *conn)
            .await
        {
            Ok(Some(data)) => {
                // Get metadata
                let metadata = match redis::cmd("GET")
                    .arg(&redis_key_meta)
                    .query_async::<_, Option<String>>(&mut *conn)
                    .await
                {
                    Ok(Some(meta_json)) => {
                        match serde_json::from_str::<HashMap<String, String>>(&meta_json) {
                            Ok(meta) => meta,
                            Err(_) => HashMap::new(),
                        }
                    }
                    _ => HashMap::new(),
                };

                {
                    let mut stats = self.stats.write().await;
                    stats.cache_hits += 1;
                }

                debug!("Cache hit (distributed): key={}, size={} bytes", key, data.len());
                Ok(Some((data, metadata)))
            }
            Ok(None) => {
                {
                    let mut stats = self.stats.write().await;
                    stats.cache_misses += 1;
                }
                Ok(None)
            }
            Err(e) => {
                warn!("Failed to get distributed cache entry: {}", e);
                Err(anyhow!("Redis GET failed: {}", e))
            }
        }
    }

    /// Get only the data part (without metadata)
    pub async fn get_data(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self.get(key).await? {
            Some((data, _)) => Ok(Some(data)),
            None => Ok(None),
        }
    }

    /// Delete an entry from distributed cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        let redis_key = format!("{}{}", self.config.redis_prefix, key);
        let redis_key_meta = format!("{}meta:{}", self.config.redis_prefix, key);

        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                if !self.config.enable_distributed {
                    debug!("Distributed cache is disabled, skipping delete operation");
                    return Ok(());
                }
                return Err(anyhow!("No Redis connection available"));
            }
        };

        let result: Result<(), redis::RedisError> = redis::pipe()
            .del(&redis_key)
            .ignore()
            .del(&redis_key_meta)
            .ignore()
            .query_async(&mut *conn)
            .await;

        match result {
            Ok(_) => {
                {
                    let mut stats = self.stats.write().await;
                    stats.delete_operations += 1;
                }
                Ok(())
            }
            Err(e) => {
                warn!("Failed to delete distributed cache entry: {}", e);
                Err(anyhow!("Redis DEL failed: {}", e))
            }
        }
    }

    /// Check if key exists in distributed cache
    pub async fn exists(&self, key: &str) -> bool {
        let redis_key = format!("{}{}", self.config.redis_prefix, key);

        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                return false;
            }
        };

        match redis::cmd("EXISTS")
            .arg(&redis_key)
            .query_async::<_, bool>(&mut *conn)
            .await
        {
            Ok(exists) => exists,
            Err(_) => false,
        }
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Test Redis connection
    pub async fn test_connection(&self) -> Result<bool> {
        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                return Ok(false);
            }
        };

        match redis::cmd("PING")
            .query_async::<_, String>(&mut *conn)
            .await
        {
            Ok(_) => {
                let mut stats = self.stats.write().await;
                stats.last_connection_test = Some(SystemTime::now());
                stats.is_connected = true;
                Ok(true)
            }
            Err(e) => {
                warn!("Redis connection test failed: {}", e);
                let mut stats = self.stats.write().await;
                stats.connection_errors += 1;
                stats.is_connected = false;
                Ok(false)
            }
        }
    }

    /// Clear all cache entries (use with caution)
    pub async fn clear(&self) -> Result<()> {
        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                if !self.config.enable_distributed {
                    debug!("Distributed cache is disabled, skipping clear operation");
                    return Ok(());
                }
                return Err(anyhow!("No Redis connection available"));
            }
        };

        let pattern = format!("{}*", self.config.redis_prefix);

        match redis::cmd("EVAL")
            .arg(r#"
                local keys = redis.call('KEYS', ARGV[1])
                for i=1,#keys,5000 do
                    redis.call('DEL', unpack(keys, i, math.min(i+4999, #keys)))
                end
                return #keys
            "#)
            .arg(0) // Number of keys
            .arg(&pattern)
            .query_async::<_, u64>(&mut *conn)
            .await
        {
            Ok(_deleted_count) => {
                info!("Distributed cache cleared with prefix: {}", self.config.redis_prefix);
                Ok(())
            }
            Err(e) => {
                warn!("Failed to clear distributed cache: {}", e);
                Err(anyhow!("Redis CLEAR failed: {}", e))
            }
        }
    }

    /// Get all keys matching a pattern
    pub async fn get_keys(&self, pattern: Option<&str>) -> Result<Vec<String>> {
        let search_pattern = pattern.unwrap_or("*");
        let full_pattern = format!("{}{}", self.config.redis_prefix, search_pattern);

        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                if !self.config.enable_distributed {
                    return Ok(Vec::new());
                }
                return Err(anyhow!("No Redis connection available"));
            }
        };

        match redis::cmd("KEYS")
            .arg(&full_pattern)
            .query_async::<_, Vec<String>>(&mut *conn)
            .await
        {
            Ok(keys) => {
                // Remove prefix from keys
                let keys_without_prefix: Vec<String> = keys
                    .iter()
                    .filter_map(|key| {
                        key.strip_prefix(&self.config.redis_prefix)
                            .map(|s| s.to_string())
                    })
                    .collect();
                
                Ok(keys_without_prefix)
            }
            Err(e) => {
                warn!("Failed to get keys from distributed cache: {}", e);
                Err(anyhow!("Redis KEYS failed: {}", e))
            }
        }
    }

    /// Get memory usage information from Redis
    pub async fn get_memory_info(&self) -> Result<RedisMemoryInfo> {
        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                if !self.config.enable_distributed {
                    return Ok(RedisMemoryInfo {
                        used_memory: 0,
                        used_memory_human: "0B".to_string(),
                        used_memory_rss: 0,
                        maxmemory: 0,
                        mem_fragmentation_ratio: 0.0,
                    });
                }
                return Err(anyhow!("No Redis connection available"));
            }
        };

        let memory_info: HashMap<String, String> = redis::cmd("INFO")
            .arg("memory")
            .query_async(&mut *conn)
            .await
            .context("Failed to get Redis memory info")?;

        Ok(RedisMemoryInfo {
            used_memory: memory_info.get("used_memory")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            used_memory_human: memory_info.get("used_memory_human")
                .cloned()
                .unwrap_or_else(|| "0B".to_string()),
            used_memory_rss: memory_info.get("used_memory_rss")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            maxmemory: memory_info.get("maxmemory")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            mem_fragmentation_ratio: memory_info.get("mem_fragmentation_ratio")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
        })
    }

    /// Execute a custom Redis command
    pub async fn execute_command(&self, command: &str, args: &[&str]) -> Result<String> {
        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                return Err(anyhow!("No Redis connection available"));
            }
        };

        let mut cmd = redis::cmd(command);
        for arg in args {
            cmd = cmd.arg(arg);
        }

        match cmd.query_async::<_, String>(&mut *conn).await {
            Ok(result) => Ok(result),
            Err(e) => {
                warn!("Failed to execute Redis command '{}': {}", command, e);
                Err(anyhow!("Redis command failed: {}", e))
            }
        }
    }

    /// Get Redis server info
    pub async fn get_server_info(&self) -> Result<RedisServerInfo> {
        let connection = self.connection.read().await;
        let mut conn = match connection.as_ref() {
            Some(conn) => conn,
            None => {
                return Err(anyhow!("No Redis connection available"));
            }
        };

        let server_info: HashMap<String, String> = redis::cmd("INFO")
            .query_async(&mut *conn)
            .await
            .context("Failed to get Redis server info")?;

        Ok(RedisServerInfo {
            redis_version: server_info.get("redis_version")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
            used_memory: server_info.get("used_memory")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            connected_clients: server_info.get("connected_clients")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            total_connections_received: server_info.get("total_connections_received")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            total_commands_processed: server_info.get("total_commands_processed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            keyspace_hits: server_info.get("keyspace_hits")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            keyspace_misses: server_info.get("keyspace_misses")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        })
    }
}

/// Redis memory information
#[derive(Debug, Clone)]
pub struct RedisMemoryInfo {
    pub used_memory: u64,
    pub used_memory_human: String,
    pub used_memory_rss: u64,
    pub maxmemory: u64,
    pub mem_fragmentation_ratio: f64,
}

/// Redis server information
#[derive(Debug, Clone)]
pub struct RedisServerInfo {
    pub redis_version: String,
    pub used_memory: u64,
    pub connected_clients: u64,
    pub total_connections_received: u64,
    pub total_commands_processed: u64,
    pub keyspace_hits: u64,
    pub keyspace_misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    #[ignore] // Requires Redis to be running
    async fn test_distributed_cache_basic_operations() {
        let config = CacheConfig {
            distributed_cache: DistributedCacheConfig {
                redis_url: "redis://localhost:6379".to_string(),
                redis_prefix: "test:".to_string(),
                enable_distributed: true,
                pool_size: 10,
                timeout: 5,
                ttl: 3600,
                enable_clustering: false,
                cluster_nodes: "".to_string(),
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = DistributedCache::new(&config).await;

        // Skip test if Redis is not available
        if cache.is_err() {
            return;
        }

        let cache = cache.unwrap();

        // Test put and get
        let test_data = b"Hello, Redis!";
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "test".to_string());
        
        cache.put("test_key", test_data, None, Some(metadata.clone())).await.unwrap();
        
        let retrieved = cache.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some((test_data.to_vec(), metadata)));

        // Test exists
        assert!(cache.exists("test_key").await);
        assert!(!cache.exists("nonexistent").await);

        // Test delete
        cache.delete("test_key").await.unwrap();
        assert!(!cache.exists("test_key").await);
    }

    #[tokio::test]
    async fn test_distributed_cache_disabled() {
        let config = CacheConfig {
            distributed_cache: DistributedCacheConfig {
                redis_url: "redis://localhost:6379".to_string(),
                redis_prefix: "test:".to_string(),
                enable_distributed: false,
                pool_size: 10,
                timeout: 5,
                ttl: 3600,
                enable_clustering: false,
                cluster_nodes: "".to_string(),
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = DistributedCache::new(&config).await.unwrap();

        // Operations should be no-ops when disabled
        let test_data = b"test";
        cache.put("test_key", test_data, None, None).await.unwrap();
        let result = cache.get("test_key").await.unwrap();
        assert_eq!(result, None);
    }
}