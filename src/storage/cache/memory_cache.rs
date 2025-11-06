use std::collections::{HashMap, BTreeMap, LinkedList};
use std::time::{SystemTime, Duration, Instant};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{info, warn, debug, error};
use dashmap::DashMap;
use lru::LruCache;

use super::cache_config::{CacheConfig, MemoryCacheConfig};

/// Memory-based cache implementation with LRU eviction
#[derive(Debug)]
pub struct MemoryCache {
    config: MemoryCacheConfig,
    // Use LruCache for automatic LRU eviction
    cache: RwLock<LruCache<String, CacheEntry>>,
    // Also maintain a separate map for custom metadata access
    metadata_map: DashMap<String, HashMap<String, String>>,
    stats: RwLock<CacheStats>,
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    created_at: SystemTime,
    last_accessed: SystemTime,
    expires_at: Option<SystemTime>,
    access_count: u64,
    size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_size: usize,
    pub total_entries: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub expired_entries: u64,
    pub last_cleanup: Option<SystemTime>,
    pub memory_usage_mb: f64,
}

impl MemoryCache {
    /// Create a new memory cache instance
    pub async fn new(config: &CacheConfig) -> Result<Self> {
        let cache = LruCache::new(config.memory_cache.max_entries);
        let stats = CacheStats {
            total_size: 0,
            total_entries: 0,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            expired_entries: 0,
            last_cleanup: None,
            memory_usage_mb: 0.0,
        };

        let memory_cache = Self {
            config: config.memory_cache.clone(),
            cache: RwLock::new(cache),
            metadata_map: DashMap::new(),
            stats: RwLock::new(stats),
            cleanup_task: None,
        };

        // Start cleanup task if enabled
        if memory_cache.config.enable_cleanup {
            memory_cache.start_cleanup_task().await;
        }

        info!("Memory cache initialized: max_entries={}, max_size_mb={}", 
              memory_cache.config.max_entries,
              memory_cache.config.max_size_mb);

        Ok(memory_cache)
    }

    /// Store data in cache
    pub async fn put(&self, key: &str, data: &[u8], ttl: Option<Duration>, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let key = key.to_string();
        let size = data.len();
        let now = SystemTime::now();
        let expires_at = ttl.map(|duration| now + duration);

        // Create cache entry
        let entry = CacheEntry {
            data: data.to_vec(),
            created_at: now,
            last_accessed: now,
            expires_at,
            access_count: 1,
            size,
        };

        // Check size limits
        self.check_size_limits(size).await?;

        // Insert into cache
        let (evicted_key, evicted_entry) = {
            let mut cache = self.cache.write().await;
            cache.push(key.clone(), entry.clone())
        };

        // Handle eviction
        if let Some((evicted_key, evicted_entry)) = evicted_key {
            self.handle_eviction(&evicted_key, evicted_entry).await;
        }

        // Store metadata
        if let Some(meta) = metadata {
            self.metadata_map.insert(key.clone(), meta);
        }

        // Update stats
        let old_size = if let Some((_, old_entry)) = &evicted_entry {
            old_entry.size
        } else {
            0
        };

        {
            let mut stats = self.stats.write().await;
            stats.total_size += size - old_size;
            stats.total_entries = self.cache.read().await.len();
            stats.memory_usage_mb = stats.total_size as f64 / (1024.0 * 1024.0);
        }

        debug!("Stored cache entry: key={}, size={} bytes", key, size);
        Ok(())
    }

    /// Retrieve data from cache
    pub async fn get(&self, key: &str) -> Result<Option<(Vec<u8>, HashMap<String, String>)>> {
        let cache_entry = {
            let mut cache = self.cache.write().await;
            
            match cache.get_mut(key) {
                Some(entry) => {
                    // Check if entry has expired
                    if let Some(expires_at) = entry.expires_at {
                        if SystemTime::now() > expires_at {
                            cache.pop(key);
                            drop(cache); // Release lock before async operation
                            self.delete(key).await?;
                            let mut stats = self.stats.write().await;
                            stats.cache_misses += 1;
                            stats.expired_entries += 1;
                            return Ok(None);
                        }
                    }

                    // Update access info
                    entry.last_accessed = SystemTime::now();
                    entry.access_count += 1;
                    
                    Some(entry.clone())
                }
                None => None,
            }
        };

        match cache_entry {
            Some(entry) => {
                // Get metadata
                let metadata = self.metadata_map.get(key)
                    .map(|entry| entry.clone())
                    .unwrap_or_default();

                // Update stats
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;

                debug!("Cache hit: key={}, size={} bytes", key, entry.data.len());
                Ok(Some((entry.data, metadata)))
            }
            None => {
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;
                Ok(None)
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

    /// Delete an entry from cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        let removed_entry = {
            let mut cache = self.cache.write().await;
            cache.pop(key).map(|(_, entry)| entry)
        };

        // Remove metadata
        self.metadata_map.remove(key);

        // Update stats
        if let Some(entry) = removed_entry {
            let mut stats = self.stats.write().await;
            stats.total_size -= entry.size;
            stats.total_entries = self.cache.read().await.len();
            stats.memory_usage_mb = stats.total_size as f64 / (1024.0 * 1024.0);
        }

        debug!("Deleted cache entry: key={}", key);
        Ok(())
    }

    /// Check if key exists in cache
    pub async fn exists(&self, key: &str) -> bool {
        let cache = self.cache.read().await;
        cache.contains(key)
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        {
            let mut cache = self.cache.write().await;
            cache.clear();
        }

        self.metadata_map.clear();

        let mut stats = self.stats.write().await;
        stats.total_size = 0;
        stats.total_entries = 0;
        stats.memory_usage_mb = 0.0;

        info!("Memory cache cleared");
        Ok(())
    }

    /// Perform cache maintenance
    pub async fn maintain(&self) -> Result<MaintenanceResult> {
        let mut expired_entries = 0;
        let mut freed_memory = 0usize;

        let expired_keys = {
            let cache = self.cache.read().await;
            let now = SystemTime::now();
            
            cache.iter()
                .filter_map(|(key, entry)| {
                    if let Some(expires_at) = entry.expires_at {
                        if now > expires_at {
                            Some(key.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };

        // Remove expired entries
        for key in expired_keys {
            if let Some(entry) = {
                let mut cache = self.cache.write().await;
                cache.pop(&key).map(|(_, entry)| entry)
            } {
                self.metadata_map.remove(&key);
                expired_entries += 1;
                freed_memory += entry.size;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_size -= freed_memory;
            stats.total_entries = self.cache.read().await.len();
            stats.expired_entries += expired_entries;
            stats.last_cleanup = Some(SystemTime::now());
            stats.memory_usage_mb = stats.total_size as f64 / (1024.0 * 1024.0);
        }

        Ok(MaintenanceResult {
            expired_entries,
            freed_memory,
        })
    }

    /// Get all keys in cache
    pub async fn get_keys(&self) -> Vec<String> {
        let cache = self.cache.read().await;
        cache.iter().map(|(key, _)| key.clone()).collect()
    }

    /// Get cache entries sorted by access time (for debugging)
    pub async fn get_entries_by_access_time(&self) -> Vec<(String, SystemTime, u64)> {
        let cache = self.cache.read().await;
        cache.iter()
            .map(|(key, entry)| (key.clone(), entry.last_accessed, entry.access_count))
            .collect()
    }

    /// Get cache entries sorted by creation time
    pub async fn get_entries_by_creation_time(&self) -> Vec<(String, SystemTime, usize)> {
        let cache = self.cache.read().await;
        cache.iter()
            .map(|(key, entry)| (key.clone(), entry.created_at, entry.size))
            .collect()
    }

    /// Pre-warm cache with data
    pub async fn warm_cache(&self, data: Vec<(String, Vec<u8>, Option<Duration>, Option<HashMap<String, String>>)>) -> Result<usize> {
        let mut warmed = 0;
        
        for (key, value, ttl, metadata) in data {
            if let Err(e) = self.put(&key, &value, ttl, metadata).await {
                warn!("Failed to warm cache entry '{}': {}", key, e);
            } else {
                warmed += 1;
            }
        }

        info!("Warmed cache with {} entries", warmed);
        Ok(warmed)
    }

    /// Start background cleanup task
    async fn start_cleanup_task(&self) {
        let interval_secs = self.config.cleanup_interval;
        let memory_cache = self.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_secs));
            
            loop {
                interval.tick().await;
                if let Err(e) = memory_cache.maintain().await {
                    error!("Memory cache maintenance failed: {}", e);
                }
            }
        });

        self.cleanup_task = Some(task);
    }

    /// Check if new entry would exceed size limits
    async fn check_size_limits(&self, new_entry_size: usize) -> Result<()> {
        let stats = self.stats.read().await;
        let max_size_bytes = self.config.max_size_mb * 1024 * 1024;
        
        if stats.total_size + new_entry_size > max_size_bytes {
            // Need to evict entries to make room
            let mut cache = self.cache.write().await;
            let mut freed_space = 0usize;
            
            // Evict least recently used entries
            while stats.total_size + new_entry_size - freed_space > max_size_bytes && !cache.is_empty() {
                if let Some((_key, entry)) = cache.pop_lru() {
                    freed_space += entry.size;
                    self.handle_eviction(&_key, entry).await;
                } else {
                    break;
                }
            }

            // Check if we have enough space now
            if stats.total_size + new_entry_size - freed_space > max_size_bytes {
                return Err(anyhow!("Not enough cache space even after eviction"));
            }
        }

        Ok(())
    }

    /// Handle cache eviction
    async fn handle_eviction(&self, _key: &str, entry: CacheEntry) {
        let mut stats = self.stats.write().await;
        stats.total_size -= entry.size;
        stats.evictions += 1;
        stats.memory_usage_mb = stats.total_size as f64 / (1024.0 * 1024.0);
        
        debug!("Evicted cache entry, freed {} bytes", entry.size);
    }

    /// Get memory usage statistics
    pub async fn get_memory_usage(&self) -> MemoryUsageStats {
        let stats = self.stats.read().await;
        
        MemoryUsageStats {
            current_size_bytes: stats.total_size,
            current_size_mb: stats.memory_usage_mb,
            max_size_bytes: self.config.max_size_mb * 1024 * 1024,
            usage_percentage: (stats.total_size as f64 / (self.config.max_size_mb * 1024 * 1024) as f64) * 100.0,
            entry_count: stats.total_entries,
            max_entries: self.config.max_entries,
            entry_usage_percentage: (stats.total_entries as f64 / self.config.max_entries as f64) * 100.0,
        }
    }

    /// Get cache efficiency metrics
    pub async fn get_efficiency_metrics(&self) -> EfficiencyMetrics {
        let stats = self.stats.read().await;
        
        let total_requests = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_requests > 0 {
            stats.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        EfficiencyMetrics {
            hit_rate,
            miss_rate: 1.0 - hit_rate,
            avg_access_count: if stats.total_entries > 0 {
                // This is an approximation since we don't track total access count
                stats.cache_hits as f64 / stats.total_entries as f64
            } else {
                0.0
            },
            eviction_rate: if stats.cache_hits + stats.evictions > 0 {
                stats.evictions as f64 / (stats.cache_hits + stats.evictions) as f64
            } else {
                0.0
            },
        }
    }
}

impl Clone for MemoryCache {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cache: self.cache.clone(),
            metadata_map: self.metadata_map.clone(),
            stats: self.stats.clone(),
            cleanup_task: None, // Don't clone the task handle
        }
    }
}

impl Drop for MemoryCache {
    fn drop(&mut self) {
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
        }
    }
}

/// Maintenance result
#[derive(Debug, Clone)]
pub struct MaintenanceResult {
    pub expired_entries: u32,
    pub freed_memory: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub current_size_bytes: usize,
    pub current_size_mb: f64,
    pub max_size_bytes: usize,
    pub usage_percentage: f64,
    pub entry_count: usize,
    pub max_entries: usize,
    pub entry_usage_percentage: f64,
}

/// Cache efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub avg_access_count: f64,
    pub eviction_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_memory_cache_basic_operations() {
        let config = CacheConfig {
            memory_cache: MemoryCacheConfig {
                max_size_mb: 1,
                max_entries: 100,
                default_ttl: 3600,
                enable_cleanup: false,
                cleanup_interval: 300,
                enable_lru: true,
                enable_stats: true,
            },
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = MemoryCache::new(&config).await.unwrap();

        // Test put and get
        let test_data = b"Hello, World!";
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "test".to_string());
        
        cache.put("test_key", test_data, None, Some(metadata.clone())).await.unwrap();
        
        let retrieved = cache.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some((test_data.to_vec(), metadata)));

        // Test get_data
        let retrieved_data = cache.get_data("test_key").await.unwrap();
        assert_eq!(retrieved_data, Some(test_data.to_vec()));

        // Test exists
        assert!(cache.exists("test_key").await);
        assert!(!cache.exists("nonexistent").await);

        // Test delete
        cache.delete("test_key").await.unwrap();
        assert!(!cache.exists("test_key").await);
    }

    #[tokio::test]
    async fn test_memory_cache_ttl() {
        let config = CacheConfig {
            memory_cache: MemoryCacheConfig {
                max_size_mb: 1,
                max_entries: 100,
                default_ttl: 1, // 1 second
                enable_cleanup: false,
                cleanup_interval: 300,
                enable_lru: true,
                enable_stats: true,
            },
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = MemoryCache::new(&config).await.unwrap();

        // Add entry with TTL
        let test_data = b"TTL test";
        cache.put("ttl_key", test_data, Some(Duration::from_secs(1)), None).await.unwrap();
        
        // Should exist immediately
        assert!(cache.exists("ttl_key").await);

        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should be expired now
        let result = cache.get("ttl_key").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_memory_cache_eviction() {
        let config = CacheConfig {
            memory_cache: MemoryCacheConfig {
                max_size_mb: 1, // Very small cache
                max_entries: 5, // Very few entries
                default_ttl: 3600,
                enable_cleanup: false,
                cleanup_interval: 300,
                enable_lru: true,
                enable_stats: true,
            },
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = MemoryCache::new(&config).await.unwrap();

        // Add more entries than the cache can hold
        for i in 0..10 {
            let data = format!("test_data_{}", i).into_bytes();
            cache.put(&format!("key_{}", i), &data, None, None).await.unwrap();
        }

        // Should have evicted some entries
        let stats = cache.get_stats().await;
        assert!(stats.evictions > 0);
    }

    #[tokio::test]
    async fn test_memory_cache_maintenance() {
        let config = CacheConfig {
            memory_cache: MemoryCacheConfig {
                max_size_mb: 1,
                max_entries: 100,
                default_ttl: 1,
                enable_cleanup: false,
                cleanup_interval: 1, // Frequent cleanup
                enable_lru: true,
                enable_stats: true,
            },
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = MemoryCache::new(&config).await.unwrap();

        // Add expired entries
        for i in 0..5 {
            let data = format!("expired_data_{}", i).into_bytes();
            cache.put(&format!("expired_key_{}", i), &data, Some(Duration::from_millis(1)), None).await.unwrap();
        }

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Run maintenance
        let result = cache.maintain().await.unwrap();
        assert!(result.expired_entries > 0);
        assert!(result.freed_memory > 0);
    }

    #[tokio::test]
    async fn test_cache_efficiency_metrics() {
        let config = CacheConfig {
            memory_cache: MemoryCacheConfig {
                max_size_mb: 10,
                max_entries: 100,
                default_ttl: 3600,
                enable_cleanup: false,
                cleanup_interval: 300,
                enable_lru: true,
                enable_stats: true,
            },
            file_cache: super::super::cache_config::FileCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = MemoryCache::new(&config).await.unwrap();

        // Add some entries
        for i in 0..10 {
            let data = format!("test_data_{}", i).into_bytes();
            cache.put(&format!("key_{}", i), &data, None, None).await.unwrap();
        }

        // Access some entries multiple times
        for _ in 0..5 {
            let _ = cache.get("key_1").await.unwrap();
            let _ = cache.get("key_2").await.unwrap();
        }

        // Miss some requests
        for _ in 0..3 {
            let _ = cache.get("nonexistent").await.unwrap();
        }

        let metrics = cache.get_efficiency_metrics().await;
        
        assert!(metrics.hit_rate >= 0.0 && metrics.hit_rate <= 1.0);
        assert!(metrics.miss_rate >= 0.0 && metrics.miss_rate <= 1.0);
        assert!((metrics.hit_rate + metrics.miss_rate).abs() < 0.001); // Should sum to 1.0
    }
}