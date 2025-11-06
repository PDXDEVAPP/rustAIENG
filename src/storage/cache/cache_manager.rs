use std::sync::Arc;
use std::time::{SystemTime, Duration};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error, instrument};

use super::cache_config::{CacheConfig, CacheStrategy, CacheTier};
use super::memory_cache::{MemoryCache, CacheStats as MemoryCacheStats};
use super::file_cache::{FileCache, CacheStats as FileCacheStats};
use super::distributed_cache::{DistributedCache, CacheStats as DistributedCacheStats};

/// Unified cache manager that coordinates multiple cache layers
#[derive(Debug)]
pub struct CacheManager {
    config: CacheConfig,
    memory_cache: Arc<MemoryCache>,
    file_cache: Arc<FileCache>,
    distributed_cache: Arc<DistributedCache>,
    strategy: CacheStrategy,
    stats: RwLock<CacheManagerStats>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheManagerStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tier_hits: HashMap<String, u64>, // "memory", "file", "distributed"
    pub tier_misses: HashMap<String, u64>,
    pub average_response_time_ms: f64,
    pub start_time: SystemTime,
    pub memory_cache_stats: Option<MemoryCacheStats>,
    pub file_cache_stats: Option<FileCacheStats>,
    pub distributed_cache_stats: Option<DistributedCacheStats>,
}

impl CacheManager {
    /// Create a new cache manager instance
    pub async fn new(config: &CacheConfig) -> Result<Self> {
        if !config.global.enable_cache {
            warn!("Cache system is disabled globally");
        }

        // Initialize cache layers based on configuration
        let memory_cache = if config.global.strategy.uses_memory() {
            Arc::new(MemoryCache::new(config).await
                .context("Failed to initialize memory cache")?)
        } else {
            Arc::new(MemoryCache::new(config).await
                .unwrap_or_else(|e| {
                    warn!("Failed to initialize memory cache: {}", e);
                    panic!("Memory cache initialization failed");
                }))
        };

        let file_cache = if config.global.strategy.uses_file() {
            Arc::new(FileCache::new(config).await
                .context("Failed to initialize file cache")?)
        } else {
            Arc::new(FileCache::new(config).await
                .unwrap_or_else(|e| {
                    warn!("Failed to initialize file cache: {}", e);
                    panic!("File cache initialization failed");
                }))
        };

        let distributed_cache = if config.global.strategy.uses_distributed() {
            Arc::new(DistributedCache::new(config).await
                .unwrap_or_else(|e| {
                    warn!("Failed to initialize distributed cache: {}", e);
                    panic!("Distributed cache initialization failed");
                }))
        } else {
            Arc::new(DistributedCache::new(config).await
                .unwrap_or_else(|e| {
                    warn!("Failed to initialize distributed cache: {}", e);
                    panic!("Distributed cache initialization failed");
                }))
        };

        let stats = CacheManagerStats {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            tier_hits: HashMap::new(),
            tier_misses: HashMap::new(),
            average_response_time_ms: 0.0,
            start_time: SystemTime::now(),
            memory_cache_stats: None,
            file_cache_stats: None,
            distributed_cache_stats: None,
        };

        let cache_manager = Self {
            config: config.clone(),
            memory_cache,
            file_cache,
            distributed_cache,
            strategy: config.global.strategy.clone(),
            stats: RwLock::new(stats),
        };

        // Initialize stats tracking
        cache_manager.update_stats().await;

        info!("Cache manager initialized with strategy: {:?}", cache_manager.strategy);

        Ok(cache_manager)
    }

    /// Store data in cache using the configured strategy
    #[instrument(skip(self, data))]
    pub async fn put(&self, key: &str, data: &[u8], ttl: Option<Duration>, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let start_time = std::time::Instant::now();

        let strategy_tiers = self.strategy.get_priority_order();

        for tier in &strategy_tiers {
            match tier {
                CacheTier::Memory => {
                    if self.config.global.strategy.uses_memory() {
                        if let Err(e) = self.memory_cache.put(key, data, ttl, metadata.clone()).await {
                            warn!("Failed to store in memory cache: {}", e);
                        }
                    }
                }
                CacheTier::File => {
                    if self.config.global.strategy.uses_file() {
                        if let Err(e) = self.file_cache.put(key, data, ttl, metadata.clone()).await {
                            warn!("Failed to store in file cache: {}", e);
                        }
                    }
                }
                CacheTier::Distributed => {
                    if self.config.global.strategy.uses_distributed() {
                        if let Err(e) = self.distributed_cache.put(key, data, ttl, metadata.clone()).await {
                            warn!("Failed to store in distributed cache: {}", e);
                        }
                    }
                }
            }
        }

        let response_time = start_time.elapsed();
        debug!("Cache PUT operation took: {:?}", response_time);
        Ok(())
    }

    /// Retrieve data from cache using the configured strategy
    #[instrument(skip(self))]
    pub async fn get(&self, key: &str) -> Result<Option<(Vec<u8>, HashMap<String, String>)>> {
        let start_time = std::time::Instant::now();
        
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        let strategy_tiers = self.strategy.get_priority_order();

        for tier in &strategy_tiers {
            let tier_name = format!("{:?}", tier).to_lowercase();
            
            match tier {
                CacheTier::Memory => {
                    if self.config.global.strategy.uses_memory() {
                        match self.memory_cache.get(key).await {
                            Ok(Some(result)) => {
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.cache_hits += 1;
                                    stats.tier_hits.entry("memory".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                let response_time = start_time.elapsed();
                                debug!("Cache HIT (memory): key={}, response_time: {:?}", key, response_time);
                                return Ok(Some(result));
                            }
                            Ok(None) => {
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.tier_misses.entry("memory".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                continue;
                            }
                            Err(e) => {
                                warn!("Memory cache lookup failed: {}", e);
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.tier_misses.entry("memory".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                continue;
                            }
                        }
                    }
                }
                CacheTier::File => {
                    if self.config.global.strategy.uses_file() {
                        match self.file_cache.get(key).await {
                            Ok(Some(result)) => {
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.cache_hits += 1;
                                    stats.tier_hits.entry("file".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                let response_time = start_time.elapsed();
                                debug!("Cache HIT (file): key={}, response_time: {:?}", key, response_time);
                                return Ok(Some(result));
                            }
                            Ok(None) => {
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.tier_misses.entry("file".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                continue;
                            }
                            Err(e) => {
                                warn!("File cache lookup failed: {}", e);
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.tier_misses.entry("file".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                continue;
                            }
                        }
                    }
                }
                CacheTier::Distributed => {
                    if self.config.global.strategy.uses_distributed() {
                        match self.distributed_cache.get(key).await {
                            Ok(Some(result)) => {
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.cache_hits += 1;
                                    stats.tier_hits.entry("distributed".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                let response_time = start_time.elapsed();
                                debug!("Cache HIT (distributed): key={}, response_time: {:?}", key, response_time);
                                return Ok(Some(result));
                            }
                            Ok(None) => {
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.tier_misses.entry("distributed".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                continue;
                            }
                            Err(e) => {
                                warn!("Distributed cache lookup failed: {}", e);
                                {
                                    let mut stats = self.stats.write().await;
                                    stats.tier_misses.entry("distributed".to_string()).and_modify(|c| *c += 1).or_insert(1);
                                }
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Cache miss - update stats
        {
            let mut stats = self.stats.write().await;
            stats.cache_misses += 1;
        }

        let response_time = start_time.elapsed();
        debug!("Cache MISS: key={}, response_time: {:?}", key, response_time);
        Ok(None)
    }

    /// Get only the data part (without metadata)
    pub async fn get_data(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self.get(key).await? {
            Some((data, _)) => Ok(Some(data)),
            None => Ok(None),
        }
    }

    /// Delete data from all cache layers
    pub async fn delete(&self, key: &str) -> Result<()> {
        let strategy_tiers = self.strategy.get_priority_order();

        for tier in &strategy_tiers {
            match tier {
                CacheTier::Memory => {
                    if self.config.global.strategy.uses_memory() {
                        if let Err(e) = self.memory_cache.delete(key).await {
                            warn!("Failed to delete from memory cache: {}", e);
                        }
                    }
                }
                CacheTier::File => {
                    if self.config.global.strategy.uses_file() {
                        if let Err(e) = self.file_cache.delete(key).await {
                            warn!("Failed to delete from file cache: {}", e);
                        }
                    }
                }
                CacheTier::Distributed => {
                    if self.config.global.strategy.uses_distributed() {
                        if let Err(e) = self.distributed_cache.delete(key).await {
                            warn!("Failed to delete from distributed cache: {}", e);
                        }
                    }
                }
            }
        }

        debug!("Deleted cache entry from all tiers: key={}", key);
        Ok(())
    }

    /// Check if key exists in any cache layer
    pub async fn exists(&self, key: &str) -> bool {
        let strategy_tiers = self.strategy.get_priority_order();

        for tier in &strategy_tiers {
            match tier {
                CacheTier::Memory => {
                    if self.config.global.strategy.uses_memory() && self.memory_cache.exists(key).await {
                        return true;
                    }
                }
                CacheTier::File => {
                    if self.config.global.strategy.uses_file() && self.file_cache.exists(key).await {
                        return true;
                    }
                }
                CacheTier::Distributed => {
                    if self.config.global.strategy.uses_distributed() && self.distributed_cache.exists(key).await {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Get comprehensive cache statistics
    pub async fn get_stats(&self) -> CacheManagerStats {
        self.update_stats().await;
        self.stats.read().await.clone()
    }

    /// Clear all cache layers
    pub async fn clear(&self) -> Result<()> {
        info!("Clearing all cache layers");

        if self.config.global.strategy.uses_memory() {
            if let Err(e) = self.memory_cache.clear().await {
                warn!("Failed to clear memory cache: {}", e);
            }
        }

        if self.config.global.strategy.uses_file() {
            if let Err(e) = self.file_cache.clear().await {
                warn!("Failed to clear file cache: {}", e);
            }
        }

        if self.config.global.strategy.uses_distributed() {
            if let Err(e) = self.distributed_cache.clear().await {
                warn!("Failed to clear distributed cache: {}", e);
            }
        }

        // Reset stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests = 0;
            stats.cache_hits = 0;
            stats.cache_misses = 0;
            stats.tier_hits.clear();
            stats.tier_misses.clear();
            stats.average_response_time_ms = 0.0;
            stats.memory_cache_stats = None;
            stats.file_cache_stats = None;
            stats.distributed_cache_stats = None;
        }

        info!("All cache layers cleared");
        Ok(())
    }

    /// Perform maintenance on all cache layers
    pub async fn maintain(&self) -> Result<CacheMaintenanceResult> {
        info!("Starting cache maintenance");

        let mut total_maintenance_results = CacheMaintenanceResult::default();

        if self.config.global.strategy.uses_memory() {
            match self.memory_cache.maintain().await {
                Ok(result) => {
                    total_maintenance_results.memory_expired += result.expired_entries as u64;
                    total_maintenance_results.memory_freed += result.freed_memory as u64;
                    debug!("Memory cache maintenance completed: {} expired, {} freed", 
                           result.expired_entries, result.freed_memory);
                }
                Err(e) => {
                    warn!("Memory cache maintenance failed: {}", e);
                }
            }
        }

        if self.config.global.strategy.uses_file() {
            match self.file_cache.maintain().await {
                Ok(result) => {
                    total_maintenance_results.file_removed += result.removed_files;
                    total_maintenance_results.file_freed += result.freed_space;
                    total_maintenance_results.file_expired += result.expired_files;
                    total_maintenance_results.file_evicted += result.evicted_entries;
                    debug!("File cache maintenance completed: {} removed, {} freed, {} expired, {} evicted", 
                           result.removed_files, result.freed_space, result.expired_files, result.evicted_entries);
                }
                Err(e) => {
                    warn!("File cache maintenance failed: {}", e);
                }
            }
        }

        // Update stats after maintenance
        self.update_stats().await;

        info!("Cache maintenance completed");
        Ok(total_maintenance_results)
    }

    /// Pre-warm the cache with frequently accessed data
    pub async fn warm_cache(&self, warm_data: Vec<(String, Vec<u8>, Option<Duration>, Option<HashMap<String, String>>)>) -> Result<CacheWarmupResult> {
        info!("Starting cache warmup with {} entries", warm_data.len());

        let mut warmed_entries = 0;
        let mut failed_entries = 0;

        for (key, data, ttl, metadata) in warm_data {
            if let Err(e) = self.put(&key, &data, ttl, metadata).await {
                warn!("Failed to warm cache entry '{}': {}", key, e);
                failed_entries += 1;
            } else {
                warmed_entries += 1;
            }
        }

        let result = CacheWarmupResult {
            warmed_entries,
            failed_entries,
        };

        info!("Cache warmup completed: {} entries warmed, {} failed", 
              warmed_entries, failed_entries);

        Ok(result)
    }

    /// Get cache performance metrics
    pub async fn get_performance_metrics(&self) -> CachePerformanceMetrics {
        let stats = self.get_stats().await;
        
        let total_requests = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_requests > 0 {
            stats.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        // Calculate tier-specific hit rates
        let mut tier_hit_rates = HashMap::new();
        for tier_name in ["memory", "file", "distributed"] {
            let tier_hits = stats.tier_hits.get(tier_name).unwrap_or(&0);
            let tier_misses = stats.tier_misses.get(tier_name).unwrap_or(&0);
            let tier_total = tier_hits + tier_misses;
            
            let tier_hit_rate = if tier_total > 0 {
                *tier_hits as f64 / tier_total as f64
            } else {
                0.0
            };
            
            tier_hit_rates.insert(tier_name.to_string(), tier_hit_rate);
        }

        CachePerformanceMetrics {
            overall_hit_rate: hit_rate,
            overall_miss_rate: 1.0 - hit_rate,
            tier_hit_rates,
            total_requests,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            average_response_time_ms: stats.average_response_time_ms,
            uptime: stats.start_time.elapsed().unwrap_or_default(),
        }
    }

    /// Update internal statistics from cache layers
    async fn update_stats(&self) {
        let mut stats = self.stats.write().await;

        if self.config.global.strategy.uses_memory() {
            stats.memory_cache_stats = Some(self.memory_cache.get_stats().await);
        }

        if self.config.global.strategy.uses_file() {
            stats.file_cache_stats = Some(self.file_cache.get_stats().await);
        }

        if self.config.global.strategy.uses_distributed() {
            stats.distributed_cache_stats = Some(self.distributed_cache.get_stats().await);
        }
    }

    /// Get the cache configuration
    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    /// Test all cache connections
    pub async fn test_connections(&self) -> HashMap<String, bool> {
        let mut results = HashMap::new();

        if self.config.global.strategy.uses_memory() {
            // Memory cache is always available if configured
            results.insert("memory".to_string(), true);
        }

        if self.config.global.strategy.uses_file() {
            // File cache is always available if configured (local filesystem)
            results.insert("file".to_string(), true);
        }

        if self.config.global.strategy.uses_distributed() {
            let connected = self.distributed_cache.test_connection().await.unwrap_or(false);
            results.insert("distributed".to_string(), connected);
        }

        results
    }

    /// Force refresh cache entry from source of truth
    pub async fn refresh(&self, key: &str, source_data: &[u8], ttl: Option<Duration>, metadata: Option<HashMap<String, String>>) -> Result<()> {
        debug!("Refreshing cache entry: key={}", key);
        
        // Delete existing entries
        self.delete(key).await?;
        
        // Re-insert with fresh data
        self.put(key, source_data, ttl, metadata).await?;
        
        Ok(())
    }

    /// Get cache size information
    pub async fn get_size_info(&self) -> CacheSizeInfo {
        let mut size_info = CacheSizeInfo::default();

        if self.config.global.strategy.uses_memory() {
            let memory_usage = self.memory_cache.get_memory_usage().await;
            size_info.memory_usage = Some(memory_usage);
        }

        if self.config.global.strategy.uses_file() {
            let file_stats = self.file_cache.get_stats().await;
            size_info.file_size = Some(file_stats.total_size);
            size_info.file_count = Some(file_stats.total_files);
        }

        if self.config.global.strategy.uses_distributed() {
            if let Ok(memory_info) = self.distributed_cache.get_memory_info().await {
                size_info.distributed_memory = Some(memory_info);
            }
        }

        size_info
    }
}

/// Cache maintenance result
#[derive(Debug, Clone, Default)]
pub struct CacheMaintenanceResult {
    pub memory_expired: u64,
    pub memory_freed: u64,
    pub file_removed: u32,
    pub file_freed: u64,
    pub file_expired: u32,
    pub file_evicted: u32,
}

/// Cache warmup result
#[derive(Debug, Clone)]
pub struct CacheWarmupResult {
    pub warmed_entries: usize,
    pub failed_entries: usize,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CachePerformanceMetrics {
    pub overall_hit_rate: f64,
    pub overall_miss_rate: f64,
    pub tier_hit_rates: HashMap<String, f64>,
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_response_time_ms: f64,
    pub uptime: Duration,
}

/// Cache size information
#[derive(Debug, Clone, Default)]
pub struct CacheSizeInfo {
    pub memory_usage: Option<super::memory_cache::MemoryUsageStats>,
    pub file_size: Option<u64>,
    pub file_count: Option<usize>,
    pub distributed_memory: Option<super::distributed_cache::RedisMemoryInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_cache_manager_basic_operations() {
        let config = CacheConfig {
            global: super::super::cache_config::GlobalCacheConfig {
                enable_cache: true,
                strategy: CacheStrategy::MultiTier,
                hit_rate_threshold: 0.8,
                enable_warming: false,
                enable_stats: true,
                warming_delay: 60,
                enable_encryption: false,
                encryption_key: "".to_string(),
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig {
                max_size_mb: 10,
                max_entries: 100,
                default_ttl: 3600,
                enable_cleanup: false,
                cleanup_interval: 300,
                enable_lru: true,
                enable_stats: true,
            },
            file_cache: super::super::cache_config::FileCacheConfig {
                cache_dir: "./test_cache".to_string(),
                max_size_mb: 100,
                max_files: 1000,
                default_ttl: 86400,
                compression_level: 6,
                enable_cleanup: false,
                cleanup_interval: 3600,
                enable_compression: false,
                enable_persistence: false,
                min_free_space: 1024,
            },
            distributed_cache: super::super::cache_config::DistributedCacheConfig {
                redis_url: "redis://localhost:6379".to_string(),
                redis_prefix: "test:".to_string(),
                enable_distributed: false,
                pool_size: 10,
                timeout: 5,
                ttl: 7200,
                enable_clustering: false,
                cluster_nodes: "".to_string(),
            },
        };

        let cache_manager = CacheManager::new(&config).await.unwrap();

        // Test put and get
        let test_data = b"Hello, Cache Manager!";
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "test".to_string());
        
        cache_manager.put("test_key", test_data, None, Some(metadata.clone())).await.unwrap();
        
        let retrieved = cache_manager.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some((test_data.to_vec(), metadata)));

        // Test get_data
        let retrieved_data = cache_manager.get_data("test_key").await.unwrap();
        assert_eq!(retrieved_data, Some(test_data.to_vec()));

        // Test exists
        assert!(cache_manager.exists("test_key").await);
        assert!(!cache_manager.exists("nonexistent").await);

        // Test delete
        cache_manager.delete("test_key").await.unwrap();
        assert!(!cache_manager.exists("test_key").await);
    }

    #[tokio::test]
    async fn test_cache_manager_strategy() {
        let config = CacheConfig {
            global: super::super::cache_config::GlobalCacheConfig {
                enable_cache: true,
                strategy: CacheStrategy::MemoryOnly,
                hit_rate_threshold: 0.8,
                enable_warming: false,
                enable_stats: true,
                warming_delay: 60,
                enable_encryption: false,
                encryption_key: "".to_string(),
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            file_cache: super::super::cache_config::FileCacheConfig {
                cache_dir: "./test_cache".to_string(),
                max_size_mb: 100,
                max_files: 1000,
                default_ttl: 86400,
                compression_level: 6,
                enable_cleanup: false,
                cleanup_interval: 3600,
                enable_compression: false,
                enable_persistence: false,
                min_free_space: 1024,
            },
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
        };

        let cache_manager = CacheManager::new(&config).await.unwrap();

        // Test that strategy is respected
        assert!(cache_manager.strategy.uses_memory());
        assert!(!cache_manager.strategy.uses_file());
        assert!(!cache_manager.strategy.uses_distributed());
    }

    #[tokio::test]
    async fn test_cache_manager_warmup() {
        let config = CacheConfig {
            global: super::super::cache_config::GlobalCacheConfig {
                enable_cache: true,
                strategy: CacheStrategy::MultiTier,
                hit_rate_threshold: 0.8,
                enable_warming: false,
                enable_stats: true,
                warming_delay: 60,
                enable_encryption: false,
                encryption_key: "".to_string(),
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            file_cache: super::super::cache_config::FileCacheConfig {
                cache_dir: "./test_cache".to_string(),
                max_size_mb: 100,
                max_files: 1000,
                default_ttl: 86400,
                compression_level: 6,
                enable_cleanup: false,
                cleanup_interval: 3600,
                enable_compression: false,
                enable_persistence: false,
                min_free_space: 1024,
            },
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
        };

        let cache_manager = CacheManager::new(&config).await.unwrap();

        // Prepare warmup data
        let warm_data = vec![
            ("key1".to_string(), b"data1".to_vec(), None, None),
            ("key2".to_string(), b"data2".to_vec(), None, None),
            ("key3".to_string(), b"data3".to_vec(), None, None),
        ];

        // Warm the cache
        let result = cache_manager.warm_cache(warm_data).await.unwrap();
        
        // Verify results
        assert_eq!(result.warmed_entries, 3);
        assert_eq!(result.failed_entries, 0);

        // Verify entries exist
        assert!(cache_manager.exists("key1").await);
        assert!(cache_manager.exists("key2").await);
        assert!(cache_manager.exists("key3").await);
    }
}