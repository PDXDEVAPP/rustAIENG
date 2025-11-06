use serde::{Deserialize, Serialize};
use anyhow::{Result};
use std::path::Path;

/// Cache configuration for different storage layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// In-memory cache configuration
    pub memory_cache: MemoryCacheConfig,
    
    /// File-based cache configuration
    pub file_cache: FileCacheConfig,
    
    /// Distributed cache configuration
    pub distributed_cache: DistributedCacheConfig,
    
    /// Global cache settings
    pub global: GlobalCacheConfig,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_cache: MemoryCacheConfig::default(),
            file_cache: FileCacheConfig::default(),
            distributed_cache: DistributedCacheConfig::default(),
            global: GlobalCacheConfig::default(),
        }
    }
}

/// In-memory cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCacheConfig {
    /// Maximum memory usage for cache (MB)
    #[serde(default = "default_memory_cache_size")]
    pub max_size_mb: usize,
    
    /// Maximum number of entries in cache
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
    
    /// Default TTL for cached items (seconds)
    #[serde(default = "default_memory_ttl")]
    pub default_ttl: u64,
    
    /// Enable automatic cleanup of expired entries
    #[serde(default = "default_enable_cleanup")]
    pub enable_cleanup: bool,
    
    /// Cleanup interval (seconds)
    #[serde(default = "default_cleanup_interval")]
    pub cleanup_interval: u64,
    
    /// Enable LRU eviction
    #[serde(default = "default_enable_lru")]
    pub enable_lru: bool,
    
    /// Enable statistics tracking
    #[serde(default = "default_enable_stats")]
    pub enable_stats: bool,
}

fn default_memory_cache_size() -> usize {
    512 // 512 MB
}

fn default_max_entries() -> usize {
    10000
}

fn default_memory_ttl() -> u64 {
    3600 // 1 hour
}

fn default_enable_cleanup() -> bool {
    true
}

fn default_cleanup_interval() -> u64 {
    300 // 5 minutes
}

fn default_enable_lru() -> bool {
    true
}

fn default_enable_stats() -> bool {
    true
}

impl Default for MemoryCacheConfig {
    fn default() -> Self {
        Self {
            max_size_mb: default_memory_cache_size(),
            max_entries: default_max_entries(),
            default_ttl: default_memory_ttl(),
            enable_cleanup: default_enable_cleanup(),
            cleanup_interval: default_cleanup_interval(),
            enable_lru: default_enable_lru(),
            enable_stats: default_enable_stats(),
        }
    }
}

/// File-based cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCacheConfig {
    /// Cache directory path
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
    
    /// Maximum disk usage for cache (MB)
    #[serde(default = "default_disk_cache_size")]
    pub max_size_mb: usize,
    
    /// Maximum number of files in cache
    #[serde(default = "default_max_files")]
    pub max_files: usize,
    
    /// Default TTL for cached files (seconds)
    #[serde(default = "default_file_ttl")]
    pub default_ttl: u64,
    
    /// Compression level for cached files (0-9)
    #[serde(default = "default_compression_level")]
    pub compression_level: u8,
    
    /// Enable automatic cleanup of expired files
    #[serde(default = "default_enable_file_cleanup")]
    pub enable_cleanup: bool,
    
    /// Cleanup interval for file cache (seconds)
    #[serde(default = "default_file_cleanup_interval")]
    pub cleanup_interval: u64,
    
    /// Enable file compression
    #[serde(default = "default_enable_compression")]
    pub enable_compression: bool,
    
    /// Enable cache persistence across restarts
    #[serde(default = "default_enable_persistence")]
    pub enable_persistence: bool,
    
    /// Minimum free disk space to maintain (MB)
    #[serde(default = "default_min_free_space")]
    pub min_free_space_mb: usize,
}

fn default_cache_dir() -> String {
    "./cache".to_string()
}

fn default_disk_cache_size() -> usize {
    10240 // 10 GB
}

fn default_max_files() -> usize {
    100000
}

fn default_file_ttl() -> u64 {
    86400 // 24 hours
}

fn default_compression_level() -> u8 {
    6
}

fn default_enable_file_cleanup() -> bool {
    true
}

fn default_file_cleanup_interval() -> u64 {
    3600 // 1 hour
}

fn default_enable_compression() -> bool {
    true
}

fn default_enable_persistence() -> bool {
    true
}

fn default_min_free_space() -> usize {
    1024 // 1 GB
}

impl Default for FileCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            max_size_mb: default_disk_cache_size(),
            max_files: default_max_files(),
            default_ttl: default_file_ttl(),
            compression_level: default_compression_level(),
            enable_cleanup: default_enable_file_cleanup(),
            cleanup_interval: default_file_cleanup_interval(),
            enable_compression: default_enable_compression(),
            enable_persistence: default_enable_persistence(),
            min_free_space_mb: default_min_free_space(),
        }
    }
}

/// Distributed cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCacheConfig {
    /// Redis connection URL
    #[serde(default = "default_redis_url")]
    pub redis_url: String,
    
    /// Redis key prefix
    #[serde(default = "default_redis_prefix")]
    pub redis_prefix: String,
    
    /// Enable distributed caching
    #[serde(default = "default_enable_distributed")]
    pub enable_distributed: bool,
    
    /// Redis connection pool size
    #[serde(default = "default_redis_pool_size")]
    pub pool_size: usize,
    
    /// Redis connection timeout (seconds)
    #[serde(default = "default_redis_timeout")]
    pub timeout: u64,
    
    /// Redis TTL for cached items (seconds)
    #[serde(default = "default_redis_ttl")]
    pub ttl: u64,
    
    /// Enable Redis clustering
    #[serde(default = "default_enable_clustering")]
    pub enable_clustering: bool,
    
    /// Redis cluster nodes (comma-separated)
    #[serde(default = "default_cluster_nodes")]
    pub cluster_nodes: String,
}

fn default_redis_url() -> String {
    "redis://localhost:6379".to_string()
}

fn default_redis_prefix() -> String {
    "ollama:cache:".to_string()
}

fn default_enable_distributed() -> bool {
    false
}

fn default_redis_pool_size() -> usize {
    10
}

fn default_redis_timeout() -> u64 {
    5
}

fn default_redis_ttl() -> u64 {
    7200 // 2 hours
}

fn default_enable_clustering() -> bool {
    false
}

fn default_cluster_nodes() -> String {
    "".to_string()
}

impl Default for DistributedCacheConfig {
    fn default() -> Self {
        Self {
            redis_url: default_redis_url(),
            redis_prefix: default_redis_prefix(),
            enable_distributed: default_enable_distributed(),
            pool_size: default_redis_pool_size(),
            timeout: default_redis_timeout(),
            ttl: default_redis_ttl(),
            enable_clustering: default_enable_clustering(),
            cluster_nodes: default_cluster_nodes(),
        }
    }
}

/// Global cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCacheConfig {
    /// Enable caching system
    #[serde(default = "default_enable_cache")]
    pub enable_cache: bool,
    
    /// Default cache strategy
    #[serde(default = "default_cache_strategy")]
    pub strategy: CacheStrategy,
    
    /// Maximum cache hit rate before optimization (percentage)
    #[serde(default = "default_hit_rate_threshold")]
    pub hit_rate_threshold: f64,
    
    /// Enable cache warming on startup
    #[serde(default = "default_enable_warming")]
    pub enable_warming: bool,
    
    /// Enable cache statistics collection
    #[serde(default = "default_enable_global_stats")]
    pub enable_stats: bool,
    
    /// Cache warming delay (seconds)
    #[serde(default = "default_warming_delay")]
    pub warming_delay: u64,
    
    /// Enable cache encryption
    #[serde(default = "default_enable_encryption")]
    pub enable_encryption: bool,
    
    /// Cache encryption key (base64 encoded)
    #[serde(default = "default_encryption_key")]
    pub encryption_key: String,
}

fn default_enable_cache() -> bool {
    true
}

fn default_cache_strategy() -> CacheStrategy {
    CacheStrategy::MultiTier
}

fn default_hit_rate_threshold() -> f64 {
    0.8 // 80%
}

fn default_enable_warming() -> bool {
    false
}

fn default_enable_global_stats() -> bool {
    true
}

fn default_warming_delay() -> u64 {
    60
}

fn default_enable_encryption() -> bool {
    false
}

fn default_encryption_key() -> String {
    "".to_string()
}

impl Default for GlobalCacheConfig {
    fn default() -> Self {
        Self {
            enable_cache: default_enable_cache(),
            strategy: default_cache_strategy(),
            hit_rate_threshold: default_hit_rate_threshold(),
            enable_warming: default_enable_warming(),
            enable_stats: default_enable_global_stats(),
            warming_delay: default_warming_delay(),
            enable_encryption: default_enable_encryption(),
            encryption_key: default_encryption_key(),
        }
    }
}

/// Cache strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// Use only memory cache
    MemoryOnly,
    
    /// Use only file cache
    FileOnly,
    
    /// Use only distributed cache
    DistributedOnly,
    
    /// Multi-tier cache (memory -> file -> distributed)
    MultiTier,
    
    /// Custom strategy
    Custom(String),
}

impl CacheStrategy {
    /// Check if this strategy uses memory cache
    pub fn uses_memory(&self) -> bool {
        match self {
            CacheStrategy::MemoryOnly | CacheStrategy::MultiTier => true,
            _ => false,
        }
    }

    /// Check if this strategy uses file cache
    pub fn uses_file(&self) -> bool {
        match self {
            CacheStrategy::FileOnly | CacheStrategy::MultiTier => true,
            _ => false,
        }
    }

    /// Check if this strategy uses distributed cache
    pub fn uses_distributed(&self) -> bool {
        match self {
            CacheStrategy::DistributedOnly | CacheStrategy::MultiTier => true,
            _ => false,
        }
    }

    /// Get the priority order for cache tiers
    pub fn get_priority_order(&self) -> Vec<CacheTier> {
        match self {
            CacheStrategy::MemoryOnly => vec![CacheTier::Memory],
            CacheStrategy::FileOnly => vec![CacheTier::File],
            CacheStrategy::DistributedOnly => vec![CacheTier::Distributed],
            CacheStrategy::MultiTier => vec![CacheTier::Memory, CacheTier::File, CacheTier::Distributed],
            CacheStrategy::Custom(_) => vec![CacheTier::Memory, CacheTier::File, CacheTier::Distributed],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    Memory,
    File,
    Distributed,
}

impl CacheConfig {
    /// Load configuration from file
    pub async fn from_file(config_path: &str) -> Result<Self> {
        let content = tokio::fs::read_to_string(config_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;

        let config: CacheConfig = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse configuration file: {}", e))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let config = CacheConfig {
            memory_cache: MemoryCacheConfig {
                max_size_mb: std::env::var("CACHE_MEMORY_SIZE_MB")
                    .unwrap_or_else(|_| default_memory_cache_size().to_string())
                    .parse()
                    .unwrap_or(default_memory_cache_size()),
                max_entries: std::env::var("CACHE_MEMORY_MAX_ENTRIES")
                    .unwrap_or_else(|_| default_max_entries().to_string())
                    .parse()
                    .unwrap_or(default_max_entries()),
                default_ttl: std::env::var("CACHE_MEMORY_TTL")
                    .unwrap_or_else(|_| default_memory_ttl().to_string())
                    .parse()
                    .unwrap_or(default_memory_ttl()),
                enable_cleanup: std::env::var("CACHE_MEMORY_ENABLE_CLEANUP")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_cleanup()),
                cleanup_interval: std::env::var("CACHE_MEMORY_CLEANUP_INTERVAL")
                    .unwrap_or_else(|_| default_cleanup_interval().to_string())
                    .parse()
                    .unwrap_or(default_cleanup_interval()),
                enable_lru: std::env::var("CACHE_MEMORY_ENABLE_LRU")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_lru()),
                enable_stats: std::env::var("CACHE_MEMORY_ENABLE_STATS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_stats()),
            },
            file_cache: FileCacheConfig {
                cache_dir: std::env::var("CACHE_FILE_DIR")
                    .unwrap_or_else(|_| default_cache_dir().to_string()),
                max_size_mb: std::env::var("CACHE_FILE_MAX_SIZE_MB")
                    .unwrap_or_else(|_| default_disk_cache_size().to_string())
                    .parse()
                    .unwrap_or(default_disk_cache_size()),
                max_files: std::env::var("CACHE_FILE_MAX_FILES")
                    .unwrap_or_else(|_| default_max_files().to_string())
                    .parse()
                    .unwrap_or(default_max_files()),
                default_ttl: std::env::var("CACHE_FILE_TTL")
                    .unwrap_or_else(|_| default_file_ttl().to_string())
                    .parse()
                    .unwrap_or(default_file_ttl()),
                compression_level: std::env::var("CACHE_FILE_COMPRESSION_LEVEL")
                    .unwrap_or_else(|_| default_compression_level().to_string())
                    .parse()
                    .unwrap_or(default_compression_level()),
                enable_cleanup: std::env::var("CACHE_FILE_ENABLE_CLEANUP")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_file_cleanup()),
                cleanup_interval: std::env::var("CACHE_FILE_CLEANUP_INTERVAL")
                    .unwrap_or_else(|_| default_file_cleanup_interval().to_string())
                    .parse()
                    .unwrap_or(default_file_cleanup_interval()),
                enable_compression: std::env::var("CACHE_FILE_ENABLE_COMPRESSION")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_compression()),
                enable_persistence: std::env::var("CACHE_FILE_ENABLE_PERSISTENCE")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_persistence()),
                min_free_space_mb: std::env::var("CACHE_FILE_MIN_FREE_SPACE")
                    .unwrap_or_else(|_| default_min_free_space().to_string())
                    .parse()
                    .unwrap_or(default_min_free_space()),
            },
            distributed_cache: DistributedCacheConfig {
                redis_url: std::env::var("CACHE_REDIS_URL")
                    .unwrap_or_else(|_| default_redis_url().to_string()),
                redis_prefix: std::env::var("CACHE_REDIS_PREFIX")
                    .unwrap_or_else(|_| default_redis_prefix().to_string()),
                enable_distributed: std::env::var("CACHE_DISTRIBUTED_ENABLE")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(default_enable_distributed()),
                pool_size: std::env::var("CACHE_REDIS_POOL_SIZE")
                    .unwrap_or_else(|_| default_redis_pool_size().to_string())
                    .parse()
                    .unwrap_or(default_redis_pool_size()),
                timeout: std::env::var("CACHE_REDIS_TIMEOUT")
                    .unwrap_or_else(|_| default_redis_timeout().to_string())
                    .parse()
                    .unwrap_or(default_redis_timeout()),
                ttl: std::env::var("CACHE_REDIS_TTL")
                    .unwrap_or_else(|_| default_redis_ttl().to_string())
                    .parse()
                    .unwrap_or(default_redis_ttl()),
                enable_clustering: std::env::var("CACHE_REDIS_CLUSTERING")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(default_enable_clustering()),
                cluster_nodes: std::env::var("CACHE_REDIS_CLUSTER_NODES")
                    .unwrap_or_else(|_| default_cluster_nodes().to_string()),
            },
            global: GlobalCacheConfig {
                enable_cache: std::env::var("CACHE_ENABLE")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_cache()),
                strategy: std::env::var("CACHE_STRATEGY")
                    .map(|s| match s.to_lowercase().as_str() {
                        "memory_only" => CacheStrategy::MemoryOnly,
                        "file_only" => CacheStrategy::FileOnly,
                        "distributed_only" => CacheStrategy::DistributedOnly,
                        "multi_tier" => CacheStrategy::MultiTier,
                        custom => CacheStrategy::Custom(custom.to_string()),
                    })
                    .unwrap_or(default_cache_strategy()),
                hit_rate_threshold: std::env::var("CACHE_HIT_RATE_THRESHOLD")
                    .unwrap_or_else(|_| default_hit_rate_threshold().to_string())
                    .parse()
                    .unwrap_or(default_hit_rate_threshold()),
                enable_warming: std::env::var("CACHE_ENABLE_WARMING")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(default_enable_warming()),
                enable_stats: std::env::var("CACHE_ENABLE_STATS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(default_enable_global_stats()),
                warming_delay: std::env::var("CACHE_WARMING_DELAY")
                    .unwrap_or_else(|_| default_warming_delay().to_string())
                    .parse()
                    .unwrap_or(default_warming_delay()),
                enable_encryption: std::env::var("CACHE_ENABLE_ENCRYPTION")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(default_enable_encryption()),
                encryption_key: std::env::var("CACHE_ENCRYPTION_KEY")
                    .unwrap_or_else(|_| default_encryption_key().to_string()),
            },
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate memory cache config
        if self.memory_cache.max_size_mb == 0 {
            return Err(anyhow::anyhow!("Memory cache max size must be greater than 0"));
        }

        if self.memory_cache.max_entries == 0 {
            return Err(anyhow::anyhow!("Memory cache max entries must be greater than 0"));
        }

        // Validate file cache config
        if self.file_cache.max_size_mb == 0 {
            return Err(anyhow::anyhow!("File cache max size must be greater than 0"));
        }

        if self.file_cache.max_files == 0 {
            return Err(anyhow::anyhow!("File cache max files must be greater than 0"));
        }

        if self.file_cache.compression_level > 9 {
            return Err(anyhow::anyhow!("Compression level must be between 0 and 9"));
        }

        // Validate distributed cache config
        if self.distributed_cache.pool_size == 0 {
            return Err(anyhow::anyhow!("Redis pool size must be greater than 0"));
        }

        // Validate global config
        if self.global.hit_rate_threshold < 0.0 || self.global.hit_rate_threshold > 1.0 {
            return Err(anyhow::anyhow!("Hit rate threshold must be between 0 and 1"));
        }

        Ok(())
    }

    /// Create a configuration template
    pub fn template() -> Self {
        Self::default()
    }

    /// Save configuration to file
    pub async fn save_to_file(&self, config_path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize configuration: {}", e))?;

        tokio::fs::write(config_path, content)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to write configuration file: {}", e))?;

        Ok(())
    }

    /// Get cache directory path
    pub fn get_cache_dir(&self) -> &std::path::Path {
        std::path::Path::new(&self.file_cache.cache_dir)
    }

    /// Get memory cache size in bytes
    pub fn get_memory_cache_size_bytes(&self) -> usize {
        self.memory_cache.max_size_mb * 1024 * 1024
    }

    /// Get file cache size in bytes
    pub fn get_file_cache_size_bytes(&self) -> usize {
        self.file_cache.max_size_mb * 1024 * 1024
    }

    /// Get min free space in bytes
    pub fn get_min_free_space_bytes(&self) -> usize {
        self.file_cache.min_free_space_mb * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_validation() {
        let mut config = CacheConfig::default();
        
        // Test invalid memory cache size
        config.memory_cache.max_size_mb = 0;
        let result = config.validate();
        assert!(result.is_err());
        
        // Test invalid compression level
        config.memory_cache.max_size_mb = 512; // Reset to valid
        config.file_cache.compression_level = 15;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_strategy() {
        assert!(CacheStrategy::MemoryOnly.uses_memory());
        assert!(!CacheStrategy::MemoryOnly.uses_file());
        
        assert!(CacheStrategy::MultiTier.uses_memory());
        assert!(CacheStrategy::MultiTier.uses_file());
        assert!(CacheStrategy::MultiTier.uses_distributed());
    }

    #[test]
    fn test_env_parsing() {
        std::env::set_var("CACHE_ENABLE", "false");
        std::env::set_var("CACHE_STRATEGY", "memory_only");
        std::env::set_var("CACHE_MEMORY_SIZE_MB", "256");

        let config = CacheConfig::from_env();
        assert!(config.is_ok());

        if let Ok(config) = config {
            assert!(!config.global.enable_cache);
            assert!(matches!(config.global.strategy, CacheStrategy::MemoryOnly));
            assert_eq!(config.memory_cache.max_size_mb, 256);
        }
    }
}