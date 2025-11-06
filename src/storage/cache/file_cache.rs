use std::path::{Path, PathBuf};
use std::collections::{HashMap, BTreeMap};
use std::time::{SystemTime, Duration, Instant};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use tokio::fs;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{info, warn, debug, error};
use flate2::{Compression, write::GzDecoder};
use lz4_flex::block::CompressMode;
use zstd::Encoder;

use super::cache_config::{CacheConfig, FileCacheConfig};
use super::file_cache::FileCacheEntry;

/// File-based cache implementation with persistence
#[derive(Debug)]
pub struct FileCache {
    config: FileCacheConfig,
    cache_dir: PathBuf,
    index: RwLock<HashMap<String, FileCacheEntry>>,
    stats: RwLock<CacheStats>,
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCacheEntry {
    pub key: String,
    pub path: PathBuf,
    pub size: u64,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub expires_at: Option<SystemTime>,
    pub metadata: HashMap<String, String>,
    pub compressed: bool,
    pub hash: String,
    pub access_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_size: u64,
    pub total_files: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub last_cleanup: Option<SystemTime>,
    pub disk_usage_percent: f64,
}

impl FileCache {
    /// Create a new file cache instance
    pub async fn new(config: &CacheConfig) -> Result<Self> {
        let cache_dir = PathBuf::from(&config.file_cache.cache_dir);
        
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir)
            .await
            .context("Failed to create cache directory")?;

        // Load existing index if persistence is enabled
        let mut index = HashMap::new();
        if config.file_cache.enable_persistence {
            index = Self::load_index(&cache_dir).await
                .unwrap_or_else(|e| {
                    warn!("Failed to load cache index, starting fresh: {}", e);
                    HashMap::new()
                });
        }

        // Verify cache files exist
        Self::verify_cache_files(&cache_dir, &mut index).await;

        // Calculate initial stats
        let total_size = index.values().map(|entry| entry.size).sum();
        let total_files = index.len();

        let stats = CacheStats {
            total_size,
            total_files,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            last_cleanup: None,
            disk_usage_percent: 0.0,
        };

        let mut cache = Self {
            config: config.file_cache.clone(),
            cache_dir,
            index: RwLock::new(index),
            stats: RwLock::new(stats),
            cleanup_task: None,
        };

        // Start cleanup task if enabled
        if cache.config.enable_cleanup {
            cache.start_cleanup_task().await;
        }

        // Update disk usage stats
        cache.update_disk_usage().await;

        info!("File cache initialized: {} files, {} bytes", 
              cache.stats.read().await.total_files,
              cache.stats.read().await.total_size);

        Ok(cache)
    }

    /// Store data in cache
    pub async fn put(&self, key: &str, data: &[u8], ttl: Option<Duration>, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let key = key.to_string();
        let hash = Self::calculate_hash(data);
        let size = data.len() as u64;

        // Check cache size limits before storing
        self.check_size_limits().await?;

        // Compress data if enabled
        let (data_to_store, compressed) = if self.config.enable_compression {
            let compressed_data = Self::compress_data(data, self.config.compression_level)?;
            (compressed_data, true)
        } else {
            (data.to_vec(), false)
        };

        // Generate file path
        let subdir = &key[0..2]; // Use first 2 chars as subdirectory
        let filename = &key[2..];
        let mut file_path = self.cache_dir.join(subdir);
        file_path.push(filename);

        // Create subdirectory if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await.context("Failed to create subdirectory")?;
        }

        // Write data to file
        fs::write(&file_path, &data_to_store)
            .await
            .context("Failed to write cache file")?;

        // Create cache entry
        let now = SystemTime::now();
        let expires_at = ttl.map(|duration| now + duration);
        
        let entry = FileCacheEntry {
            key: key.clone(),
            path: file_path.clone(),
            size: size, // Original size, not compressed size
            created_at: now,
            last_accessed: now,
            expires_at,
            metadata: metadata.unwrap_or_default(),
            compressed,
            hash,
            access_count: 1,
        };

        // Update index and stats
        {
            let mut index = self.index.write().await;
            let old_entry = index.insert(key.clone(), entry.clone());

            let mut stats = self.stats.write().await;
            if let Some(old_entry) = old_entry {
                stats.total_size -= old_entry.size;
            }
            stats.total_size += entry.size;
            stats.total_files = index.len();
        }

        // Persist index if enabled
        if self.config.enable_persistence {
            self.save_index().await?;
        }

        debug!("Stored cache entry: key={}, size={} bytes", key, size);
        Ok(())
    }

    /// Retrieve data from cache
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let index = self.index.read().await;
        let entry = match index.get(key) {
            Some(entry) => entry.clone(),
            None => {
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;
                return Ok(None);
            }
        };

        // Check if entry has expired
        if let Some(expires_at) = entry.expires_at {
            if SystemTime::now() > expires_at {
                drop(index); // Release lock before async operation
                self.delete(key).await?;
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;
                return Ok(None);
            }
        }

        // Read data from file
        let data = match fs::read(&entry.path).await {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to read cache file: {}: {}", entry.path.display(), e);
                drop(index); // Release lock before async operation
                self.delete(key).await?;
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;
                return Ok(None);
            }
        };

        // Decompress data if needed
        let data = if entry.compressed {
            match Self::decompress_data(&data, entry.compressed) {
                Ok(decompressed) => decompressed,
                Err(e) => {
                    warn!("Failed to decompress cache file: {}: {}", entry.path.display(), e);
                    drop(index); // Release lock before async operation
                    self.delete(key).await?;
                    return Ok(None);
                }
            }
        } else {
            data
        };

        // Update access statistics
        {
            let mut index = self.index.write().await;
            if let Some(entry) = index.get_mut(key) {
                entry.last_accessed = SystemTime::now();
                entry.access_count += 1;
            }
        }

        let mut stats = self.stats.write().await;
        stats.cache_hits += 1;

        debug!("Cache hit: key={}, size={} bytes", key, data.len());
        Ok(Some(data))
    }

    /// Delete an entry from cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        let mut index = self.index.write().await;
        let entry = match index.remove(key) {
            Some(entry) => entry,
            None => return Ok(()),
        };

        // Delete file
        if let Err(e) = fs::remove_file(&entry.path).await {
            warn!("Failed to delete cache file: {}: {}", entry.path.display(), e);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_size -= entry.size;
        stats.total_files = index.len();

        // Persist index if enabled
        if self.config.enable_persistence {
            self.save_index().await?;
        }

        debug!("Deleted cache entry: key={}", key);
        Ok(())
    }

    /// Check if key exists in cache
    pub async fn exists(&self, key: &str) -> bool {
        let index = self.index.read().await;
        index.contains_key(key)
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        // Delete all files
        let entries = {
            let index = self.index.read().await;
            index.values().cloned().collect::<Vec<_>>()
        };

        for entry in entries {
            if let Err(e) = fs::remove_file(&entry.path).await {
                warn!("Failed to delete cache file during clear: {}: {}", entry.path.display(), e);
            }
        }

        // Clear index and stats
        {
            let mut index = self.index.write().await;
            let mut stats = self.stats.write().await;
            index.clear();
            stats.total_size = 0;
            stats.total_files = 0;
        }

        // Save empty index if persistence enabled
        if self.config.enable_persistence {
            self.save_index().await?;
        }

        info!("Cache cleared");
        Ok(())
    }

    /// Perform cache maintenance
    pub async fn maintain(&self) -> Result<MaintenanceResult> {
        let mut removed_files = 0;
        let mut freed_space = 0u64;
        let mut expired_files = 0;

        // Remove expired entries
        let now = SystemTime::now();
        let keys_to_remove = {
            let index = self.index.read().await;
            index.iter()
                .filter(|(_, entry)| {
                    if let Some(expires_at) = entry.expires_at {
                        if now > expires_at {
                            return true;
                        }
                    }
                    false
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>()
        };

        for key in keys_to_remove {
            if let Some(entry) = self.index.read().await.get(&key).cloned() {
                if let Err(e) = fs::remove_file(&entry.path).await {
                    warn!("Failed to delete expired cache file: {}: {}", entry.path.display(), e);
                } else {
                    removed_files += 1;
                    freed_space += entry.size;
                    expired_files += 1;
                }
            }
        }

        // Remove keys from index
        {
            let mut index = self.index.write().await;
            for key in &keys_to_remove {
                if let Some(entry) = index.remove(key) {
                    freed_space += entry.size;
                }
            }

            let mut stats = self.stats.write().await;
            stats.total_size -= freed_space;
            stats.total_files -= removed_files;
            stats.last_cleanup = Some(SystemTime::now());
        }

        // Evict entries if cache is too large
        let evicted = self.evict_if_needed().await?;

        // Update disk usage
        self.update_disk_usage().await;

        // Save index if persistence enabled
        if self.config.enable_persistence {
            self.save_index().await?;
        }

        Ok(MaintenanceResult {
            removed_files,
            freed_space,
            expired_files,
            evicted_entries: evicted,
        })
    }

    /// Start background cleanup task
    async fn start_cleanup_task(&self) {
        let interval_secs = self.config.cleanup_interval;
        let cache_dir = self.cache_dir.clone();
        let file_cache = self.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_secs));
            
            loop {
                interval.tick().await;
                if let Err(e) = file_cache.maintain().await {
                    error!("Cache maintenance failed: {}", e);
                }
            }
        });

        self.cleanup_task = Some(task);
    }

    /// Calculate hash of data
    fn calculate_hash(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// Compress data
    fn compress_data(data: &[u8], level: u8) -> Result<Vec<u8>> {
        let compression = match level {
            0 => Compression::none(),
            1..=3 => Compression::fast(),
            4..=6 => Compression::default(),
            7..=9 => Compression::best(),
            _ => Compression::default(),
        };

        // Use LZ4 for better performance on binary data
        if level <= 3 {
            let compressed = lz4_flex::compress_into(data, CompressMode::Fast);
            Ok(compressed)
        } else {
            // Use gzip for better compression ratios
            let mut encoder = GzDecoder::new(Vec::new());
            encoder.write_all(data)?;
            Ok(encoder.finish()?)
        }
    }

    /// Decompress data
    fn decompress_data(data: &[u8], compressed: bool) -> Result<Vec<u8>> {
        if !compressed {
            return Ok(data.to_vec());
        }

        // Try LZ4 first (used for fast compression)
        if let Ok(decompressed) = lz4_flex::decompress(data, data.len() * 2) {
            return Ok(decompressed);
        }

        // Fallback to gzip
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        std::io::copy(&mut decoder, &mut decompressed)?;
        Ok(decompressed)
    }

    /// Check if cache size exceeds limits
    async fn check_size_limits(&self) -> Result<()> {
        let stats = self.stats.read().await;
        
        // Check file size limit
        let max_size_bytes = (self.config.max_size_mb * 1024 * 1024) as u64;
        if stats.total_files >= self.config.max_files as u64 || stats.total_size >= max_size_bytes {
            drop(stats); // Release lock before potentially evicting
            self.evict_if_needed().await?;
        }

        Ok(())
    }

    /// Evict entries if cache is too large
    async fn evict_if_needed(&self) -> Result<u32> {
        let max_size_bytes = (self.config.max_size_mb * 1024 * 1024) as u64;
        let mut evicted = 0;

        let (needs_eviction_size, needs_eviction_count) = {
            let stats = self.stats.read().await;
            (stats.total_size >= max_size_bytes, stats.total_files >= self.config.max_files as u64)
        };

        if !needs_eviction_size && !needs_eviction_count {
            return Ok(0);
        }

        // LRU eviction strategy
        let keys_to_evict = {
            let mut entries = {
                let index = self.index.read().await;
                index.iter().collect::<Vec<_>>()
            };

            // Sort by last accessed (least recently used first)
            entries.sort_by(|a, b| {
                a.1.last_accessed.cmp(&b.1.last_accessed)
            });

            // Determine how many to evict
            let target_files = (self.config.max_files as f64 * 0.9) as usize; // Evict 10% when full
            let target_size = (self.config.max_size_mb as f64 * 1024 * 1024 * 0.9) as u64; // Evict 10% when full
            
            let mut evict_count = 0;
            let mut evict_size = 0u64;

            for (key, entry) in entries.iter() {
                evict_count += 1;
                evict_size += entry.size;

                if evict_count >= target_files || evict_size >= target_size {
                    break;
                }
            }

            entries.into_iter().take(evict_count).map(|(key, _)| key.clone()).collect::<Vec<_>>()
        };

        for key in keys_to_evict {
            if let Some(entry) = self.index.read().await.get(&key).cloned() {
                if let Err(e) = fs::remove_file(&entry.path).await {
                    warn!("Failed to evict cache file: {}: {}", entry.path.display(), e);
                } else {
                    {
                        let mut index = self.index.write().await;
                        index.remove(&key);
                    }
                    
                    {
                        let mut stats = self.stats.write().await;
                        stats.total_size -= entry.size;
                        stats.total_files -= 1;
                        stats.evictions += 1;
                    }
                    
                    evicted += 1;
                }
            }
        }

        debug!("Evicted {} cache entries", evicted);
        Ok(evicted)
    }

    /// Update disk usage statistics
    async fn update_disk_usage(&self) {
        match fs::metadata(&self.cache_dir).await {
            Ok(metadata) => {
                let total_space = metadata.len() as f64;
                let used_space = self.stats.read().await.total_size as f64;
                let usage_percent = (used_space / total_space) * 100.0;

                let mut stats = self.stats.write().await;
                stats.disk_usage_percent = usage_percent;
            }
            Err(e) => {
                warn!("Failed to get cache directory stats: {}", e);
            }
        }
    }

    /// Load cache index from disk
    async fn load_index(cache_dir: &Path) -> Result<HashMap<String, FileCacheEntry>> {
        let index_path = cache_dir.join(".cache_index");
        
        if !index_path.exists() {
            return Ok(HashMap::new());
        }

        let content = fs::read_to_string(&index_path).await?;
        let index: HashMap<String, FileCacheEntry> = toml::from_str(&content)?;
        
        info!("Loaded cache index with {} entries", index.len());
        Ok(index)
    }

    /// Save cache index to disk
    async fn save_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join(".cache_index");
        
        {
            let index = self.index.read().await;
            let content = toml::to_string_pretty(&*index)
                .context("Failed to serialize cache index")?;
            
            fs::write(&index_path, content)
                .await
                .context("Failed to write cache index")?;
        }

        Ok(())
    }

    /// Verify that cache files exist
    async fn verify_cache_files(cache_dir: &Path, index: &mut HashMap<String, FileCacheEntry>) {
        let keys_to_remove = {
            let mut invalid_keys = Vec::new();
            
            for (key, entry) in index.iter() {
                if !entry.path.exists() {
                    warn!("Cache file not found, removing from index: {}", entry.path.display());
                    invalid_keys.push(key.clone());
                }
            }
            
            invalid_keys
        };

        for key in keys_to_remove {
            index.remove(&key);
        }

        if !keys_to_remove.is_empty() {
            info!("Removed {} invalid cache entries", keys_to_remove.len());
        }
    }
}

impl Clone for FileCache {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cache_dir: self.cache_dir.clone(),
            index: self.index.clone(),
            stats: self.stats.clone(),
            cleanup_task: None, // Don't clone the task handle
        }
    }
}

impl Drop for FileCache {
    fn drop(&mut self) {
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
        }
    }
}

/// Maintenance result
#[derive(Debug, Clone)]
pub struct MaintenanceResult {
    pub removed_files: u32,
    pub freed_space: u64,
    pub expired_files: u32,
    pub evicted_entries: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_cache_basic_operations() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        
        let config = CacheConfig {
            file_cache: FileCacheConfig {
                cache_dir: cache_dir.to_string_lossy().to_string(),
                max_size_mb: 10,
                max_files: 100,
                default_ttl: 3600,
                compression_level: 6,
                enable_cleanup: false,
                cleanup_interval: 3600,
                enable_compression: false,
                enable_persistence: false,
                min_free_space: 1024,
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = FileCache::new(&config).await.unwrap();

        // Test put and get
        let test_data = b"Hello, World!";
        cache.put("test_key", test_data, None, None).await.unwrap();
        
        let retrieved = cache.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some(test_data.to_vec()));

        // Test exists
        assert!(cache.exists("test_key").await);
        assert!(!cache.exists("nonexistent").await);

        // Test delete
        cache.delete("test_key").await.unwrap();
        assert!(!cache.exists("test_key").await);
    }

    #[tokio::test]
    async fn test_file_cache_compression() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        
        let config = CacheConfig {
            file_cache: FileCacheConfig {
                cache_dir: cache_dir.to_string_lossy().to_string(),
                max_size_mb: 10,
                max_files: 100,
                default_ttl: 3600,
                compression_level: 6,
                enable_cleanup: false,
                cleanup_interval: 3600,
                enable_compression: true,
                enable_persistence: false,
                min_free_space: 1024,
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = FileCache::new(&config).await.unwrap();

        // Test compression with repeated data
        let test_data = b"Hello, World! Hello, World! Hello, World!";
        cache.put("compressed_key", test_data, None, None).await.unwrap();
        
        let retrieved = cache.get("compressed_key").await.unwrap();
        assert_eq!(retrieved, Some(test_data.to_vec()));
    }

    #[tokio::test]
    async fn test_cache_maintenance() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        
        let config = CacheConfig {
            file_cache: FileCacheConfig {
                cache_dir: cache_dir.to_string_lossy().to_string(),
                max_size_mb: 1,
                max_files: 5,
                default_ttl: 1, // Very short TTL
                compression_level: 6,
                enable_cleanup: false,
                cleanup_interval: 3600,
                enable_compression: false,
                enable_persistence: false,
                min_free_space: 1024,
            },
            memory_cache: super::super::cache_config::MemoryCacheConfig::default(),
            distributed_cache: super::super::cache_config::DistributedCacheConfig::default(),
            global: super::super::cache_config::GlobalCacheConfig::default(),
        };

        let cache = FileCache::new(&config).await.unwrap();

        // Add some entries
        for i in 0..10 {
            let data = format!("test_data_{}", i).into_bytes();
            cache.put(&format!("key_{}", i), &data, None, None).await.unwrap();
        }

        // Run maintenance
        let result = cache.maintain().await.unwrap();
        
        // Should have evicted some entries due to size limits
        assert!(result.evicted_entries > 0 || result.expired_files > 0);
    }
}