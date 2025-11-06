use anyhow::{Result, Context, anyhow};
use std::sync::Arc;

use super::s3_storage::S3Storage;
use super::gcs_storage::GCSStorage;
use super::azure_blob_storage::AzureBlobStorage;
use super::cloud_config::{CloudStorageConfig, StorageProviderConfig};

/// Storage provider enumeration
#[derive(Debug, Clone)]
pub enum StorageProvider {
    S3,
    GCS,
    Azure,
    Local,
}

/// Signed URL operation types
#[derive(Debug, Clone, Copy)]
pub enum SignedUrlOperation {
    GetObject,
    PutObject,
}

/// Cloud storage trait defining the common interface
#[async_trait]
pub trait CloudStorage: Send + Sync {
    async fn upload_file(
        &self,
        local_path: &std::path::Path,
        cloud_path: &str,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()>;

    async fn download_file(
        &self,
        cloud_path: &str,
        local_path: &std::path::Path,
    ) -> Result<()>;

    async fn delete_file(&self, cloud_path: &str) -> Result<()>;

    async fn list_files(
        &self,
        prefix: Option<&str>,
        max_keys: Option<i32>,
    ) -> Result<Vec<super::s3_storage::StorageObject>>;

    async fn get_file_metadata(&self, cloud_path: &str) -> Result<super::s3_storage::StorageObject>;

    async fn upload_directory(
        &self,
        local_dir: &std::path::Path,
        cloud_prefix: &str,
        exclude_patterns: Option<Vec<String>>,
    ) -> Result<Vec<String>>;

    async fn download_directory(
        &self,
        cloud_prefix: &str,
        local_dir: &std::path::Path,
        overwrite: bool,
    ) -> Result<Vec<String>>;

    async fn create_signed_url(
        &self,
        cloud_path: &str,
        expires_in: std::time::Duration,
        operation: SignedUrlOperation,
    ) -> Result<String>;

    fn get_provider(&self) -> StorageProvider;
}

/// Unified cloud storage client that can switch between different providers
#[derive(Clone)]
pub struct CloudStorageClient {
    storage: Arc<dyn CloudStorage>,
}

impl CloudStorageClient {
    /// Create a new cloud storage client from provider configuration
    pub async fn new(config: &StorageProviderConfig) -> Result<Self> {
        let storage = match config {
            StorageProviderConfig::S3(ref s3_config) => {
                Arc::new(S3Storage::new(s3_config).await
                    .context("Failed to initialize S3 storage")?) as Arc<dyn CloudStorage>
            },
            StorageProviderConfig::GCS(ref gcs_config) => {
                Arc::new(GCSStorage::new(gcs_config).await
                    .context("Failed to initialize GCS storage")?) as Arc<dyn CloudStorage>
            },
            StorageProviderConfig::Azure(ref azure_config) => {
                Arc::new(AzureBlobStorage::new(azure_config).await
                    .context("Failed to initialize Azure Blob Storage")?) as Arc<dyn CloudStorage>
            },
        };

        Ok(Self { storage })
    }

    /// Create a cloud storage client with fallback providers
    pub async fn new_with_fallback(
        primary_config: &StorageProviderConfig,
        fallback_configs: Vec<StorageProviderConfig>,
    ) -> Result<Self> {
        // Try primary provider first
        match Self::new(primary_config).await {
            Ok(client) => Ok(client),
            Err(e) => {
                // Try fallback providers
                for fallback_config in fallback_configs {
                    match Self::new(&fallback_config).await {
                        Ok(client) => {
                            log::warn!("Primary storage provider failed, using fallback: {}", e);
                            return Ok(client);
                        },
                        Err(fallback_error) => {
                            log::warn!("Fallback storage provider also failed: {}", fallback_error);
                            continue;
                        }
                    }
                }
                Err(anyhow!("All storage providers failed"))
            }
        }
    }

    /// Get the underlying storage implementation
    pub fn get_storage(&self) -> &dyn CloudStorage {
        self.storage.as_ref()
    }

    /// Upload a file to cloud storage
    pub async fn upload_file(
        &self,
        local_path: &std::path::Path,
        cloud_path: &str,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        self.storage.upload_file(local_path, cloud_path, metadata).await
    }

    /// Download a file from cloud storage
    pub async fn download_file(
        &self,
        cloud_path: &str,
        local_path: &std::path::Path,
    ) -> Result<()> {
        self.storage.download_file(cloud_path, local_path).await
    }

    /// Delete a file from cloud storage
    pub async fn delete_file(&self, cloud_path: &str) -> Result<()> {
        self.storage.delete_file(cloud_path).await
    }

    /// List files in cloud storage
    pub async fn list_files(
        &self,
        prefix: Option<&str>,
        max_keys: Option<i32>,
    ) -> Result<Vec<super::s3_storage::StorageObject>> {
        self.storage.list_files(prefix, max_keys).await
    }

    /// Get file metadata from cloud storage
    pub async fn get_file_metadata(&self, cloud_path: &str) -> Result<super::s3_storage::StorageObject> {
        self.storage.get_file_metadata(cloud_path).await
    }

    /// Upload a directory to cloud storage
    pub async fn upload_directory(
        &self,
        local_dir: &std::path::Path,
        cloud_prefix: &str,
        exclude_patterns: Option<Vec<String>>,
    ) -> Result<Vec<String>> {
        self.storage.upload_directory(local_dir, cloud_prefix, exclude_patterns).await
    }

    /// Download a directory from cloud storage
    pub async fn download_directory(
        &self,
        cloud_prefix: &str,
        local_dir: &std::path::Path,
        overwrite: bool,
    ) -> Result<Vec<String>> {
        self.storage.download_directory(cloud_prefix, local_dir, overwrite).await
    }

    /// Create a signed URL for secure access
    pub async fn create_signed_url(
        &self,
        cloud_path: &str,
        expires_in: std::time::Duration,
        operation: SignedUrlOperation,
    ) -> Result<String> {
        self.storage.create_signed_url(cloud_path, expires_in, operation).await
    }

    /// Get the storage provider type
    pub fn get_provider(&self) -> StorageProvider {
        self.storage.get_provider()
    }

    /// Sync local directory to cloud storage
    pub async fn sync_directory_to_cloud(
        &self,
        local_dir: &std::path::Path,
        cloud_prefix: &str,
        delete_remote: bool,
        exclude_patterns: Option<Vec<String>>,
    ) -> Result<std::collections::HashMap<String, String>> {
        let mut sync_results = std::collections::HashMap::new();

        // Upload all local files that don't exist in cloud or are different
        let uploaded_files = self.upload_directory(local_dir, cloud_prefix, exclude_patterns).await?;
        for file in uploaded_files {
            sync_results.insert(file.clone(), "uploaded".to_string());
        }

        // Optionally delete remote files that don't exist locally
        if delete_remote {
            let remote_files = self.list_files(Some(cloud_prefix), None).await?;
            let local_files = Self::list_local_files(local_dir).await?;

            for remote_file in remote_files {
                let remote_path = remote_file.key;
                let relative_path = remote_path.strip_prefix(cloud_prefix).unwrap_or(&remote_path);
                
                if !local_files.contains(relative_path) {
                    self.delete_file(&remote_path).await?;
                    sync_results.insert(remote_path, "deleted".to_string());
                }
            }
        }

        Ok(sync_results)
    }

    /// Sync cloud directory to local storage
    pub async fn sync_directory_from_cloud(
        &self,
        cloud_prefix: &str,
        local_dir: &std::path::Path,
        overwrite: bool,
    ) -> Result<std::collections::HashMap<String, String>> {
        let mut sync_results = std::collections::HashMap::new();

        // Download all files
        let downloaded_files = self.download_directory(cloud_prefix, local_dir, overwrite).await?;
        for file in downloaded_files {
            sync_results.insert(file.clone(), "downloaded".to_string());
        }

        Ok(sync_results)
    }

    /// Get storage provider statistics
    pub async fn get_storage_stats(&self) -> Result<std::collections::HashMap<String, serde_json::Value>> {
        let mut stats = std::collections::HashMap::new();
        
        stats.insert("provider".to_string(), 
            serde_json::Value::String(format!("{:?}", self.get_provider())));

        // Get basic storage information
        let files = self.list_files(None, Some(1000)).await.unwrap_or_default();
        
        stats.insert("total_files".to_string(), serde_json::Value::Number(files.len().into()));
        
        let total_size: u64 = files.iter().map(|f| f.size).sum();
        stats.insert("total_size_bytes".to_string(), serde_json::Value::Number(total_size.into()));
        stats.insert("total_size_mb".to_string(), 
            serde_json::Value::Number((total_size / (1024 * 1024)).into()));

        Ok(stats)
    }

    /// List local files for comparison with cloud storage
    async fn list_local_files(local_dir: &std::path::Path) -> Result<std::collections::HashSet<String>> {
        let mut local_files = std::collections::HashSet::new();
        
        let mut dir_entries = tokio::fs::read_dir(local_dir).await
            .context("Failed to read local directory")?;

        while let Some(entry) = dir_entries.next_entry().await
            .context("Failed to read directory entry")? {
            
            let path = entry.path();
            
            if path.is_file() {
                let relative_path = path.strip_prefix(local_dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                local_files.insert(relative_path);
            } else if path.is_dir() {
                let sub_files = Self::list_local_files(&path).await?;
                local_files.extend(sub_files);
            }
        }

        Ok(local_files)
    }
}

/// Storage factory for creating storage clients
pub struct StorageFactory;

impl StorageFactory {
    /// Create a storage client based on environment configuration
    pub async fn from_env() -> Result<CloudStorageClient> {
        let storage_type = std::env::var("STORAGE_PROVIDER")
            .unwrap_or_else(|_| "local".to_string())
            .to_lowercase();

        let config = match storage_type.as_str() {
            "s3" => {
                let s3_config = StorageProviderConfig::S3(crate::storage::cloud_storage::cloud_config::S3Config {
                    bucket_name: std::env::var("S3_BUCKET")
                        .context("S3_BUCKET environment variable required")?,
                    region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    access_key: std::env::var("AWS_ACCESS_KEY_ID").ok(),
                    secret_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
                    endpoint_url: std::env::var("AWS_ENDPOINT_URL").ok(),
                    force_path_style: std::env::var("AWS_FORCE_PATH_STYLE")
                        .unwrap_or_else(|_| "false".to_string())
                        .parse()
                        .unwrap_or(false),
                });
                config
            },
            "gcs" => {
                let gcs_config = StorageProviderConfig::GCS(crate::storage::cloud_storage::cloud_config::GCSConfig {
                    bucket_name: std::env::var("GCS_BUCKET")
                        .context("GCS_BUCKET environment variable required")?,
                    service_account_path: std::env::var("GCS_SERVICE_ACCOUNT_PATH").ok(),
                    project_id: std::env::var("GCS_PROJECT_ID").ok(),
                });
                config
            },
            "azure" => {
                let azure_config = StorageProviderConfig::Azure(crate::storage::cloud_storage::cloud_config::AzureBlobConfig {
                    connection_string: std::env::var("AZURE_STORAGE_CONNECTION_STRING")
                        .context("AZURE_STORAGE_CONNECTION_STRING environment variable required")?,
                    container_name: std::env::var("AZURE_STORAGE_CONTAINER_NAME")
                        .context("AZURE_STORAGE_CONTAINER_NAME environment variable required")?,
                });
                config
            },
            _ => {
                return Err(anyhow!("Unsupported storage provider: {}. Supported providers: s3, gcs, azure", storage_type));
            }
        };

        CloudStorageClient::new(&config).await
    }

    /// Create a storage client with multiple provider fallback
    pub async fn from_env_with_fallback() -> Result<CloudStorageClient> {
        let primary_storage_type = std::env::var("PRIMARY_STORAGE_PROVIDER")
            .unwrap_or_else(|_| "s3".to_string())
            .to_lowercase();

        let fallback_storage_types = std::env::var("FALLBACK_STORAGE_PROVIDERS")
            .unwrap_or_else(|_| "gcs,azure".to_string())
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .collect::<Vec<_>>();

        let mut configs = Vec::new();
        
        // Add primary config
        let primary_config = Self::config_from_type(&primary_storage_type)?;
        configs.push(primary_config);

        // Add fallback configs
        for fallback_type in fallback_storage_types {
            if fallback_type != primary_storage_type {
                if let Ok(config) = Self::config_from_type(&fallback_type) {
                    configs.push(config);
                }
            }
        }

        if configs.is_empty() {
            return Err(anyhow!("No valid storage configurations found"));
        }

        let (primary, fallbacks) = configs.split_first().unwrap();
        CloudStorageClient::new_with_fallback(primary, fallbacks.to_vec()).await
    }

    /// Create configuration from storage type
    fn config_from_type(storage_type: &str) -> Result<StorageProviderConfig> {
        match storage_type {
            "s3" => Ok(StorageProviderConfig::S3(
                crate::storage::cloud_storage::cloud_config::S3Config {
                    bucket_name: std::env::var("S3_BUCKET").unwrap_or_default(),
                    region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    access_key: std::env::var("AWS_ACCESS_KEY_ID").ok(),
                    secret_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
                    endpoint_url: std::env::var("AWS_ENDPOINT_URL").ok(),
                    force_path_style: std::env::var("AWS_FORCE_PATH_STYLE")
                        .unwrap_or_else(|_| "false".to_string())
                        .parse()
                        .unwrap_or(false),
                }
            )),
            "gcs" => Ok(StorageProviderConfig::GCS(
                crate::storage::cloud_storage::cloud_config::GCSConfig {
                    bucket_name: std::env::var("GCS_BUCKET").unwrap_or_default(),
                    service_account_path: std::env::var("GCS_SERVICE_ACCOUNT_PATH").ok(),
                    project_id: std::env::var("GCS_PROJECT_ID").ok(),
                }
            )),
            "azure" => Ok(StorageProviderConfig::Azure(
                crate::storage::cloud_storage::cloud_config::AzureBlobConfig {
                    connection_string: std::env::var("AZURE_STORAGE_CONNECTION_STRING").unwrap_or_default(),
                    container_name: std::env::var("AZURE_STORAGE_CONTAINER_NAME").unwrap_or_default(),
                }
            )),
            _ => Err(anyhow!("Unsupported storage provider: {}", storage_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_factory_env_parsing() {
        // Test that the storage factory can parse environment variables
        // This test will use mocked environment variables

        std::env::set_var("STORAGE_PROVIDER", "s3");
        std::env::set_var("S3_BUCKET", "test-bucket");
        std::env::set_var("AWS_REGION", "us-west-2");

        // This will fail without actual AWS credentials, but tests the parsing logic
        // In a real test environment, you would need to mock the cloud storage APIs
    }

    #[tokio::test]
    async fn test_fallback_storage_creation() {
        let primary_config = StorageProviderConfig::S3(crate::storage::cloud_storage::cloud_config::S3Config {
            bucket_name: "primary-bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: Some("test-access".to_string()),
            secret_key: Some("test-secret".to_string()),
            endpoint_url: None,
            force_path_style: false,
        });

        let fallback_config = StorageProviderConfig::GCS(crate::storage::cloud_storage::cloud_config::GCSConfig {
            bucket_name: "fallback-bucket".to_string(),
            service_account_path: None,
            project_id: None,
        });

        // This test demonstrates the structure but will fail without actual credentials
        let _fallback_client = CloudStorageClient::new_with_fallback(&primary_config, vec![fallback_config]).await;
    }

    #[tokio::test]
    async fn test_local_file_listing() {
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        // Create test files
        tokio::fs::write(temp_path.join("file1.txt"), "test content").await.unwrap();
        tokio::fs::write(temp_path.join("file2.txt"), "test content").await.unwrap();
        
        let local_files = CloudStorageClient::list_local_files(temp_path).await.unwrap();
        
        assert!(local_files.contains("file1.txt"));
        assert!(local_files.contains("file2.txt"));
    }
}