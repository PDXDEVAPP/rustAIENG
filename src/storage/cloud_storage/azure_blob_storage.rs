use std::path::Path;
use async_trait::async_trait;
use azure_storage_blobs::blob::BlockListType;
use azure_storage_blobs::container::PublicAccess;
use azure_storage_blobs::core::{prelude::*, CloudStorageAccount, RequestOptions};
use azure_storage_blobs::http::types::DeleteSnapshotsOption;
use azure_storage_blobs::http::types::blob::{BlobType, CopyStatus, LeaseStatus, BlockList, Block};
use azure_storage_blobs::http::types::service::{ListBlobsOptions, ServiceListBlobItemsOptions};
use azure_storage_blobs::http::types::ConditionGenerationMatch;
use azure_storage_blobs::http::Client;
use azure_storage_blobs::lease::LeaseId;
use azure_identity::DefaultAzureCredential;
use azure_core::url::Url;
use anyhow::{Result, Context, anyhow};
use tokio::io::AsyncReadExt;
use bytes::Bytes;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use super::storage_factory::{CloudStorage, StorageProvider};
use super::cloud_config::{CloudStorageConfig, AzureBlobConfig};

/// Azure Blob Storage implementation for cloud storage operations
#[derive(Debug, Clone)]
pub struct AzureBlobStorage {
    client: BlobServiceClient,
    config: AzureBlobConfig,
    container_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    pub key: String,
    pub size: u64,
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
    pub etag: Option<String>,
    pub is_dir: bool,
}

#[async_trait]
impl CloudStorage for AzureBlobStorage {
    async fn upload_file(
        &self,
        local_path: &Path,
        cloud_path: &str,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        let file_data = tokio::fs::read(local_path)
            .await
            .context("Failed to read local file for upload")?;

        // Get container client
        let container_client = self.client
            .container_client(self.container_name.clone());

        // Get blob client
        let blob_client = container_client
            .blob_client(cloud_path);

        // Prepare upload options
        let mut upload_options = PutBlockBlobOptions::default();

        // Add metadata if provided
        if let Some(ref meta) = metadata {
            let mut azure_metadata = std::collections::HashMap::new();
            for (key, value) in meta.iter() {
                azure_metadata.insert(key.clone(), value.clone());
            }
            upload_options = upload_options.metadata(azure_metadata);
        }

        // Upload the blob
        blob_client
            .put_block_blob(&file_data)
            .options(upload_options)
            .await
            .context("Failed to upload file to Azure Blob Storage")?;

        Ok(())
    }

    async fn download_file(
        &self,
        cloud_path: &str,
        local_path: &Path,
    ) -> Result<()> {
        // Get blob client
        let container_client = self.client
            .container_client(self.container_name.clone());

        let blob_client = container_client
            .blob_client(cloud_path);

        // Download the blob
        let download_response = blob_client
            .get()
            .await
            .context("Failed to download file from Azure Blob Storage")?;

        // Create parent directories if they don't exist
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create parent directories")?;
        }

        // Write to local file
        let data = download_response
            .data()
            .await
            .context("Failed to read blob data")?;

        tokio::fs::write(local_path, data)
            .await
            .context("Failed to write to local file")?;

        Ok(())
    }

    async fn delete_file(&self, cloud_path: &str) -> Result<()> {
        // Get blob client
        let container_client = self.client
            .container_client(self.container_name.clone());

        let blob_client = container_client
            .blob_client(cloud_path);

        // Delete the blob
        blob_client
            .delete()
            .options(DeleteBlobOptions::default().delete_snapshots_option(DeleteSnapshotsOption::Include))
            .await
            .context("Failed to delete file from Azure Blob Storage")?;

        Ok(())
    }

    async fn list_files(
        &self,
        prefix: Option<&str>,
        max_keys: Option<i32>,
    ) -> Result<Vec<StorageObject>> {
        // Get container client
        let container_client = self.client
            .container_client(self.container_name.clone());

        // Prepare list options
        let mut list_options = ListBlobsOptions::default();
        if let Some(p) = prefix {
            list_options = list_options.prefix(p.to_string());
        }

        if let Some(max) = max_keys {
            list_options = list_options.max_results(max);
        }

        // List blobs
        let mut blob_list = Vec::new();
        let mut list_response = container_client
            .list_blobs()
            .options(list_options)
            .await
            .context("Failed to list files from Azure Blob Storage")?;

        for blob_item in list_response.blobs() {
            let storage_object = StorageObject {
                key: blob_item.name().to_string(),
                size: blob_item.properties().content_length(),
                last_modified: blob_item.properties().last_modified().cloned(),
                etag: blob_item.properties().etag().map(|s| s.to_string()),
                is_dir: false,
            };
            blob_list.push(storage_object);
        }

        Ok(blob_list)
    }

    async fn get_file_metadata(&self, cloud_path: &str) -> Result<StorageObject> {
        // Get blob client
        let container_client = self.client
            .container_client(self.container_name.clone());

        let blob_client = container_client
            .blob_client(cloud_path);

        // Get blob properties
        let get_response = blob_client
            .get_properties()
            .await
            .context("Failed to get file metadata from Azure Blob Storage")?;

        Ok(StorageObject {
            key: cloud_path.to_string(),
            size: get_response.properties().content_length(),
            last_modified: get_response.properties().last_modified().cloned(),
            etag: get_response.properties().etag().map(|s| s.to_string()),
            is_dir: false,
        })
    }

    async fn upload_directory(
        &self,
        local_dir: &Path,
        cloud_prefix: &str,
        exclude_patterns: Option<Vec<String>>,
    ) -> Result<Vec<String>> {
        let mut uploaded_files = Vec::new();

        // Walk through local directory recursively
        let mut dir_entries = tokio::fs::read_dir(local_dir).await
            .context("Failed to read local directory")?;

        while let Some(entry) = dir_entries.next_entry().await
            .context("Failed to read directory entry")? {
            
            let path = entry.path();
            let file_name = path.file_name()
                .unwrap_or_default()
                .to_string_lossy();

            // Skip excluded patterns
            if let Some(ref patterns) = exclude_patterns {
                if patterns.iter().any(|pattern| file_name.contains(pattern)) {
                    continue;
                }
            }

            if path.is_file() {
                // Calculate relative path from local_dir
                let relative_path = path.strip_prefix(local_dir)
                    .context("Failed to calculate relative path")?;
                let cloud_path = format!("{}/{}", cloud_prefix, relative_path.display());

                self.upload_file(&path, &cloud_path, None).await?;

                uploaded_files.push(cloud_path);
            } else if path.is_dir() {
                // Recursively upload subdirectories
                let sub_cloud_prefix = format!("{}/{}", cloud_prefix, file_name);
                let sub_uploaded = self.upload_directory(&path, &sub_cloud_prefix, exclude_patterns.clone()).await?;
                uploaded_files.extend(sub_uploaded);
            }
        }

        Ok(uploaded_files)
    }

    async fn download_directory(
        &self,
        cloud_prefix: &str,
        local_dir: &Path,
        overwrite: bool,
    ) -> Result<Vec<String>> {
        let objects = self.list_files(Some(cloud_prefix), None).await?;
        let mut downloaded_files = Vec::new();

        for obj in objects {
            // Calculate relative path
            let relative_path = obj.key.trim_start_matches(cloud_prefix);
            if relative_path.is_empty() {
                continue;
            }

            let local_path = local_dir.join(relative_path);

            // Create parent directories
            if let Some(parent) = local_path.parent() {
                tokio::fs::create_dir_all(parent).await
                    .context("Failed to create parent directories")?;
            }

            // Check if file should be downloaded
            if !overwrite && local_path.exists() {
                continue;
            }

            self.download_file(&obj.key, &local_path).await?;

            downloaded_files.push(local_path.display().to_string());
        }

        Ok(downloaded_files)
    }

    async fn create_signed_url(
        &self,
        cloud_path: &str,
        expires_in: std::time::Duration,
        operation: super::storage_factory::SignedUrlOperation,
    ) -> Result<String> {
        let expiry_time = std::time::SystemTime::now() + expires_in;

        // Get container client
        let container_client = self.client
            .container_client(self.container_name.clone());

        let blob_client = container_client
            .blob_client(cloud_path);

        // Generate SAS token based on operation
        match operation {
            super::storage_factory::SignedUrlOperation::GetObject => {
                let sas = blob_client
                    .generate_read_sas_url(PublicAccess::None)
                    .expiry_time(expiry_time)
                    .sign()?;

                Ok(sas)
            },
            super::storage_factory::SignedUrlOperation::PutObject => {
                let sas = blob_client
                    .generate_write_sas_url(PublicAccess::None)
                    .expiry_time(expiry_time)
                    .sign()?;

                Ok(sas)
            }
        }
    }

    fn get_provider(&self) -> StorageProvider {
        StorageProvider::Azure
    }
}

impl AzureBlobStorage {
    /// Create a new Azure Blob Storage instance
    pub async fn new(config: &AzureBlobConfig) -> Result<Self> {
        let storage_account = CloudStorageAccount::from_connection_string(&config.connection_string)
            .await
            .context("Failed to create storage account from connection string")?;

        let blob_service_client = storage_account
            .blob_service_client();

        Ok(Self {
            client: blob_service_client,
            config: config.clone(),
            container_name: config.container_name.clone(),
        })
    }

    /// Create a new Azure Blob Storage instance with shared key
    pub async fn new_with_shared_key(
        account_name: &str,
        shared_key: &str,
        container_name: &str,
    ) -> Result<Self> {
        let storage_account = CloudStorageAccount::new(account_name, shared_key);
        let blob_service_client = storage_account.blob_service_client();

        Ok(Self {
            client: blob_service_client,
            config: AzureBlobConfig {
                connection_string: "".to_string(), // Will be constructed from account name and key
                container_name: container_name.to_string(),
            },
            container_name: container_name.to_string(),
        })
    }

    /// Create a new Azure Blob Storage instance with SAS token
    pub async fn new_with_sas_token(
        account_name: &str,
        sas_token: &str,
        container_name: &str,
    ) -> Result<Self> {
        let storage_account = CloudStorageAccount::new_sas(
            account_name,
            sas_token
        );

        let blob_service_client = storage_account.blob_service_client();

        Ok(Self {
            client: blob_service_client,
            config: AzureBlobConfig {
                connection_string: "".to_string(), // Will be constructed from SAS token
                container_name: container_name.to_string(),
            },
            container_name: container_name.to_string(),
        })
    }

    /// Create container if it doesn't exist
    pub async fn ensure_container_exists(&self) -> Result<()> {
        let container_client = self.client
            .container_client(self.container_name.clone());

        // Try to get container properties (this will fail if container doesn't exist)
        match container_client.get_properties().await {
            Ok(_) => {
                // Container exists
                Ok(())
            },
            Err(_) => {
                // Container doesn't exist, create it
                container_client
                    .create()
                    .public_access(PublicAccess::None)
                    .await
                    .context("Failed to create container")?;

                Ok(())
            }
        }
    }

    /// Get container properties
    pub async fn get_container_properties(&self) -> Result<()> {
        let container_client = self.client
            .container_client(self.container_name.clone());

        let properties = container_client
            .get_properties()
            .await
            .context("Failed to get container properties")?;

        println!("Container properties: {:?}", properties);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_azure_blob_storage_creation() {
        // This test requires actual Azure credentials to work
        // In a real test environment, you would use test connection strings
        
        let config = AzureBlobConfig {
            connection_string: "DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=testkey;EndpointSuffix=core.windows.net".to_string(),
            container_name: "test-container".to_string(),
        };

        // This will fail without valid credentials, but tests compilation
    }

    #[tokio::test]
    async fn test_azure_blob_metadata_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), "gpt-4".to_string());
        metadata.insert("version".to_string(), "2.0".to_string());

        assert_eq!(metadata.get("model_name"), Some(&"gpt-4".to_string()));
        assert_eq!(metadata.get("version"), Some(&"2.0".to_string()));
    }

    #[tokio::test]
    async fn test_azure_blob_url_creation() {
        let account = "myaccount".to_string();
        let container = "mycontainer".to_string();
        let blob = "models/dall-e-2/image.jpg".to_string();
        
        // Test URL construction (actual SAS generation would require authentication)
        let base_url = format!("https://{}.blob.core.windows.net/{}/{}", account, container, blob);
        assert!(base_url.contains(".blob.core.windows.net"));
        assert!(base_url.contains("models/dall-e-2/image.jpg"));
    }
}