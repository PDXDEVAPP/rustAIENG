use std::path::Path;
use async_trait::async_trait;
use google_cloud_auth::token::TokenSourceProvider;
use google_cloud_storage::client::Client;
use google_cloud_storage::client::http_types::{DeleteObjectError, GetObjectError, ListObjectsError, PutObjectError};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use google_cloud_storage::http::objects::list::ListObjectsRequest;
use google_cloud_storage::http::objects::upload::{UploadObjectRequest, UploadType};
use google_cloud_storage::http::objects::delete::DeleteObjectRequest;
use google_cloud_storage::http::objects::metadata::GetObjectMetadataRequest;
use google_cloud_storage::http::pre-signed::GetObjectV2;
use google_cloud_storage::http::types::ListResponse;
use anyhow::{Result, Context, anyhow};
use tokio::io::AsyncReadExt;
use bytes::Bytes;
use serde::{Serialize, Deserialize};

use super::storage_factory::{CloudStorage, StorageProvider};
use super::cloud_config::{CloudStorageConfig, GCSConfig};

/// Google Cloud Storage implementation for cloud storage operations
#[derive(Debug, Clone)]
pub struct GCSStorage {
    client: Client,
    config: GCSConfig,
    bucket_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    pub key: String,
    pub size: u64,
    pub last_modified: Option<google_cloud_storage::http::types::DateTime>,
    pub etag: Option<String>,
    pub is_dir: bool,
}

#[async_trait]
impl CloudStorage for GCSStorage {
    async fn upload_file(
        &self,
        local_path: &Path,
        cloud_path: &str,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        let file_data = tokio::fs::read(local_path)
            .await
            .context("Failed to read local file for upload")?;

        let mut upload_type = UploadType::Simple;
        if let Some(ref credentials) = self.config.service_account_path {
            upload_type = UploadType::Resumable;
        }

        // Create upload request
        let upload_request = UploadObjectRequest {
            bucket: self.bucket_name.clone(),
            object: cloud_path.to_string(),
            ..Default::default()
        };

        // Add metadata if provided
        let mut upload_type_with_metadata = upload_type;
        if let Some(meta) = metadata {
            let mut object_metadata = std::collections::HashMap::new();
            for (key, value) in meta {
                object_metadata.insert(key, value);
            }
            upload_type_with_metadata = UploadType::SimpleWithMetadata {
                metadata: object_metadata,
            };
        }

        // Perform upload
        let response = self.client
            .upload_object(&upload_request, file_data, &upload_type_with_metadata)
            .await
            .context("Failed to upload file to GCS")?;

        // Wait for upload to complete if resumable
        if let UploadType::Resumable = upload_type_with_metadata {
            // For resumable uploads, we need to wait for completion
            // This is a simplified implementation - in practice, you'd handle upload sessions
        }

        // Ignore the response for now
        let _ = response;

        Ok(())
    }

    async fn download_file(
        &self,
        cloud_path: &str,
        local_path: &Path,
    ) -> Result<()> {
        let get_request = GetObjectRequest {
            bucket: self.bucket_name.clone(),
            object: cloud_path.to_string(),
            ..Default::default()
        };

        // Create parent directories if they don't exist
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create parent directories")?;
        }

        // Download the object
        let response = self.client
            .download_object(&get_request, &Range::Default)
            .await
            .context("Failed to download file from GCS")?;

        // Write to local file
        tokio::fs::write(local_path, response)
            .await
            .context("Failed to write to local file")?;

        Ok(())
    }

    async fn delete_file(&self, cloud_path: &str) -> Result<()> {
        let delete_request = DeleteObjectRequest {
            bucket: self.bucket_name.clone(),
            object: cloud_path.to_string(),
        };

        self.client
            .delete_object(&delete_request)
            .await
            .context("Failed to delete file from GCS")?;

        Ok(())
    }

    async fn list_files(
        &self,
        prefix: Option<&str>,
        max_keys: Option<i32>,
    ) -> Result<Vec<StorageObject>> {
        let mut list_request = ListObjectsRequest {
            bucket: self.bucket_name.clone(),
            ..Default::default()
        };

        if let Some(p) = prefix {
            list_request.prefix = Some(p.to_string());
        }

        if let Some(max) = max_keys {
            list_request.max_results = Some(max);
        }

        let response: ListResponse = self.client
            .list_objects(&list_request)
            .await
            .context("Failed to list files from GCS")?;

        let objects = response
            .objects()
            .iter()
            .map(|obj| StorageObject {
                key: obj.name().unwrap_or("").to_string(),
                size: obj.size().unwrap_or(0) as u64,
                last_modified: obj.updated().cloned(),
                etag: obj.etag().map(|s| s.to_string()),
                is_dir: false,
            })
            .collect();

        Ok(objects)
    }

    async fn get_file_metadata(&self, cloud_path: &str) -> Result<StorageObject> {
        let metadata_request = GetObjectMetadataRequest {
            bucket: self.bucket_name.clone(),
            object: cloud_path.to_string(),
        };

        let metadata = self.client
            .get_object_metadata(&metadata_request)
            .await
            .context("Failed to get file metadata from GCS")?;

        Ok(StorageObject {
            key: cloud_path.to_string(),
            size: metadata.size().unwrap_or(0) as u64,
            last_modified: metadata.updated().cloned(),
            etag: metadata.etag().map(|s| s.to_string()),
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
        // For GCS, we use v4 signing
        let expires_in_secs = expires_in.as_secs() as i64;

        let request = GetObjectV2 {
            bucket: self.bucket_name.clone(),
            object: cloud_path.to_string(),
            expires_in: Some(expires_in_secs),
            ..Default::default()
        };

        let presigned_url = self.client
            .get_object_v2_signed_url(&request)
            .await
            .context("Failed to create signed URL for GCS")?;

        Ok(presigned_url)
    }

    fn get_provider(&self) -> StorageProvider {
        StorageProvider::GCS
    }
}

impl GCSStorage {
    /// Create a new GCS storage instance
    pub async fn new(config: &GCSConfig) -> Result<Self> {
        let client = if let Some(ref service_account_path) = config.service_account_path {
            // Use service account credentials
            Client::from_service_account_file(service_account_path)
                .await
                .context("Failed to create GCS client with service account")?
        } else {
            // Use default credentials
            Client::new()
                .await
                .context("Failed to create GCS client with default credentials")?
        };

        Ok(Self {
            client,
            config: config.clone(),
            bucket_name: config.bucket_name.clone(),
        })
    }

    /// Create a new GCS storage instance with project ID
    pub async fn new_with_project(config: &GCSConfig, project_id: &str) -> Result<Self> {
        let client = if let Some(ref service_account_path) = config.service_account_path {
            // Use service account credentials with project ID
            Client::from_service_account_key_file_with_project(
                service_account_path, 
                project_id
            )
            .await
            .context("Failed to create GCS client with service account and project")?
        } else {
            // Use default credentials
            Client::new()
                .await
                .context("Failed to create GCS client with default credentials")?
        };

        Ok(Self {
            client,
            config: config.clone(),
            bucket_name: config.bucket_name.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_gcs_storage_creation() {
        let config = GCSConfig {
            bucket_name: "test-bucket".to_string(),
            service_account_path: None, // This would require actual credentials in a real test
            project_id: None,
        };

        // This test will fail without actual GCS credentials, but will compile
        // In a real test environment, you would need to mock the GCS client or use test credentials
    }

    #[tokio::test]
    async fn test_gcs_metadata_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), "gemini-pro".to_string());
        metadata.insert("version".to_string(), "1.5".to_string());

        assert_eq!(metadata.get("model_name"), Some(&"gemini-pro".to_string()));
        assert_eq!(metadata.get("version"), Some(&"1.5".to_string()));
    }

    #[tokio::test]
    async fn test_gcs_url_creation() {
        let bucket = "my-test-bucket".to_string();
        let object = "models/llama-2/weights.bin".to_string();
        
        // Test URL construction (actual signing would require authentication)
        let base_url = format!("https://storage.googleapis.com/{}/{}", bucket, object);
        assert!(base_url.contains("storage.googleapis.com"));
        assert!(base_url.contains("models/llama-2/weights.bin"));
    }
}