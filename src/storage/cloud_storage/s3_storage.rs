use std::path::Path;
use async_trait::async_trait;
use aws_sdk_s3::{Client, Config, Region, primitives::ByteStream};
use aws_sdk_s3::error::{PutObjectError, GetObjectError, DeleteObjectError, ListObjectsV2Error};
use aws_sdk_s3::operation::{put_object::PutObjectOutput, get_object::GetObjectOutput, 
                             delete_object::DeleteObjectOutput, list_objects_v2::ListObjectsV2Output};
use aws_sdk_s3::types::{Object, Bucket};
use aws_sdk_credentials::{Credentials, ProvideCredentials};
use aws_config::meta::region::RegionProvider;
use aws_sdk_s3::config::Builder as S3ConfigBuilder;
use anyhow::{Result, Context, anyhow};
use tokio::io::AsyncReadExt;
use bytes::Bytes;
use serde::{Serialize, Deserialize};

use super::storage_factory::{CloudStorage, StorageProvider};
use super::cloud_config::{CloudStorageConfig, S3Config};

/// S3 Storage implementation for cloud storage operations
#[derive(Debug, Clone)]
pub struct S3Storage {
    client: Client,
    config: S3Config,
    bucket_name: String,
}

#[async_trait]
impl CloudStorage for S3Storage {
    async fn upload_file(
        &self,
        local_path: &Path,
        cloud_path: &str,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        let file_data = tokio::fs::read(local_path)
            .await
            .context("Failed to read local file for upload")?;

        let mut put_request = self.client
            .put_object()
            .bucket(&self.bucket_name)
            .key(cloud_path)
            .body(ByteStream::from(file_data));

        // Add metadata if provided
        if let Some(meta) = metadata {
            for (key, value) in meta {
                put_request = put_request.metadata(key, value);
            }
        }

        put_request
            .send()
            .await
            .context("Failed to upload file to S3")?;

        Ok(())
    }

    async fn download_file(
        &self,
        cloud_path: &str,
        local_path: &Path,
    ) -> Result<()> {
        let mut response = self.client
            .get_object()
            .bucket(&self.bucket_name)
            .key(cloud_path)
            .send()
            .await
            .context("Failed to download file from S3")?;

        // Create parent directories if they don't exist
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create parent directories")?;
        }

        let mut file = tokio::fs::File::create(local_path)
            .await
            .context("Failed to create local file")?;

        while let Some(chunk) = response.body.next().await {
            let chunk = chunk.context("Error reading S3 object data")?;
            tokio::io::copy(&mut chunk.as_ref(), &mut file)
                .await
                .context("Failed to write chunk to local file")?;
        }

        Ok(())
    }

    async fn delete_file(&self, cloud_path: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(&self.bucket_name)
            .key(cloud_path)
            .send()
            .await
            .context("Failed to delete file from S3")?;

        Ok(())
    }

    async fn list_files(
        &self,
        prefix: Option<&str>,
        max_keys: Option<i32>,
    ) -> Result<Vec<StorageObject>> {
        let mut request = self.client
            .list_objects_v2()
            .bucket(&self.bucket_name);

        if let Some(p) = prefix {
            request = request.prefix(p);
        }

        if let Some(max) = max_keys {
            request = request.max_keys(max);
        }

        let response = request
            .send()
            .await
            .context("Failed to list files from S3")?;

        let objects = response.contents()
            .unwrap_or_default()
            .iter()
            .map(|obj| StorageObject {
                key: obj.key().unwrap_or("").to_string(),
                size: obj.size().unwrap_or(0) as u64,
                last_modified: obj.last_modified().cloned(),
                etag: obj.etag().map(|s| s.to_string()),
                is_dir: false,
            })
            .collect();

        Ok(objects)
    }

    async fn get_file_metadata(&self, cloud_path: &str) -> Result<StorageObject> {
        let response = self.client
            .head_object()
            .bucket(&self.bucket_name)
            .key(cloud_path)
            .send()
            .await
            .context("Failed to get file metadata from S3")?;

        Ok(StorageObject {
            key: cloud_path.to_string(),
            size: response.content_length().unwrap_or(0) as u64,
            last_modified: response.last_modified().cloned(),
            etag: response.e_tag().map(|s| s.to_string()),
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
        let expires_in_secs = expires_in.as_secs() as i64;

        let request = match operation {
            super::storage_factory::SignedUrlOperation::GetObject => {
                self.client
                    .get_object()
                    .bucket(&self.bucket_name)
                    .key(cloud_path)
            },
            super::storage_factory::SignedUrlOperation::PutObject => {
                self.client
                    .put_object()
                    .bucket(&self.bucket_name)
                    .key(cloud_path)
            },
        };

        let presigned_request = request
            .presigned()
            .expires_in(expires_in_secs)
            .send()
            .await
            .context("Failed to create signed URL")?;

        Ok(presigned_request.uri().to_string())
    }

    fn get_provider(&self) -> StorageProvider {
        StorageProvider::S3
    }
}

impl S3Storage {
    /// Create a new S3 storage instance
    pub async fn new(config: &S3Config) -> Result<Self> {
        let region = Region::new(config.region.clone());

        // Build AWS credentials
        let credentials = if let (Some(ref access_key), Some(ref secret_key)) = (&config.access_key, &config.secret_key) {
            Credentials::new(access_key.clone(), secret_key.clone(), None, None, "custom")
        } else {
            return Err(anyhow!("AWS credentials not configured"));
        };

        // Build S3 config
        let s3_config = S3ConfigBuilder::new()
            .region(region)
            .credentials_provider(credentials)
            .endpoint_url(config.endpoint_url.clone())
            .force_path_style(config.force_path_style)
            .build();

        let client = Client::from_conf(s3_config);

        Ok(Self {
            client,
            config: config.clone(),
            bucket_name: config.bucket_name.clone(),
        })
    }

    /// Create a new S3 storage instance with default region
    pub async fn new_with_default_region(config: &S3Config, default_region: &str) -> Result<Self> {
        let region = if config.region.is_empty() {
            default_region.to_string()
        } else {
            config.region.clone()
        };

        let region = Region::new(region);

        // Build AWS credentials
        let credentials = if let (Some(ref access_key), Some(ref secret_key)) = (&config.access_key, &config.secret_key) {
            Credentials::new(access_key.clone(), secret_key.clone(), None, None, "custom")
        } else {
            return Err(anyhow!("AWS credentials not configured"));
        };

        // Build S3 config
        let s3_config = S3ConfigBuilder::new()
            .region(region)
            .credentials_provider(credentials)
            .endpoint_url(config.endpoint_url.clone())
            .force_path_style(config.force_path_style)
            .build();

        let client = Client::from_conf(s3_config);

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
    async fn test_s3_storage_creation() {
        let config = S3Config {
            bucket_name: "test-bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: Some("test-access-key".to_string()),
            secret_key: Some("test-secret-key".to_string()),
            endpoint_url: None,
            force_path_style: false,
        };

        // This test will fail without actual AWS credentials, but will compile
        // In a real test environment, you would need to mock the AWS client
    }

    #[tokio::test]
    async fn test_metadata_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), "llama-2".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        assert_eq!(metadata.get("model_name"), Some(&"llama-2".to_string()));
        assert_eq!(metadata.get("version"), Some(&"1.0".to_string()));
    }
}