use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::path::Path;

/// Cloud storage configuration for different providers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StorageProviderConfig {
    S3(S3Config),
    GCS(GCSConfig),
    Azure(AzureBlobConfig),
}

/// Amazon S3 storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    pub bucket_name: String,
    pub region: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    #[serde(default = "default_force_path_style")]
    pub force_path_style: bool,
}

fn default_force_path_style() -> bool {
    false
}

/// Google Cloud Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCSConfig {
    pub bucket_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_account_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

/// Azure Blob Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureBlobConfig {
    pub connection_string: String,
    pub container_name: String,
}

/// Cloud storage configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudStorageConfig {
    /// Primary storage provider configuration
    pub primary: StorageProviderConfig,
    
    /// Optional fallback storage provider configurations
    #[serde(skip_serializing_if = "Vec::is_empty", default = "default_fallbacks")]
    pub fallbacks: Vec<StorageProviderConfig>,
    
    /// Default cloud storage prefix/namespace
    #[serde(default = "default_cloud_prefix")]
    pub default_prefix: String,
    
    /// Enable automatic retries for failed operations
    #[serde(default = "default_enable_retries")]
    pub enable_retries: bool,
    
    /// Maximum number of retry attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    
    /// Base delay for exponential backoff (milliseconds)
    #[serde(default = "default_retry_delay_ms")]
    pub retry_delay_ms: u64,
    
    /// Enable signed URL generation
    #[serde(default = "default_enable_signed_urls")]
    pub enable_signed_urls: bool,
    
    /// Default signed URL expiration time (minutes)
    #[serde(default = "default_signed_url_expiry")]
    pub signed_url_expiry: u64,
    
    /// Enable upload/download progress tracking
    #[serde(default = "default_enable_progress")]
    pub enable_progress: bool,
    
    /// Default chunk size for file operations (MB)
    #[serde(default = "default_chunk_size_mb")]
    pub chunk_size_mb: u64,
}

fn default_fallbacks() -> Vec<StorageProviderConfig> {
    vec![]
}

fn default_cloud_prefix() -> String {
    "ollama-models".to_string()
}

fn default_enable_retries() -> bool {
    true
}

fn default_max_retries() -> u32 {
    3
}

fn default_retry_delay_ms() -> u64 {
    1000
}

fn default_enable_signed_urls() -> bool {
    true
}

fn default_signed_url_expiry() -> u64 {
    3600 // 1 hour
}

fn default_enable_progress() -> bool {
    true
}

fn default_chunk_size_mb() -> u64 {
    16 // 16MB chunks
}

impl CloudStorageConfig {
    /// Load configuration from file
    pub async fn from_file(config_path: &str) -> Result<Self> {
        let content = tokio::fs::read_to_string(config_path)
            .await
            .context(format!("Failed to read config file: {}", config_path))?;

        let config: CloudStorageConfig = toml::from_str(&content)
            .context("Failed to parse configuration file")?;

        // Validate configuration
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let primary_provider = std::env::var("STORAGE_PROVIDER")
            .unwrap_or_else(|_| "s3".to_string())
            .to_lowercase();

        let primary = match primary_provider.as_str() {
            "s3" => {
                let s3_config = S3Config {
                    bucket_name: std::env::var("S3_BUCKET")
                        .context("S3_BUCKET environment variable required for S3 provider")?,
                    region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    access_key: std::env::var("AWS_ACCESS_KEY_ID").ok(),
                    secret_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
                    endpoint_url: std::env::var("AWS_ENDPOINT_URL").ok(),
                    force_path_style: std::env::var("AWS_FORCE_PATH_STYLE")
                        .unwrap_or_else(|_| "false".to_string())
                        .parse()
                        .unwrap_or(false),
                };
                StorageProviderConfig::S3(s3_config)
            },
            "gcs" => {
                let gcs_config = GCSConfig {
                    bucket_name: std::env::var("GCS_BUCKET")
                        .context("GCS_BUCKET environment variable required for GCS provider")?,
                    service_account_path: std::env::var("GCS_SERVICE_ACCOUNT_PATH").ok(),
                    project_id: std::env::var("GCS_PROJECT_ID").ok(),
                };
                StorageProviderConfig::GCS(gcs_config)
            },
            "azure" => {
                let azure_config = AzureBlobConfig {
                    connection_string: std::env::var("AZURE_STORAGE_CONNECTION_STRING")
                        .context("AZURE_STORAGE_CONNECTION_STRING environment variable required for Azure provider")?,
                    container_name: std::env::var("AZURE_STORAGE_CONTAINER_NAME")
                        .context("AZURE_STORAGE_CONTAINER_NAME environment variable required for Azure provider")?,
                };
                StorageProviderConfig::Azure(azure_config)
            },
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported storage provider: {}. Supported: s3, gcs, azure",
                    primary_provider
                ));
            }
        };

        // Parse fallback providers
        let fallback_providers = std::env::var("FALLBACK_STORAGE_PROVIDERS")
            .unwrap_or_else(|_| "".to_string());

        let mut fallbacks = Vec::new();
        for provider in fallback_providers.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            match provider.to_lowercase().as_str() {
                "s3" => {
                    let s3_config = S3Config {
                        bucket_name: std::env::var("FALLBACK_S3_BUCKET").unwrap_or_default(),
                        region: std::env::var("FALLBACK_AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                        access_key: std::env::var("FALLBACK_AWS_ACCESS_KEY_ID").ok(),
                        secret_key: std::env::var("FALLBACK_AWS_SECRET_ACCESS_KEY").ok(),
                        endpoint_url: std::env::var("FALLBACK_AWS_ENDPOINT_URL").ok(),
                        force_path_style: std::env::var("FALLBACK_AWS_FORCE_PATH_STYLE")
                            .unwrap_or_else(|_| "false".to_string())
                            .parse()
                            .unwrap_or(false),
                    };
                    fallbacks.push(StorageProviderConfig::S3(s3_config));
                },
                "gcs" => {
                    let gcs_config = GCSConfig {
                        bucket_name: std::env::var("FALLBACK_GCS_BUCKET").unwrap_or_default(),
                        service_account_path: std::env::var("FALLBACK_GCS_SERVICE_ACCOUNT_PATH").ok(),
                        project_id: std::env::var("FALLBACK_GCS_PROJECT_ID").ok(),
                    };
                    fallbacks.push(StorageProviderConfig::GCS(gcs_config));
                },
                "azure" => {
                    let azure_config = AzureBlobConfig {
                        connection_string: std::env::var("FALLBACK_AZURE_STORAGE_CONNECTION_STRING").unwrap_or_default(),
                        container_name: std::env::var("FALLBACK_AZURE_STORAGE_CONTAINER_NAME").unwrap_or_default(),
                    };
                    fallbacks.push(StorageProviderConfig::Azure(azure_config));
                },
                _ => {
                    log::warn!("Unknown fallback storage provider: {}", provider);
                }
            }
        }

        let config = CloudStorageConfig {
            primary,
            fallbacks,
            default_prefix: std::env::var("CLOUD_STORAGE_PREFIX").unwrap_or_else(|_| "ollama-models".to_string()),
            enable_retries: std::env::var("CLOUD_STORAGE_ENABLE_RETRIES")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            max_retries: std::env::var("CLOUD_STORAGE_MAX_RETRIES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
            retry_delay_ms: std::env::var("CLOUD_STORAGE_RETRY_DELAY_MS")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()
                .unwrap_or(1000),
            enable_signed_urls: std::env::var("CLOUD_STORAGE_ENABLE_SIGNED_URLS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            signed_url_expiry: std::env::var("CLOUD_STORAGE_SIGNED_URL_EXPIRY")
                .unwrap_or_else(|_| "3600".to_string())
                .parse()
                .unwrap_or(3600),
            enable_progress: std::env::var("CLOUD_STORAGE_ENABLE_PROGRESS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            chunk_size_mb: std::env::var("CLOUD_STORAGE_CHUNK_SIZE_MB")
                .unwrap_or_else(|_| "16".to_string())
                .parse()
                .unwrap_or(16),
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate primary configuration
        match &self.primary {
            StorageProviderConfig::S3(config) => {
                if config.bucket_name.is_empty() {
                    return Err(anyhow::anyhow!("S3 bucket name cannot be empty"));
                }
                if config.region.is_empty() {
                    return Err(anyhow::anyhow!("S3 region cannot be empty"));
                }
            },
            StorageProviderConfig::GCS(config) => {
                if config.bucket_name.is_empty() {
                    return Err(anyhow::anyhow!("GCS bucket name cannot be empty"));
                }
                // Service account path is optional for GCS (can use default credentials)
            },
            StorageProviderConfig::Azure(config) => {
                if config.connection_string.is_empty() {
                    return Err(anyhow::anyhow!("Azure connection string cannot be empty"));
                }
                if config.container_name.is_empty() {
                    return Err(anyhow::anyhow!("Azure container name cannot be empty"));
                }
            },
        }

        // Validate fallback configurations
        for (i, fallback) in self.fallbacks.iter().enumerate() {
            match fallback {
                StorageProviderConfig::S3(config) => {
                    if config.bucket_name.is_empty() {
                        return Err(anyhow::anyhow!("Fallback S3 bucket name cannot be empty (index: {})", i));
                    }
                },
                StorageProviderConfig::GCS(config) => {
                    if config.bucket_name.is_empty() {
                        return Err(anyhow::anyhow!("Fallback GCS bucket name cannot be empty (index: {})", i));
                    }
                },
                StorageProviderConfig::Azure(config) => {
                    if config.connection_string.is_empty() {
                        return Err(anyhow::anyhow!("Fallback Azure connection string cannot be empty (index: {})", i));
                    }
                },
            }
        }

        // Validate numeric parameters
        if self.max_retries == 0 {
            return Err(anyhow::anyhow!("max_retries must be greater than 0"));
        }

        if self.retry_delay_ms == 0 {
            return Err(anyhow::anyhow!("retry_delay_ms must be greater than 0"));
        }

        if self.chunk_size_mb == 0 {
            return Err(anyhow::anyhow!("chunk_size_mb must be greater than 0"));
        }

        Ok(())
    }

    /// Create a configuration template
    pub fn template() -> Self {
        CloudStorageConfig {
            primary: StorageProviderConfig::S3(S3Config {
                bucket_name: "your-s3-bucket".to_string(),
                region: "us-east-1".to_string(),
                access_key: None,
                secret_key: None,
                endpoint_url: None,
                force_path_style: false,
            }),
            fallbacks: vec![
                StorageProviderConfig::GCS(GCSConfig {
                    bucket_name: "your-gcs-bucket".to_string(),
                    service_account_path: None,
                    project_id: None,
                }),
                StorageProviderConfig::Azure(AzureBlobConfig {
                    connection_string: "DefaultEndpointsProtocol=https;AccountName=youraccount;AccountKey=yourkey;EndpointSuffix=core.windows.net".to_string(),
                    container_name: "your-container".to_string(),
                }),
            ],
            default_prefix: "ollama-models".to_string(),
            enable_retries: true,
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_signed_urls: true,
            signed_url_expiry: 3600,
            enable_progress: true,
            chunk_size_mb: 16,
        }
    }

    /// Save configuration to file
    pub async fn save_to_file(&self, config_path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;

        tokio::fs::write(config_path, content)
            .await
            .context(format!("Failed to write configuration file: {}", config_path))?;

        Ok(())
    }

    /// Get the default signed URL expiration duration
    pub fn signed_url_duration(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.signed_url_expiry)
    }

    /// Get the chunk size in bytes
    pub fn chunk_size_bytes(&self) -> usize {
        (self.chunk_size_mb * 1024 * 1024) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = CloudStorageConfig::template();
        
        // Test invalid S3 config
        match &mut config.primary {
            StorageProviderConfig::S3(ref mut s3_config) => {
                s3_config.bucket_name = "".to_string();
            },
            _ => {},
        }

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_env_parsing() {
        std::env::set_var("STORAGE_PROVIDER", "s3");
        std::env::set_var("S3_BUCKET", "test-bucket");
        std::env::set_var("AWS_REGION", "us-west-2");

        let config = CloudStorageConfig::from_env();
        assert!(config.is_ok());

        if let Ok(config) = config {
            match config.primary {
                StorageProviderConfig::S3(ref s3_config) => {
                    assert_eq!(s3_config.bucket_name, "test-bucket");
                    assert_eq!(s3_config.region, "us-west-2");
                },
                _ => panic!("Expected S3 config"),
            }
        }
    }

    #[test]
    fn test_template_creation() {
        let template = CloudStorageConfig::template();
        
        assert_eq!(template.default_prefix, "ollama-models");
        assert_eq!(template.max_retries, 3);
        assert_eq!(template.chunk_size_mb, 16);
    }
}