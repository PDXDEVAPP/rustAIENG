use crate::core::DatabaseManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    pub id: u32,
    pub name: String,
    pub sql: String,
    pub checksum: String,
    pub applied_at: chrono::DateTime<chrono::Utc>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub migrations_to_apply: Vec<Migration>,
    pub migrations_to_rollback: Vec<Migration>,
    pub current_version: u32,
    pub target_version: u32,
}

pub struct DatabaseMigrations {
    database: DatabaseManager,
    migrations_table: String,
}

impl DatabaseMigrations {
    pub fn new(database: DatabaseManager) -> Self {
        Self {
            database,
            migrations_table: "schema_migrations".to_string(),
        }
    }

    pub async fn initialize(&self) -> Result<(), MigrationError> {
        info!("Initializing database migrations...");

        // Create migrations table if it doesn't exist
        let create_table_sql = format!(r#"
            CREATE TABLE IF NOT EXISTS {} (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                sql TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at DATETIME NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                rollback_sql TEXT
            );
        "#, self.migrations_table);

        sqlx::query(&create_table_sql)
            .execute(&self.database.pool)
            .await
            .map_err(|e| MigrationError::DatabaseError(format!("Failed to create migrations table: {}", e)))?;

        info!("Database migrations table created/verified");
        Ok(())
    }

    pub async fn get_current_version(&self) -> Result<u32, MigrationError> {
        let result = sqlx::query(&format!(
            "SELECT MAX(id) as version FROM {}",
            self.migrations_table
        ))
        .fetch_optional(&self.database.pool)
        .await
        .map_err(|e| MigrationError::DatabaseError(format!("Failed to get current version: {}", e)))?;

        match result {
            Some(row) => {
                let version: Option<i64> = row.get("version");
                Ok(version.unwrap_or(0) as u32)
            }
            None => Ok(0),
        }
    }

    pub async fn get_applied_migrations(&self) -> Result<Vec<Migration>, MigrationError> {
        let rows = sqlx::query(&format!(
            "SELECT * FROM {} ORDER BY id ASC",
            self.migrations_table
        ))
        .fetch_all(&self.database.pool)
        .await
        .map_err(|e| MigrationError::DatabaseError(format!("Failed to fetch applied migrations: {}", e)))?;

        let migrations = rows.into_iter().map(|row| {
            Migration {
                id: row.get("id"),
                name: row.get("name"),
                sql: row.get("sql"),
                checksum: row.get("checksum"),
                applied_at: row.get("applied_at"),
                execution_time_ms: row.get("execution_time_ms"),
            }
        }).collect();

        Ok(migrations)
    }

    pub async fn apply_migration(&self, migration: &Migration) -> Result<(), MigrationError> {
        info!("Applying migration: {} - {}", migration.id, migration.name);

        let start_time = std::time::Instant::now();

        // Begin transaction
        let mut tx = self.database.pool.begin().await
            .map_err(|e| MigrationError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

        // Apply the migration
        let migration_statements = self.split_sql_statements(&migration.sql);
        
        for statement in migration_statements {
            if !statement.trim().is_empty() {
                sqlx::query(statement)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| MigrationError::MigrationFailed {
                        migration_id: migration.id,
                        error: format!("Failed to execute statement: {}", e)
                    })?;
            }
        }

        // Record the migration
        let applied_at = chrono::Utc::now();
        let execution_time = start_time.elapsed().as_millis() as u64;

        sqlx::query(&format!(
            "INSERT INTO {} (id, name, sql, checksum, applied_at, execution_time_ms) VALUES (?, ?, ?, ?, ?, ?)",
            self.migrations_table
        ))
        .bind(migration.id)
        .bind(&migration.name)
        .bind(&migration.sql)
        .bind(&migration.checksum)
        .bind(applied_at)
        .bind(execution_time)
        .execute(&mut *tx)
        .await
        .map_err(|e| MigrationError::DatabaseError(format!("Failed to record migration: {}", e)))?;

        // Commit transaction
        tx.commit().await
            .map_err(|e| MigrationError::DatabaseError(format!("Failed to commit transaction: {}", e)))?;

        info!("Migration applied successfully: {} ({} ms)", migration.name, execution_time);
        Ok(())
    }

    pub async fn rollback_migration(&self, migration: &Migration) -> Result<(), MigrationError> {
        info!("Rolling back migration: {} - {}", migration.id, migration.name);

        // This is a simplified rollback - in production, you'd store rollback SQL with each migration
        warn!("Rollback functionality requires manual implementation for this migration");

        // Remove migration record
        sqlx::query(&format!(
            "DELETE FROM {} WHERE id = ?",
            self.migrations_table
        ))
        .bind(migration.id)
        .execute(&self.database.pool)
        .await
        .map_err(|e| MigrationError::DatabaseError(format!("Failed to rollback migration record: {}", e)))?;

        info!("Migration rolled back: {}", migration.name);
        Ok(())
    }

    pub async fn create_migration_plan(&self, target_version: Option<u32>) -> Result<MigrationPlan, MigrationError> {
        let current_version = self.get_current_version().await?;
        let target = target_version.unwrap_or(u32::MAX); // Apply all pending migrations if not specified

        let applied_migrations = self.get_applied_migrations().await?;
        
        // Get all available migrations from filesystem
        let available_migrations = self.load_migrations_from_filesystem().await?;

        let migrations_to_apply: Vec<Migration> = available_migrations
            .into_iter()
            .filter(|migration| migration.id > current_version && migration.id <= target)
            .collect();

        let migrations_to_rollback: Vec<Migration> = applied_migrations
            .into_iter()
            .filter(|migration| migration.id > target)
            .collect();

        Ok(MigrationPlan {
            migrations_to_apply,
            migrations_to_rollback,
            current_version,
            target_version: target,
        })
    }

    pub async fn run_pending_migrations(&self, target_version: Option<u32>) -> Result<Vec<Migration>, MigrationError> {
        info!("Running pending migrations...");

        let plan = self.create_migration_plan(target_version).await?;
        
        if plan.migrations_to_apply.is_empty() {
            info!("No migrations to apply");
            return Ok(vec![]);
        }

        info!("Found {} migrations to apply", plan.migrations_to_apply.len());

        let mut applied_migrations = Vec::new();

        for migration in &plan.migrations_to_apply {
            self.apply_migration(migration).await?;
            applied_migrations.push(migration.clone());
        }

        Ok(applied_migrations)
    }

    pub async fn rollback_to_version(&self, target_version: u32) -> Result<Vec<Migration>, MigrationError> {
        info!("Rolling back to version {}", target_version);

        let plan = self.create_migration_plan(Some(target_version)).await?;
        
        if plan.migrations_to_rollback.is_empty() {
            info!("No migrations to rollback");
            return Ok(vec![]);
        }

        info!("Rolling back {} migrations", plan.migrations_to_rollback.len());

        let mut rolled_back_migrations = Vec::new();

        // Rollback in reverse order
        for migration in plan.migrations_to_rollback.iter().rev() {
            self.rollback_migration(migration).await?;
            rolled_back_migrations.push(migration.clone());
        }

        Ok(rolled_back_migrations)
    }

    async fn load_migrations_from_filesystem(&self) -> Result<Vec<Migration>, MigrationError> {
        let migrations_dir = Path::new("migrations");
        
        if !migrations_dir.exists() {
            info!("Migrations directory does not exist, creating it...");
            std::fs::create_dir_all(migrations_dir)
                .map_err(|e| MigrationError::FilesystemError(format!("Failed to create migrations directory: {}", e)))?;
        }

        let mut migrations = Vec::new();

        // Read migration files
        if let Ok(entries) = std::fs::read_dir(migrations_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "sql") {
                        let migration = self.load_migration_from_file(&path).await?;
                        migrations.push(migration);
                    }
                }
            }
        }

        // Sort by migration ID
        migrations.sort_by_key(|m| m.id);
        Ok(migrations)
    }

    async fn load_migration_from_file(&self, path: &Path) -> Result<Migration, MigrationError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MigrationError::FilesystemError(format!("Failed to read migration file: {}", e)))?;

        let filename = path.file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| MigrationError::FilesystemError("Invalid migration filename".to_string()))?;

        // Parse filename: timestamp_name.sql
        let parts: Vec<&str> = filename.split('_').collect();
        if parts.len() < 2 {
            return Err(MigrationError::InvalidMigrationName {
                filename: filename.to_string(),
                reason: "Invalid filename format. Expected: timestamp_name.sql".to_string()
            });
        }

        let id_str = parts[0];
        let id = id_str.parse::<u32>()
            .map_err(|_| MigrationError::InvalidMigrationName {
                filename: filename.to_string(),
                reason: format!("Invalid migration ID: {}", id_str)
            })?;

        let name = parts[1..].join("_").trim_end_matches(".sql").to_string();

        // Calculate checksum
        let checksum = format!("{:x}", md5::compute(&content));

        Ok(Migration {
            id,
            name,
            sql: content,
            checksum,
            applied_at: chrono::Utc::now(), // Will be updated when actually applied
            execution_time_ms: 0, // Will be updated when actually applied
        })
    }

    fn split_sql_statements(&self, sql: &str) -> Vec<&str> {
        sql.split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    }

    pub async fn verify_migrations(&self) -> Result<bool, MigrationError> {
        info!("Verifying applied migrations...");

        let applied_migrations = self.get_applied_migrations().await?;

        for migration in applied_migrations {
            // Calculate expected checksum
            let actual_checksum = format!("{:x}", md5::compute(&migration.sql));

            if actual_checksum != migration.checksum {
                error!("Migration checksum mismatch for {}: expected {}, got {}", 
                       migration.name, migration.checksum, actual_checksum);
                return Ok(false);
            }
        }

        info!("All migrations verified successfully");
        Ok(true)
    }

    pub async fn generate_migration(&self, name: &str, sql: &str) -> Result<String, MigrationError> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("{}_{}.sql", timestamp, name);
        let filepath = Path::new("migrations").join(&filename);

        std::fs::write(&filepath, sql)
            .map_err(|e| MigrationError::FilesystemError(format!("Failed to write migration file: {}", e)))?;

        info!("Generated migration file: {}", filepath.display());
        Ok(filepath.to_string_lossy().to_string())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Filesystem error: {0}")]
    FilesystemError(String),
    
    #[error("Migration failed for ID {migration_id}: {error}")]
    MigrationFailed {
        migration_id: u32,
        error: String,
    },
    
    #[error("Invalid migration name: {filename} - {reason}")]
    InvalidMigrationName {
        filename: String,
        reason: String,
    },
}

// Migration templates for common operations
pub struct MigrationTemplates;

impl MigrationTemplates {
    pub fn create_models_table() -> &'static str {
        r#"
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            model_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size BIGINT NOT NULL,
            quantization TEXT,
            parameters TEXT,
            context_length INTEGER,
            gpu_layers INTEGER,
            download_url TEXT,
            sha256 TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            usage_count INTEGER DEFAULT 0,
            last_used DATETIME,
            metadata TEXT
        );
        "#
    }

    pub fn create_running_models_table() -> &'static str {
        r#"
        CREATE TABLE IF NOT EXISTS running_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            instance_id TEXT NOT NULL UNIQUE,
            device TEXT NOT NULL,
            memory_usage BIGINT,
            context_length INTEGER,
            prompt_tokens INTEGER,
            generation_tokens INTEGER,
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id)
        );
        "#
    }

    pub fn create_performance_metrics_table() -> &'static str {
        r#"
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            response_time_ms INTEGER NOT NULL,
            tokens_generated INTEGER,
            prompt_tokens INTEGER,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id)
        );
        "#
    }

    pub fn create_indexes() -> &'static str {
        r#"
        CREATE INDEX IF NOT EXISTS idx_models_name ON models (name);
        CREATE INDEX IF NOT EXISTS idx_models_active ON models (is_active);
        CREATE INDEX IF NOT EXISTS idx_running_models_model_id ON running_models (model_id);
        CREATE INDEX IF NOT EXISTS idx_running_models_instance ON running_models (instance_id);
        CREATE INDEX IF NOT EXISTS idx_performance_model_timestamp ON performance_metrics (model_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_endpoint ON performance_metrics (endpoint);
        "#
    }
}