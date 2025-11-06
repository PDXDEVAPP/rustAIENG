use sqlx::{Row, Sqlite, SqlitePool};
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub display_name: Option<String>,
    pub file_path: PathBuf,
    pub size_bytes: u64,
    pub model_type: ModelType,
    pub quantization: Option<String>,
    pub context_length: Option<usize>,
    pub max_tokens: Option<usize>,
    pub parameters: Option<String>, // JSON string with additional parameters
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub in_use: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunningModel {
    pub model_id: String,
    pub process_id: u32,
    pub memory_usage: u64,
    pub load_time_ms: u64,
    pub last_used: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde:: Deserialize)]
pub struct ModelSession {
    pub session_id: String,
    pub model_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub context_history: Option<String>, // JSON string
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ModelType {
    LLaMA,
    Mistral,
    CodeLLaMA,
    Gemma,
    Phi,
    Custom,
}

impl ModelType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => ModelType::LLaMA,
            "mistral" | "mixtral" => ModelType::Mistral,
            "codellama" => ModelType::CodeLLaMA,
            "gemma" => ModelType::Gemma,
            "phi" => ModelType::Phi,
            _ => ModelType::Custom,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatabaseManager {
    pool: SqlitePool,
}

impl DatabaseManager {
    pub async fn new(db_path: &str) -> anyhow::Result<Self> {
        let pool = SqlitePool::connect(db_path).await?;
        
        // Run migrations
        Self::run_migrations(&pool).await?;
        
        Ok(DatabaseManager { pool })
    }

    async fn run_migrations(pool: &SqlitePool) -> anyhow::Result<()> {
        // Create models table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                display_name TEXT,
                file_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                quantization TEXT,
                context_length INTEGER,
                max_tokens INTEGER,
                parameters TEXT,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                in_use INTEGER DEFAULT 0
            )
        "#).execute(pool).await?;

        // Create running_models table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS running_models (
                model_id TEXT PRIMARY KEY,
                process_id INTEGER NOT NULL,
                memory_usage INTEGER NOT NULL,
                load_time_ms INTEGER NOT NULL,
                last_used TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        "#).execute(pool).await?;

        // Create model_sessions table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS model_sessions (
                session_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                total_requests INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                context_history TEXT,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        "#).execute(pool).await?;

        // Create performance_metrics table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                request_id TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_duration_ms INTEGER DEFAULT 0,
                load_duration_ms INTEGER DEFAULT 0,
                prompt_eval_duration_ms INTEGER DEFAULT 0,
                completion_eval_duration_ms INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        "#).execute(pool).await?;

        Ok(())
    }

    pub async fn add_model(&self, model: Model) -> anyhow::Result<()> {
        let now = chrono::Utc::now();
        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO models 
            (id, name, display_name, file_path, size_bytes, model_type, quantization, 
             context_length, max_tokens, parameters, description, created_at, updated_at, in_use)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            model.id,
            model.name,
            model.display_name,
            model.file_path.to_string_lossy(),
            model.size_bytes as i64,
            format!("{:?}", model.model_type),
            model.quantization,
            model.context_length.map(|x| x as i64),
            model.max_tokens.map(|x| x as i64),
            model.parameters,
            model.description,
            now,
            now,
            if model.in_use { 1 } else { 0 }
        ).execute(&self.pool).await?;

        Ok(())
    }

    pub async fn list_models(&self) -> anyhow::Result<Vec<Model>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, name, display_name, file_path, size_bytes, model_type, quantization,
                   context_length, max_tokens, parameters, description, created_at, updated_at, in_use
            FROM models ORDER BY created_at DESC
            "#
        ).fetch_all(&self.pool).await?;

        rows.iter().map(|row| {
            Ok(Model {
                id: row.id.clone(),
                name: row.name.clone(),
                display_name: row.display_name.clone(),
                file_path: PathBuf::from(&row.file_path),
                size_bytes: row.size_bytes as u64,
                model_type: ModelType::from_str(&row.model_type),
                quantization: row.quantization.clone(),
                context_length: row.context_length.map(|x| x as usize),
                max_tokens: row.max_tokens.map(|x| x as usize),
                parameters: row.parameters.clone(),
                description: row.description.clone(),
                created_at: chrono::DateTime::parse_from_rfc3339(&row.created_at)
                    .map_err(|e| anyhow::anyhow!("Invalid date: {}", e))?
                    .with_timezone(&chrono::Utc),
                updated_at: chrono::DateTime::parse_from_rfc3339(&row.updated_at)
                    .map_err(|e| anyhow::anyhow!("Invalid date: {}", e))?
                    .with_timezone(&chrono::Utc),
                in_use: row.in_use != 0,
            })
        }).collect()
    }

    pub async fn get_model(&self, model_id: &str) -> anyhow::Result<Option<Model>> {
        let row = sqlx::query!(
            r#"
            SELECT id, name, display_name, file_path, size_bytes, model_type, quantization,
                   context_length, max_tokens, parameters, description, created_at, updated_at, in_use
            FROM models WHERE id = ?
            "#,
            model_id
        ).fetch_optional(&self.pool).await?;

        if let Some(row) = row {
            Ok(Some(Model {
                id: row.id.clone(),
                name: row.name.clone(),
                display_name: row.display_name.clone(),
                file_path: PathBuf::from(&row.file_path),
                size_bytes: row.size_bytes as u64,
                model_type: ModelType::from_str(&row.model_type),
                quantization: row.quantization.clone(),
                context_length: row.context_length.map(|x| x as usize),
                max_tokens: row.max_tokens.map(|x| x as usize),
                parameters: row.parameters.clone(),
                description: row.description.clone(),
                created_at: chrono::DateTime::parse_from_rfc3339(&row.created_at)
                    .map_err(|e| anyhow::anyhow!("Invalid date: {}", e))?
                    .with_timezone(&chrono::Utc),
                updated_at: chrono::DateTime::parse_from_rfc3339(&row.updated_at)
                    .map_err(|e| anyhow::anyhow!("Invalid date: {}", e))?
                    .with_timezone(&chrono::Utc),
                in_use: row.in_use != 0,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn remove_model(&self, model_id: &str) -> anyhow::Result<bool> {
        let result = sqlx::query!("DELETE FROM models WHERE id = ?", model_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    pub async fn mark_model_in_use(&self, model_id: &str, in_use: bool) -> anyhow::Result<()> {
        sqlx::query!("UPDATE models SET in_use = ? WHERE id = ?", 
                    if in_use { 1 } else { 0 }, model_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}