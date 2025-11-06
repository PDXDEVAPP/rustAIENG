use axum::{
    extract::State,
    response::Json,
    routing::get,
    Router,
};
use k8s_openapi::api::core::v1::{Pod, Service};
use kube::{Client, Api, Config};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesInfo {
    pub cluster_name: String,
    pub namespace: String,
    pub node_name: String,
    pub pod_name: String,
    pub pod_ip: String,
    pub service_account: String,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesProbeConfig {
    pub path: String,
    pub port: u16,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
    pub success_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesServiceConfig {
    pub service_type: String,
    pub port: u16,
    pub target_port: u16,
    pub protocol: String,
    pub session_affinity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesDeploymentConfig {
    pub replicas: u32,
    pub strategy_type: String,
    pub rolling_update_max_unavailable: Option<u32>,
    pub rolling_update_max_surge: Option<u32>,
    pub revision_history_limit: Option<u32>,
}

pub struct KubernetesIntegration {
    client: Option<Client>,
    namespace: String,
    pod_name: String,
    node_name: String,
    cluster_name: String,
}

impl KubernetesIntegration {
    pub async fn new() -> Result<Self, KubernetesError> {
        // Try to detect if running in Kubernetes
        let client = match Client::try_default().await {
            Ok(client) => {
                info!("Connected to Kubernetes cluster");
                Some(client)
            }
            Err(e) => {
                warn!("Not running in Kubernetes or failed to connect: {}", e);
                None
            }
        };

        let (namespace, pod_name, node_name, cluster_name) = if client.is_some() {
            Self::detect_kubernetes_info(client.as_ref().unwrap()).await?
        } else {
            ("default".to_string(), "local".to_string(), "local".to_string(), "local".to_string())
        };

        Ok(Self {
            client,
            namespace,
            pod_name,
            node_name,
            cluster_name,
        })
    }

    async fn detect_kubernetes_info(client: &Client) -> Result<(String, String, String, String), KubernetesError> {
        // Get current pod info from environment variables and API
        let namespace = std::env::var("POD_NAMESPACE")
            .or_else(|_| std::env::var("NAMESPACE"))
            .unwrap_or_else(|_| "default".to_string());
        
        let pod_name = std::env::var("POD_NAME")
            .unwrap_or_else(|_| "local".to_string());
        
        let node_name = std::env::var("NODE_NAME")
            .unwrap_or_else(|_| "local".to_string());
        
        let cluster_name = "kubernetes".to_string();

        if client.is_some() {
            debug!("Kubernetes environment detected:");
            debug!("  Namespace: {}", namespace);
            debug!("  Pod: {}", pod_name);
            debug!("  Node: {}", node_name);
        }

        Ok((namespace, pod_name, node_name, cluster_name))
    }

    pub async fn get_cluster_info(&self) -> Result<KubernetesInfo, KubernetesError> {
        let mut labels = HashMap::new();
        let mut annotations = HashMap::new();

        // Try to get pod info from Kubernetes API
        if let Some(client) = &self.client {
            let pods: Api<Pod> = Api::namespaced(client.clone(), &self.namespace);
            
            if let Ok(pod) = pods.get(&self.pod_name).await {
                if let Some(metadata) = &pod.metadata {
                    for (k, v) in metadata.labels.as_ref().unwrap_or(&std::collections::HashMap::new()) {
                        labels.insert(k.clone(), v.clone());
                    }
                    
                    for (k, v) in metadata.annotations.as_ref().unwrap_or(&std::collections::HashMap::new()) {
                        annotations.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        let pod_ip = std::env::var("POD_IP").unwrap_or_else(|_| "127.0.0.1".to_string());
        let service_account = std::env::var("SERVICE_ACCOUNT").unwrap_or_else(|_| "default".to_string());

        Ok(KubernetesInfo {
            cluster_name: self.cluster_name.clone(),
            namespace: self.namespace.clone(),
            node_name: self.node_name.clone(),
            pod_name: self.pod_name.clone(),
            pod_ip,
            service_account,
            labels,
            annotations,
        })
    }

    pub async fn check_pod_health(&self) -> Result<bool, KubernetesError> {
        if let Some(client) = &self.client {
            let pods: Api<Pod> = Api::namespaced(client.clone(), &self.namespace);
            
            match pods.get(&self.pod_name).await {
                Ok(pod) => {
                    let status = pod.status.as_ref().unwrap();
                    
                    // Check if pod is running
                    if let Some(phase) = &status.phase {
                        if phase == "Running" {
                            // Check if all containers are ready
                            let containers_ready = status.container_statuses
                                .as_ref()
                                .map_or(false, |statuses| {
                                    statuses.iter().all(|cs| cs.ready.unwrap_or(false))
                                });
                            
                            if containers_ready {
                                info!("Pod is healthy and all containers are ready");
                                return Ok(true);
                            } else {
                                warn!("Pod is running but not all containers are ready");
                                return Ok(false);
                            }
                        } else {
                            warn!("Pod phase is: {}", phase);
                            return Ok(false);
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to get pod status: {}", e);
                    return Ok(false);
                }
            }
        }
        
        Ok(false) // Not running in Kubernetes
    }

    pub async fn generate_manifests(&self) -> Result<KubernetesManifests, KubernetesError> {
        Ok(KubernetesManifests {
            deployment: self.generate_deployment_manifest().await?,
            service: self.generate_service_manifest().await?,
            hpa: self.generate_hpa_manifest().await?,
            pdb: self.generate_pdb_manifest().await?,
        })
    }

    async fn generate_deployment_manifest(&self) -> Result<String, KubernetesError> {
        let manifest = format!(r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rust-ollama
  namespace: {namespace}
  labels:
    app: rust-ollama
    version: v0.2.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: rust-ollama
  template:
    metadata:
      labels:
        app: rust-ollama
        version: v0.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: rust-ollama
      containers:
      - name: rust-ollama
        image: rust-ollama:v0.2.0
        ports:
        - containerPort: 11434
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        env:
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 11434
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 11434
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: rust-ollama-models
      - name: config
        configMap:
          name: rust-ollama-config
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rust-ollama
  namespace: {namespace}
automountServiceAccountToken: false
"#, namespace = self.namespace);

        Ok(manifest.to_string())
    }

    async fn generate_service_manifest(&self) -> Result<String, KubernetesError> {
        let manifest = format!(r#"
apiVersion: v1
kind: Service
metadata:
  name: rust-ollama
  namespace: {namespace}
  labels:
    app: rust-ollama
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
spec:
  type: ClusterIP
  ports:
  - port: 11434
    targetPort: 11434
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: rust-ollama
---
apiVersion: v1
kind: Service
metadata:
  name: rust-ollama-headless
  namespace: {namespace}
  labels:
    app: rust-ollama
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - port: 11434
    targetPort: 11434
    protocol: TCP
    name: http
  selector:
    app: rust-ollama
"#, namespace = self.namespace);

        Ok(manifest.to_string())
    }

    async fn generate_hpa_manifest(&self) -> Result<String, KubernetesError> {
        let manifest = format!(r#"
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rust-ollama
  namespace: {namespace}
  labels:
    app: rust-ollama
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rust-ollama
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
"#, namespace = self.namespace);

        Ok(manifest.to_string())
    }

    async fn generate_pdb_manifest(&self) -> Result<String, KubernetesError> {
        let manifest = format!(r#"
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rust-ollama
  namespace: {namespace}
  labels:
    app: rust-ollama
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: rust-ollama
"#, namespace = self.namespace);

        Ok(manifest.to_string())
    }

    pub async fn check_service_health(&self) -> Result<bool, KubernetesError> {
        if let Some(client) = &self.client {
            let services: Api<Service> = Api::namespaced(client.clone(), &self.namespace);
            
            match services.get("rust-ollama").await {
                Ok(service) => {
                    if let Some(spec) = &service.spec {
                        if let Some(port) = spec.ports.as_ref().and_then(|p| p.first()) {
                            info!("Service port: {}", port.port);
                        }
                    }
                    Ok(true)
                }
                Err(e) => {
                    error!("Failed to get service status: {}", e);
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesManifests {
    pub deployment: String,
    pub service: String,
    pub hpa: String,
    pub pdb: String,
}

#[derive(Debug, thiserror::Error)]
pub enum KubernetesError {
    #[error("Kubernetes client error: {0}")]
    ClientError(String),
    
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("Detection error: {0}")]
    DetectionError(String),
    
    #[error("Manifest generation error: {0}")]
    ManifestError(String),
}

// Kubernetes-specific routes for the web server
pub fn kubernetes_routes(k8s: Arc<KubernetesIntegration>) -> Router {
    Router::new()
        .route("/kubernetes/info", get(kubernetes_info_endpoint))
        .route("/kubernetes/health", get(kubernetes_health_endpoint))
        .route("/kubernetes/manifests", get(kubernetes_manifests_endpoint))
        .with_state(k8s)
}

async fn kubernetes_info_endpoint(
    State(k8s): State<Arc<KubernetesIntegration>>,
) -> Result<Json<KubernetesInfo>, axum::http::StatusCode> {
    match k8s.get_cluster_info().await {
        Ok(info) => Ok(Json(info)),
        Err(e) => {
            error!("Failed to get Kubernetes info: {}", e);
            Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn kubernetes_health_endpoint(
    State(k8s): State<Arc<KubernetesIntegration>>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    let pod_healthy = k8s.check_pod_health().await.unwrap_or(false);
    let service_healthy = k8s.check_service_health().await.unwrap_or(false);

    let health_status = serde_json::json!({
        "pod_healthy": pod_healthy,
        "service_healthy": service_healthy,
        "overall_healthy": pod_healthy && service_healthy,
        "cluster_info": k8s.get_cluster_info().await.ok()
    });

    if health_status["overall_healthy"].as_bool().unwrap_or(false) {
        Ok(Json(health_status))
    } else {
        Err(axum::http::StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn kubernetes_manifests_endpoint(
    State(k8s): State<Arc<KubernetesIntegration>>,
) -> Result<Json<KubernetesManifests>, axum::http::StatusCode> {
    match k8s.generate_manifests().await {
        Ok(manifests) => Ok(Json(manifests)),
        Err(e) => {
            error!("Failed to generate Kubernetes manifests: {}", e);
            Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}