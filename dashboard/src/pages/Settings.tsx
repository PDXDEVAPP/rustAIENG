import React, { useState, useEffect } from 'react'
import { Save, RefreshCw, Key, Database, Cloud, Monitor, Shield } from 'lucide-react'
import { useWebSocket } from '../contexts/WebSocketContext'

interface ServerConfig {
  host: string
  port: number
  maxConnections: number
  timeout: number
  logLevel: 'debug' | 'info' | 'warn' | 'error'
  corsOrigins: string[]
}

interface ModelConfig {
  maxModelSize: number
  cacheSize: number
  quantizationEnabled: boolean
  gpuEnabled: boolean
  batchSize: number
  maxTokens: number
}

interface SecurityConfig {
  apiKeyRequired: boolean
  allowedIPs: string[]
  rateLimit: number
  jwtSecret: string
  passwordHash: string
}

interface MonitoringConfig {
  metricsEnabled: boolean
  prometheusPort: number
  tracingEnabled: boolean
  logRetention: number
}

export function Settings() {
  const { socket, isConnected } = useWebSocket()
  const [activeTab, setActiveTab] = useState<'server' | 'models' | 'security' | 'monitoring'>('server')
  
  const [serverConfig, setServerConfig] = useState<ServerConfig>({
    host: '0.0.0.0',
    port: 11435,
    maxConnections: 100,
    timeout: 30000,
    logLevel: 'info',
    corsOrigins: ['http://localhost:3000'],
  })

  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    maxModelSize: 8589934592, // 8GB
    cacheSize: 2147483648, // 2GB
    quantizationEnabled: true,
    gpuEnabled: true,
    batchSize: 32,
    maxTokens: 4096,
  })

  const [securityConfig, setSecurityConfig] = useState<SecurityConfig>({
    apiKeyRequired: false,
    allowedIPs: [],
    rateLimit: 1000,
    jwtSecret: '',
    passwordHash: '',
  })

  const [monitoringConfig, setMonitoringConfig] = useState<MonitoringConfig>({
    metricsEnabled: true,
    prometheusPort: 9090,
    tracingEnabled: true,
    logRetention: 30,
  })

  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    if (!socket) return

    socket.on('config_data', (data: {
      server: ServerConfig
      models: ModelConfig
      security: SecurityConfig
      monitoring: MonitoringConfig
    }) => {
      setServerConfig(data.server)
      setModelConfig(data.models)
      setSecurityConfig(data.security)
      setMonitoringConfig(data.monitoring)
      setLoading(false)
    })

    socket.on('config_saved', () => {
      setSaving(false)
      setHasChanges(false)
    })

    // Request current configuration
    socket.emit('get_config')
    setLoading(true)

    return () => {
      socket.off('config_data')
      socket.off('config_saved')
    }
  }, [socket])

  const handleSave = () => {
    if (!socket) return
    
    setSaving(true)
    socket.emit('save_config', {
      server: serverConfig,
      models: modelConfig,
      security: securityConfig,
      monitoring: monitoringConfig,
    })
  }

  const handleReset = () => {
    if (!socket) return
    socket.emit('get_config')
    setHasChanges(false)
  }

  const handleConfigChange = (
    section: 'server' | 'models' | 'security' | 'monitoring',
    key: string,
    value: any
  ) => {
    switch (section) {
      case 'server':
        setServerConfig(prev => ({ ...prev, [key]: value }))
        break
      case 'models':
        setModelConfig(prev => ({ ...prev, [key]: value }))
        break
      case 'security':
        setSecurityConfig(prev => ({ ...prev, [key]: value }))
        break
      case 'monitoring':
        setMonitoringConfig(prev => ({ ...prev, [key]: value }))
        break
    }
    setHasChanges(true)
  }

  const tabs = [
    { id: 'server', name: 'Server', icon: Monitor },
    { id: 'models', name: 'Models', icon: Database },
    { id: 'security', name: 'Security', icon: Shield },
    { id: 'monitoring', name: 'Monitoring', icon: Key },
  ] as const

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Settings</h1>
          <p className="text-text-secondary mt-2">
            Configure your Ollama server, models, security, and monitoring options
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <button 
            onClick={handleReset}
            disabled={!hasChanges || loading}
            className="btn-secondary"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset
          </button>
          
          <button 
            onClick={handleSave}
            disabled={!hasChanges || saving || loading}
            className="btn-primary"
          >
            <Save className="w-4 h-4 mr-2" />
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Configuration Status */}
      {hasChanges && (
        <div className="bg-semantic-warning/10 border border-semantic-warning rounded-lg p-4">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-semantic-warning rounded-full mr-3" />
            <span className="text-semantic-warning font-medium">
              You have unsaved changes
            </span>
          </div>
        </div>
      )}

      {/* Settings Interface */}
      <div className="bg-bg-surface border border-border-default rounded-lg overflow-hidden">
        {/* Tab Navigation */}
        <div className="border-b border-border-default">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center py-4 border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-500'
                      : 'border-transparent text-text-secondary hover:text-text-primary'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-2" />
                  {tab.name}
                </button>
              )
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {loading ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-text-secondary">Loading configuration...</p>
            </div>
          ) : (
            <>
              {/* Server Configuration */}
              {activeTab === 'server' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-text-primary">Server Configuration</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Host</label>
                      <input
                        type="text"
                        value={serverConfig.host}
                        onChange={(e) => handleConfigChange('server', 'host', e.target.value)}
                        className="input-field w-full"
                        placeholder="0.0.0.0"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Port</label>
                      <input
                        type="number"
                        value={serverConfig.port}
                        onChange={(e) => handleConfigChange('server', 'port', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1"
                        max="65535"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Max Connections</label>
                      <input
                        type="number"
                        value={serverConfig.maxConnections}
                        onChange={(e) => handleConfigChange('server', 'maxConnections', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Timeout (ms)</label>
                      <input
                        type="number"
                        value={serverConfig.timeout}
                        onChange={(e) => handleConfigChange('server', 'timeout', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1000"
                        step="1000"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Log Level</label>
                      <select
                        value={serverConfig.logLevel}
                        onChange={(e) => handleConfigChange('server', 'logLevel', e.target.value)}
                        className="input-field w-full"
                      >
                        <option value="debug">Debug</option>
                        <option value="info">Info</option>
                        <option value="warn">Warning</option>
                        <option value="error">Error</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-text-secondary text-sm mb-2">CORS Origins</label>
                    <textarea
                      value={serverConfig.corsOrigins.join('\n')}
                      onChange={(e) => handleConfigChange('server', 'corsOrigins', e.target.value.split('\n').filter(Boolean))}
                      className="input-field w-full h-24 resize-none"
                      placeholder="http://localhost:3000&#10;https://yourdomain.com"
                    />
                    <p className="text-text-secondary text-xs mt-1">
                      Enter one origin per line
                    </p>
                  </div>
                </div>
              )}

              {/* Model Configuration */}
              {activeTab === 'models' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-text-primary">Model Configuration</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Max Model Size (bytes)</label>
                      <input
                        type="number"
                        value={modelConfig.maxModelSize}
                        onChange={(e) => handleConfigChange('models', 'maxModelSize', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1048576"
                        step="1048576"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Cache Size (bytes)</label>
                      <input
                        type="number"
                        value={modelConfig.cacheSize}
                        onChange={(e) => handleConfigChange('models', 'cacheSize', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1048576"
                        step="1048576"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Batch Size</label>
                      <input
                        type="number"
                        value={modelConfig.batchSize}
                        onChange={(e) => handleConfigChange('models', 'batchSize', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1"
                        max="128"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Max Tokens</label>
                      <input
                        type="number"
                        value={modelConfig.maxTokens}
                        onChange={(e) => handleConfigChange('models', 'maxTokens', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="512"
                        max="32768"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="quantizationEnabled"
                        checked={modelConfig.quantizationEnabled}
                        onChange={(e) => handleConfigChange('models', 'quantizationEnabled', e.target.checked)}
                        className="mr-3"
                      />
                      <label htmlFor="quantizationEnabled" className="text-text-primary">
                        Enable Model Quantization
                      </label>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="gpuEnabled"
                        checked={modelConfig.gpuEnabled}
                        onChange={(e) => handleConfigChange('models', 'gpuEnabled', e.target.checked)}
                        className="mr-3"
                      />
                      <label htmlFor="gpuEnabled" className="text-text-primary">
                        Enable GPU Acceleration
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* Security Configuration */}
              {activeTab === 'security' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-text-primary">Security Configuration</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="apiKeyRequired"
                        checked={securityConfig.apiKeyRequired}
                        onChange={(e) => handleConfigChange('security', 'apiKeyRequired', e.target.checked)}
                        className="mr-3"
                      />
                      <label htmlFor="apiKeyRequired" className="text-text-primary">
                        Require API Key for Access
                      </label>
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Rate Limit (requests/minute)</label>
                      <input
                        type="number"
                        value={securityConfig.rateLimit}
                        onChange={(e) => handleConfigChange('security', 'rateLimit', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1"
                        max="10000"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">JWT Secret</label>
                      <input
                        type="password"
                        value={securityConfig.jwtSecret}
                        onChange={(e) => handleConfigChange('security', 'jwtSecret', e.target.value)}
                        className="input-field w-full"
                        placeholder="Your secret key"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Allowed IP Addresses</label>
                      <textarea
                        value={securityConfig.allowedIPs.join('\n')}
                        onChange={(e) => handleConfigChange('security', 'allowedIPs', e.target.value.split('\n').filter(Boolean))}
                        className="input-field w-full h-24 resize-none"
                        placeholder="192.168.1.100&#10;10.0.0.0/8"
                      />
                      <p className="text-text-secondary text-xs mt-1">
                        Leave empty to allow all IPs
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Monitoring Configuration */}
              {activeTab === 'monitoring' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-text-primary">Monitoring Configuration</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Prometheus Port</label>
                      <input
                        type="number"
                        value={monitoringConfig.prometheusPort}
                        onChange={(e) => handleConfigChange('monitoring', 'prometheusPort', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1024"
                        max="65535"
                      />
                    </div>

                    <div>
                      <label className="block text-text-secondary text-sm mb-2">Log Retention (days)</label>
                      <input
                        type="number"
                        value={monitoringConfig.logRetention}
                        onChange={(e) => handleConfigChange('monitoring', 'logRetention', parseInt(e.target.value))}
                        className="input-field w-full"
                        min="1"
                        max="365"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="metricsEnabled"
                        checked={monitoringConfig.metricsEnabled}
                        onChange={(e) => handleConfigChange('monitoring', 'metricsEnabled', e.target.checked)}
                        className="mr-3"
                      />
                      <label htmlFor="metricsEnabled" className="text-text-primary">
                        Enable Metrics Collection
                      </label>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="tracingEnabled"
                        checked={monitoringConfig.tracingEnabled}
                        onChange={(e) => handleConfigChange('monitoring', 'tracingEnabled', e.target.checked)}
                        className="mr-3"
                      />
                      <label htmlFor="tracingEnabled" className="text-text-primary">
                        Enable Distributed Tracing
                      </label>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}