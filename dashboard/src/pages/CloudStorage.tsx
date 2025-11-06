import React, { useState, useEffect } from 'react'
import { Cloud, Upload, Download, Trash2, FolderOpen, Settings, Eye, EyeOff } from 'lucide-react'
import { useWebSocket } from '../contexts/WebSocketContext'
import { formatBytes } from '../lib/utils'

interface StorageProvider {
  type: 's3' | 'azure'
  name: string
  status: 'connected' | 'disconnected' | 'error'
  bucket?: string
  container?: string
  region?: string
  lastSync?: string
}

interface StorageFile {
  id: string
  name: string
  size: number
  type: 'folder' | 'file'
  lastModified: Date
  path: string
  url?: string
}

interface CloudConfig {
  aws: {
    accessKeyId: string
    secretAccessKey: string
    region: string
    bucket: string
  }
  azure: {
    accountName: string
    accountKey: string
    container: string
  }
}

export function CloudStorage() {
  const { socket, isConnected } = useWebSocket()
  const [providers, setProviders] = useState<StorageProvider[]>([])
  const [files, setFiles] = useState<StorageFile[]>([])
  const [currentPath, setCurrentPath] = useState('')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [config, setConfig] = useState<CloudConfig>({
    aws: { accessKeyId: '', secretAccessKey: '', region: 'us-east-1', bucket: '' },
    azure: { accountName: '', accountKey: '', container: '' }
  })
  const [showSecrets, setShowSecrets] = useState(false)

  useEffect(() => {
    if (!socket) return

    socket.on('storage_providers', (data: StorageProvider[]) => {
      setProviders(data)
      if (data.length > 0 && !selectedProvider) {
        setSelectedProvider(data[0].type)
      }
    })

    socket.on('storage_files', (data: { files: StorageFile[], path: string }) => {
      setFiles(data.files)
      setCurrentPath(data.path)
      setLoading(false)
    })

    socket.on('storage_status', (data: { provider: string, status: StorageProvider['status'] }) => {
      setProviders(prev => prev.map(p => 
        p.type === data.provider ? { ...p, status: data.status } : p
      ))
    })

    // Request initial data
    socket.emit('get_storage_providers')

    return () => {
      socket.off('storage_providers')
      socket.off('storage_files')
      socket.off('storage_status')
    }
  }, [socket, selectedProvider])

  const handleProviderChange = (providerType: string) => {
    setSelectedProvider(providerType)
    if (socket) {
      socket.emit('browse_storage', { 
        provider: providerType, 
        path: currentPath || '' 
      })
      setLoading(true)
    }
  }

  const handleNavigate = (path: string) => {
    if (!selectedProvider || !socket) return
    
    socket.emit('browse_storage', { 
      provider: selectedProvider, 
      path 
    })
    setLoading(true)
  }

  const handleConfigSave = () => {
    if (!socket) return
    
    socket.emit('configure_storage', {
      provider: selectedProvider,
      config: config
    })
    
    setShowConfig(false)
    // Refresh provider status
    socket.emit('get_storage_providers')
  }

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || !selectedProvider || !socket) return

    const uploadData = {
      provider: selectedProvider,
      path: currentPath,
      file: file,
      fileName: file.name
    }

    socket.emit('upload_file', uploadData)
  }

  const handleDelete = (fileId: string) => {
    if (!selectedProvider || !socket) return
    
    socket.emit('delete_file', {
      provider: selectedProvider,
      fileId
    })
  }

  const getProviderIcon = (type: string) => {
    switch (type) {
      case 's3':
        return '🟠'
      case 'azure':
        return '🔵'
      default:
        return '☁️'
    }
  }

  const getStatusColor = (status: StorageProvider['status']) => {
    switch (status) {
      case 'connected':
        return 'text-semantic-success'
      case 'disconnected':
        return 'text-text-secondary'
      case 'error':
        return 'text-semantic-error'
    }
  }

  const getStatusIcon = (status: StorageProvider['status']) => {
    switch (status) {
      case 'connected':
        return '●'
      case 'disconnected':
        return '○'
      case 'error':
        return '×'
    }
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Cloud Storage</h1>
          <p className="text-text-secondary mt-2">
            Manage files across AWS S3 and Azure Blob Storage
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <button 
            onClick={() => setShowConfig(!showConfig)}
            className="btn-secondary"
          >
            <Settings className="w-4 h-4 mr-2" />
            Configure
          </button>
          
          <label className="btn-primary cursor-pointer">
            <Upload className="w-4 h-4 mr-2" />
            Upload File
            <input
              type="file"
              className="hidden"
              onChange={handleUpload}
            />
          </label>
        </div>
      </div>

      {/* Storage Providers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {providers.map((provider) => (
          <div 
            key={provider.type}
            className={`bg-bg-surface border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
              selectedProvider === provider.type 
                ? 'border-primary-500 bg-primary-500/5' 
                : 'border-border-default hover:border-border-interactive'
            }`}
            onClick={() => handleProviderChange(provider.type)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <span className="text-2xl mr-3">{getProviderIcon(provider.type)}</span>
                <div>
                  <h3 className="text-text-primary font-semibold">{provider.name}</h3>
                  <p className="text-text-secondary text-sm">
                    {provider.type === 's3' 
                      ? `${provider.region} - ${provider.bucket}`
                      : `${provider.container}`
                    }
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className={`text-sm ${getStatusColor(provider.status)}`}>
                  {getStatusIcon(provider.status)} {provider.status}
                </span>
              </div>
            </div>

            {provider.lastSync && (
              <p className="text-text-secondary text-sm">
                Last synced: {provider.lastSync}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Configuration Panel */}
      {showConfig && (
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Storage Configuration
          </h3>

          <div className="space-y-6">
            {/* AWS S3 Configuration */}
            <div>
              <h4 className="text-text-primary font-medium mb-3">AWS S3 Configuration</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-text-secondary text-sm mb-1">Access Key ID</label>
                  <div className="relative">
                    <input
                      type={showSecrets ? 'text' : 'password'}
                      value={config.aws.accessKeyId}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        aws: { ...prev.aws, accessKeyId: e.target.value }
                      }))}
                      className="input-field w-full pr-10"
                      placeholder="AKIA..."
                    />
                    <button
                      type="button"
                      onClick={() => setShowSecrets(!showSecrets)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-text-secondary hover:text-text-primary"
                    >
                      {showSecrets ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-text-secondary text-sm mb-1">Secret Access Key</label>
                  <input
                    type={showSecrets ? 'text' : 'password'}
                    value={config.aws.secretAccessKey}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      aws: { ...prev.aws, secretAccessKey: e.target.value }
                    }))}
                    className="input-field w-full"
                    placeholder="Secret key"
                  />
                </div>

                <div>
                  <label className="block text-text-secondary text-sm mb-1">Region</label>
                  <select
                    value={config.aws.region}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      aws: { ...prev.aws, region: e.target.value }
                    }))}
                    className="input-field w-full"
                  >
                    <option value="us-east-1">US East (N. Virginia)</option>
                    <option value="us-west-2">US West (Oregon)</option>
                    <option value="eu-west-1">Europe (Ireland)</option>
                    <option value="ap-southeast-1">Asia Pacific (Singapore)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-text-secondary text-sm mb-1">Bucket Name</label>
                  <input
                    type="text"
                    value={config.aws.bucket}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      aws: { ...prev.aws, bucket: e.target.value }
                    }))}
                    className="input-field w-full"
                    placeholder="my-bucket"
                  />
                </div>
              </div>
            </div>

            {/* Azure Configuration */}
            <div>
              <h4 className="text-text-primary font-medium mb-3">Azure Blob Storage Configuration</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-text-secondary text-sm mb-1">Account Name</label>
                  <input
                    type="text"
                    value={config.azure.accountName}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      azure: { ...prev.azure, accountName: e.target.value }
                    }))}
                    className="input-field w-full"
                    placeholder="mystorageaccount"
                  />
                </div>

                <div>
                  <label className="block text-text-secondary text-sm mb-1">Account Key</label>
                  <input
                    type={showSecrets ? 'text' : 'password'}
                    value={config.azure.accountKey}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      azure: { ...prev.azure, accountKey: e.target.value }
                    }))}
                    className="input-field w-full"
                    placeholder="Account key"
                  />
                </div>

                <div>
                  <label className="block text-text-secondary text-sm mb-1">Container Name</label>
                  <input
                    type="text"
                    value={config.azure.container}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      azure: { ...prev.azure, container: e.target.value }
                    }))}
                    className="input-field w-full"
                    placeholder="my-container"
                  />
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-4">
              <button 
                onClick={() => setShowConfig(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button 
                onClick={handleConfigSave}
                className="btn-primary"
              >
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      )}

      {/* File Browser */}
      {selectedProvider && (
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <h3 className="text-lg font-semibold text-text-primary">
                {providers.find(p => p.type === selectedProvider)?.name} Files
              </h3>
              <div className="text-text-secondary">
                {currentPath && (
                  <span className="text-sm">{currentPath}</span>
                )}
              </div>
            </div>

            <button 
              onClick={() => handleNavigate('')}
              disabled={!currentPath}
              className="btn-secondary disabled:opacity-50"
            >
              <FolderOpen className="w-4 h-4 mr-2" />
              Root
            </button>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-text-secondary">Loading files...</p>
            </div>
          ) : (
            <div className="space-y-2">
              {files.map((file) => (
                <div 
                  key={file.id}
                  className="flex items-center justify-between p-3 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <div className="flex items-center">
                    {file.type === 'folder' ? (
                      <FolderOpen className="w-5 h-5 text-primary-500 mr-3" />
                    ) : (
                      <Cloud className="w-5 h-5 text-text-secondary mr-3" />
                    )}
                    <div>
                      <p className="text-text-primary font-medium">{file.name}</p>
                      <p className="text-text-secondary text-sm">
                        {file.type === 'file' ? formatBytes(file.size) : 'Folder'}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4">
                    <span className="text-text-secondary text-sm">
                      {file.lastModified.toLocaleDateString()}
                    </span>
                    
                    <div className="flex items-center space-x-2">
                      {file.type === 'folder' ? (
                        <button
                          onClick={() => handleNavigate(file.path)}
                          className="p-2 hover:bg-white/10 rounded text-text-secondary hover:text-text-primary transition-colors"
                        >
                          <FolderOpen className="w-4 h-4" />
                        </button>
                      ) : (
                        <>
                          {file.url && (
                            <button className="p-2 hover:bg-white/10 rounded text-text-secondary hover:text-text-primary transition-colors">
                              <Download className="w-4 h-4" />
                            </button>
                          )}
                        </>
                      )}
                      
                      <button 
                        onClick={() => handleDelete(file.id)}
                        className="p-2 hover:bg-semantic-error/10 rounded text-text-secondary hover:text-semantic-error transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}

              {files.length === 0 && (
                <div className="text-center py-12">
                  <Cloud className="w-16 h-16 text-text-secondary mx-auto mb-4" />
                  <h3 className="text-text-primary font-semibold mb-2">No files found</h3>
                  <p className="text-text-secondary">
                    This directory is empty or you don't have access to view files
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}