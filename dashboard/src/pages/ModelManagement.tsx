import React, { useState, useEffect } from 'react'
import { Brain, Upload, Download, Trash2, RefreshCw, Eye } from 'lucide-react'
import { useWebSocket } from '../contexts/WebSocketContext'
import { formatBytes, formatDuration } from '../lib/utils'

interface Model {
  id: string
  name: string
  size: number
  status: 'loading' | 'loaded' | 'error' | 'downloading'
  progress?: number
  description?: string
  parameters?: string
  quantization?: string
  lastUsed?: string
  downloadUrl?: string
}

export function ModelManagement() {
  const { socket, isConnected } = useWebSocket()
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  useEffect(() => {
    if (!socket) return

    socket.on('models_list', (data: Model[]) => {
      setModels(data)
      setLoading(false)
    })

    socket.on('model_progress', (data: { modelId: string, progress: number, status: string }) => {
      setModels(prev => prev.map(model => 
        model.id === data.modelId 
          ? { ...model, progress: data.progress, status: data.status as Model['status'] }
          : model
      ))
    })

    socket.on('model_loaded', (data: { modelId: string }) => {
      setModels(prev => prev.map(model => 
        model.id === data.modelId 
          ? { ...model, status: 'loaded' as const, progress: 100 }
          : model
      ))
    })

    // Request initial model list
    socket.emit('get_models')
    setLoading(true)

    return () => {
      socket.off('models_list')
      socket.off('model_progress')
      socket.off('model_loaded')
    }
  }, [socket])

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || !socket) return

    const modelData = {
      name: file.name,
      size: file.size,
      file: file,
    }

    setUploadProgress(0)
    socket.emit('upload_model', modelData)
  }

  const handleDownload = (model: Model) => {
    if (!socket || !model.downloadUrl) return
    
    socket.emit('download_model', { 
      modelId: model.id, 
      url: model.downloadUrl 
    })
  }

  const handleDelete = (modelId: string) => {
    if (!socket) return
    socket.emit('delete_model', { modelId })
  }

  const handleRefresh = () => {
    if (!socket) return
    socket.emit('get_models')
    setLoading(true)
  }

  const getStatusColor = (status: Model['status']) => {
    switch (status) {
      case 'loaded':
        return 'text-semantic-success'
      case 'loading':
      case 'downloading':
        return 'text-semantic-warning'
      case 'error':
        return 'text-semantic-error'
    }
  }

  const getStatusIcon = (status: Model['status']) => {
    switch (status) {
      case 'loaded':
        return '●'
      case 'loading':
      case 'downloading':
        return '◐'
      case 'error':
        return '×'
    }
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Model Management</h1>
          <p className="text-text-secondary mt-2">
            Upload, manage, and monitor your language models
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button 
            onClick={handleRefresh}
            disabled={loading}
            className="btn-secondary"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          
          <label className="btn-primary cursor-pointer">
            <Upload className="w-4 h-4 mr-2" />
            Upload Model
            <input
              type="file"
              className="hidden"
              onChange={handleUpload}
              accept=".gguf,.bin,.safetensors"
            />
          </label>
        </div>
      </div>

      {/* Upload Progress */}
      {uploadProgress > 0 && (
        <div className="bg-bg-surface border border-border-default rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-text-primary">Uploading model...</span>
            <span className="text-text-secondary">{uploadProgress}%</span>
          </div>
          <div className="w-full bg-bg-page rounded-full h-2">
            <div 
              className="bg-primary-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <div key={model.id} className="bg-bg-surface border border-border-default rounded-lg p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center">
                <Brain className="w-6 h-6 text-primary-500 mr-3" />
                <div>
                  <h3 className="text-text-primary font-semibold">{model.name}</h3>
                  <p className="text-text-secondary text-sm">{formatBytes(model.size)}</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className={`text-sm ${getStatusColor(model.status)}`}>
                  {getStatusIcon(model.status)} {model.status}
                </span>
              </div>
            </div>

            {/* Progress Bar */}
            {(model.status === 'loading' || model.status === 'downloading') && model.progress !== undefined && (
              <div className="mb-4">
                <div className="w-full bg-bg-page rounded-full h-2">
                  <div 
                    className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${model.progress}%` }}
                  />
                </div>
                <p className="text-text-secondary text-xs mt-1">{model.progress}% complete</p>
              </div>
            )}

            {/* Model Details */}
            {model.description && (
              <p className="text-text-secondary text-sm mb-4">{model.description}</p>
            )}
            
            <div className="space-y-2 mb-4 text-sm">
              {model.parameters && (
                <div className="flex justify-between">
                  <span className="text-text-secondary">Parameters:</span>
                  <span className="text-text-primary">{model.parameters}</span>
                </div>
              )}
              {model.quantization && (
                <div className="flex justify-between">
                  <span className="text-text-secondary">Quantization:</span>
                  <span className="text-text-primary">{model.quantization}</span>
                </div>
              )}
              {model.lastUsed && (
                <div className="flex justify-between">
                  <span className="text-text-secondary">Last used:</span>
                  <span className="text-text-primary">{model.lastUsed}</span>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center space-x-2">
              <button className="btn-secondary flex-1 text-sm">
                <Eye className="w-4 h-4 mr-1" />
                View
              </button>
              
              {model.downloadUrl && (
                <button 
                  onClick={() => handleDownload(model)}
                  className="btn-primary flex-1 text-sm"
                >
                  <Download className="w-4 h-4 mr-1" />
                  Download
                </button>
              )}
              
              <button 
                onClick={() => handleDelete(model.id)}
                className="p-2 text-text-secondary hover:text-semantic-error hover:bg-semantic-error/10 rounded-md transition-colors"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}

        {/* Add New Model Card */}
        <div className="bg-bg-surface border-2 border-dashed border-border-default rounded-lg p-6 flex flex-col items-center justify-center min-h-[300px]">
          <Brain className="w-12 h-12 text-text-secondary mb-4" />
          <h3 className="text-text-primary font-semibold mb-2">Add New Model</h3>
          <p className="text-text-secondary text-sm text-center mb-4">
            Upload a new model file or download from model registry
          </p>
          <div className="space-y-2 w-full">
            <label className="btn-primary w-full text-center cursor-pointer">
              <Upload className="w-4 h-4 mr-2" />
              Upload File
              <input
                type="file"
                className="hidden"
                onChange={handleUpload}
                accept=".gguf,.bin,.safetensors"
              />
            </label>
            <button className="btn-secondary w-full text-center">
              <Download className="w-4 h-4 mr-2" />
              Browse Registry
            </button>
          </div>
        </div>
      </div>

      {models.length === 0 && !loading && (
        <div className="text-center py-12">
          <Brain className="w-16 h-16 text-text-secondary mx-auto mb-4" />
          <h3 className="text-text-primary font-semibold mb-2">No models installed</h3>
          <p className="text-text-secondary mb-6">
            Get started by uploading your first model or downloading from the registry
          </p>
          <label className="btn-primary cursor-pointer">
            <Upload className="w-4 h-4 mr-2" />
            Upload Your First Model
            <input
              type="file"
              className="hidden"
              onChange={handleUpload}
              accept=".gguf,.bin,.safetensors"
            />
          </label>
        </div>
      )}
    </div>
  )
}