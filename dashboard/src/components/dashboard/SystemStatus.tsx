import React, { useState, useEffect } from 'react'
import { CheckCircle, XCircle, AlertCircle, Database, Cloud, Cpu } from 'lucide-react'
import { useWebSocket } from '../../contexts/WebSocketContext'

interface StatusItem {
  name: string
  status: 'healthy' | 'warning' | 'error'
  message: string
  lastCheck: string
}

export function SystemStatus() {
  const { socket, isConnected } = useWebSocket()
  const [statuses, setStatuses] = useState<StatusItem[]>([
    {
      name: 'API Server',
      status: 'healthy',
      message: 'All endpoints responding normally',
      lastCheck: 'Just now',
    },
    {
      name: 'Database',
      status: 'healthy',
      message: 'SQLite connection stable',
      lastCheck: 'Just now',
    },
    {
      name: 'Model Cache',
      status: 'healthy',
      message: 'All models loaded successfully',
      lastCheck: 'Just now',
    },
    {
      name: 'Cloud Storage',
      status: 'warning',
      message: 'AWS S3 connection timeout',
      lastCheck: '2 min ago',
    },
    {
      name: 'WebSocket',
      status: isConnected ? 'healthy' : 'error',
      message: isConnected ? 'Real-time connections active' : 'Connection lost',
      lastCheck: 'Just now',
    },
  ])

  useEffect(() => {
    if (!socket) return

    // Listen for status updates
    socket.on('status_update', (data: Partial<StatusItem>) => {
      setStatuses(prev => prev.map(status => 
        status.name === data.name ? { ...status, ...data } : status
      ))
    })

    // Request initial status
    socket.emit('get_system_status')

    return () => {
      socket.off('status_update')
    }
  }, [socket])

  useEffect(() => {
    // Update WebSocket status
    setStatuses(prev => prev.map(status => 
      status.name === 'WebSocket' 
        ? { 
            ...status, 
            status: isConnected ? 'healthy' as const : 'error' as const,
            message: isConnected ? 'Real-time connections active' : 'Connection lost',
            lastCheck: 'Just now'
          }
        : status
    ))
  }, [isConnected])

  const getStatusIcon = (status: StatusItem['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-semantic-success" />
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-semantic-warning" />
      case 'error':
        return <XCircle className="w-5 h-5 text-semantic-error" />
    }
  }

  const getStatusColor = (status: StatusItem['status']) => {
    switch (status) {
      case 'healthy':
        return 'text-semantic-success'
      case 'warning':
        return 'text-semantic-warning'
      case 'error':
        return 'text-semantic-error'
    }
  }

  const getComponentIcon = (name: string) => {
    if (name.includes('Database')) return Database
    if (name.includes('Cloud')) return Cloud
    if (name.includes('CPU') || name.includes('System')) return Cpu
    return CheckCircle
  }

  return (
    <div className="bg-bg-surface border border-border-default rounded-lg p-6">
      <h3 className="text-lg font-semibold text-text-primary mb-4">
        System Status
      </h3>
      
      <div className="space-y-4">
        {statuses.map((status) => {
          const ComponentIcon = getComponentIcon(status.name)
          return (
            <div key={status.name} className="flex items-center justify-between p-3 rounded-lg bg-bg-page/50">
              <div className="flex items-center">
                <div className="p-2 bg-bg-surface rounded-lg mr-3">
                  <ComponentIcon className="w-4 h-4 text-text-secondary" />
                </div>
                <div>
                  <div className="flex items-center">
                    <span className="text-text-primary font-medium mr-2">
                      {status.name}
                    </span>
                    {getStatusIcon(status.status)}
                  </div>
                  <p className="text-text-secondary text-sm">
                    {status.message}
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <p className={`text-sm font-medium ${getStatusColor(status.status)}`}>
                  {status.status.toUpperCase()}
                </p>
                <p className="text-text-secondary text-xs">
                  {status.lastCheck}
                </p>
              </div>
            </div>
          )
        })}
      </div>
      
      <div className="mt-6 pt-4 border-t border-border-default">
        <button className="w-full btn-secondary">
          Run Full Health Check
        </button>
      </div>
    </div>
  )
}