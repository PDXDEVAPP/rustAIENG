import React, { useState, useEffect } from 'react'
import { Activity, Brain, Server, Cpu, HardDrive, Zap } from 'lucide-react'
import { MetricCard } from '../components/dashboard/MetricCard'
import { RealTimeChart } from '../components/dashboard/RealTimeChart'
import { SystemStatus } from '../components/dashboard/SystemStatus'
import { useWebSocket } from '../contexts/WebSocketContext'

interface SystemMetrics {
  activeModels: number
  totalRequests: number
  avgResponseTime: number
  cpuUsage: number
  memoryUsage: number
  activeConnections: number
}

export function Dashboard() {
  const { isConnected, socket } = useWebSocket()
  const [metrics, setMetrics] = useState<SystemMetrics>({
    activeModels: 0,
    totalRequests: 0,
    avgResponseTime: 0,
    cpuUsage: 0,
    memoryUsage: 0,
    activeConnections: 0,
  })
  
  const [chartData, setChartData] = useState<Array<{time: string, requests: number, responseTime: number}>>([])

  useEffect(() => {
    if (!socket) return

    // Listen for real-time metrics updates
    socket.on('metrics_update', (data: SystemMetrics) => {
      setMetrics(data)
      
      // Update chart data
      const now = new Date().toLocaleTimeString()
      setChartData(prev => {
        const newData = [...prev, { time: now, requests: data.totalRequests, responseTime: data.avgResponseTime }]
        return newData.slice(-20) // Keep last 20 data points
      })
    })

    // Request initial metrics
    socket.emit('get_metrics')

    return () => {
      socket.off('metrics_update')
    }
  }, [socket])

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-text-primary">Dashboard</h1>
        <p className="text-text-secondary mt-2">
          Monitor your Ollama server performance and system health in real-time
        </p>
      </div>

      {/* Connection Status */}
      <div className="bg-bg-surface border border-border-default rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-3 ${isConnected ? 'bg-semantic-success' : 'bg-semantic-error'}`} />
            <span className="text-text-primary font-medium">
              {isConnected ? 'Connected to Ollama Server' : 'Disconnected'}
            </span>
          </div>
          <div className="text-sm text-text-secondary">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard
          title="Active Models"
          value={metrics.activeModels}
          icon={Brain}
          trend={metrics.activeModels > 0 ? 'up' : 'neutral'}
          trendValue={metrics.activeModels}
        />
        
        <MetricCard
          title="Total Requests"
          value={metrics.totalRequests}
          icon={Activity}
          trend="up"
          trendValue={metrics.totalRequests}
        />
        
        <MetricCard
          title="Avg Response Time"
          value={`${metrics.avgResponseTime}ms`}
          icon={Zap}
          trend={metrics.avgResponseTime < 1000 ? 'down' : 'up'}
          trendValue={metrics.avgResponseTime}
        />
        
        <MetricCard
          title="CPU Usage"
          value={`${metrics.cpuUsage.toFixed(1)}%`}
          icon={Cpu}
          trend={metrics.cpuUsage < 80 ? 'down' : 'up'}
          trendValue={metrics.cpuUsage}
        />
        
        <MetricCard
          title="Memory Usage"
          value={`${metrics.memoryUsage.toFixed(1)}%`}
          icon={HardDrive}
          trend={metrics.memoryUsage < 80 ? 'down' : 'up'}
          trendValue={metrics.memoryUsage}
        />
        
        <MetricCard
          title="Active Connections"
          value={metrics.activeConnections}
          icon={Server}
          trend="neutral"
          trendValue={metrics.activeConnections}
        />
      </div>

      {/* Charts and Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Chart */}
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Request Performance
          </h3>
          <RealTimeChart data={chartData} />
        </div>

        {/* System Status */}
        <SystemStatus />
      </div>

      {/* Quick Actions */}
      <div className="bg-bg-surface border border-border-default rounded-lg p-6">
        <h3 className="text-lg font-semibold text-text-primary mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="btn-primary">
            <Brain className="w-4 h-4 mr-2" />
            Load New Model
          </button>
          <button className="btn-secondary">
            <Activity className="w-4 h-4 mr-2" />
            Run Health Check
          </button>
          <button className="btn-secondary">
            <Server className="w-4 h-4 mr-2" />
            View Logs
          </button>
        </div>
      </div>
    </div>
  )
}