import React, { useState, useEffect } from 'react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Activity, Clock, Zap, Database, Server } from 'lucide-react'
import { useWebSocket } from '../contexts/WebSocketContext'
import { formatDuration } from '../lib/utils'

interface PerformanceData {
  requestsOverTime: Array<{ time: string, requests: number, errors: number }>
  responseTimeDistribution: Array<{ range: string, count: number }>
  modelUsage: Array<{ model: string, requests: number, avgTime: number }>
  systemMetrics: {
    cpu: number
    memory: number
    disk: number
    network: number
  }
  activeConnections: number
  totalRequests: number
  errorRate: number
}

export function Performance() {
  const { socket, isConnected } = useWebSocket()
  const [performanceData, setPerformanceData] = useState<PerformanceData>({
    requestsOverTime: [],
    responseTimeDistribution: [],
    modelUsage: [],
    systemMetrics: { cpu: 0, memory: 0, disk: 0, network: 0 },
    activeConnections: 0,
    totalRequests: 0,
    errorRate: 0,
  })
  
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('1h')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!socket) return

    socket.on('performance_data', (data: PerformanceData) => {
      setPerformanceData(data)
      setLoading(false)
    })

    socket.on('real_time_metrics', (data: any) => {
      // Update real-time data
      setPerformanceData(prev => ({
        ...prev,
        systemMetrics: data.systemMetrics || prev.systemMetrics,
        activeConnections: data.activeConnections || prev.activeConnections,
        totalRequests: data.totalRequests || prev.totalRequests,
        errorRate: data.errorRate || prev.errorRate,
      }))
    })

    // Request initial data
    socket.emit('get_performance_data', { timeRange })
    setLoading(true)

    return () => {
      socket.off('performance_data')
      socket.off('real_time_metrics')
    }
  }, [socket, timeRange])

  const handleTimeRangeChange = (range: '1h' | '24h' | '7d') => {
    setTimeRange(range)
    if (socket) {
      socket.emit('get_performance_data', { timeRange: range })
      setLoading(true)
    }
  }

  const pieColors = ['#A855F7', '#F5A524', '#238636', '#DA3633', '#8B5CF6']

  const systemMetricsData = [
    { name: 'CPU', value: performanceData.systemMetrics.cpu, color: '#A855F7' },
    { name: 'Memory', value: performanceData.systemMetrics.memory, color: '#F5A524' },
    { name: 'Disk', value: performanceData.systemMetrics.disk, color: '#238636' },
    { name: 'Network', value: performanceData.systemMetrics.network, color: '#DA3633' },
  ]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Performance Analytics</h1>
          <p className="text-text-secondary mt-2">
            Monitor system performance, request patterns, and resource utilization
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-text-secondary text-sm">Time Range:</span>
            <select
              value={timeRange}
              onChange={(e) => handleTimeRangeChange(e.target.value as any)}
              className="input-field"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm font-medium">Total Requests</p>
              <p className="text-2xl font-bold text-text-primary mt-2">
                {performanceData.totalRequests.toLocaleString()}
              </p>
            </div>
            <Activity className="w-8 h-8 text-primary-500" />
          </div>
        </div>

        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm font-medium">Active Connections</p>
              <p className="text-2xl font-bold text-text-primary mt-2">
                {performanceData.activeConnections}
              </p>
            </div>
            <Server className="w-8 h-8 text-semantic-success" />
          </div>
        </div>

        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm font-medium">Error Rate</p>
              <p className="text-2xl font-bold text-text-primary mt-2">
                {performanceData.errorRate.toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-semantic-error" />
          </div>
        </div>

        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm font-medium">Avg Response Time</p>
              <p className="text-2xl font-bold text-text-primary mt-2">
                {performanceData.responseTimeDistribution.length > 0 
                  ? formatDuration(
                      performanceData.responseTimeDistribution.reduce((acc, curr) => 
                        acc + (parseInt(curr.range.split('-')[0]) * curr.count), 0
                      ) / Math.max(performanceData.responseTimeDistribution.reduce((acc, curr) => acc + curr.count, 0), 1)
                    )
                  : '0ms'
                }
              </p>
            </div>
            <Clock className="w-8 h-8 text-semantic-warning" />
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Requests Over Time */}
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Requests Over Time
          </h3>
          
          {loading ? (
            <div className="h-64 flex items-center justify-center">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData.requestsOverTime}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
                <XAxis 
                  dataKey="time" 
                  stroke="#848D97"
                  fontSize={12}
                  tick={{ fill: '#848D97' }}
                />
                <YAxis 
                  stroke="#848D97"
                  fontSize={12}
                  tick={{ fill: '#848D97' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#161B22', 
                    border: '1px solid #30363D',
                    borderRadius: '8px'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="requests"
                  stroke="#A855F7"
                  strokeWidth={2}
                  name="Requests"
                />
                <Line
                  type="monotone"
                  dataKey="errors"
                  stroke="#DA3633"
                  strokeWidth={2}
                  name="Errors"
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* System Resource Usage */}
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            System Resource Usage
          </h3>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={systemMetricsData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {systemMetricsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#161B22', 
                    border: '1px solid #30363D',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Usage']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Legend */}
          <div className="grid grid-cols-2 gap-2 mt-4">
            {systemMetricsData.map((metric, index) => (
              <div key={metric.name} className="flex items-center">
                <div 
                  className="w-3 h-3 rounded-full mr-2"
                  style={{ backgroundColor: metric.color }}
                />
                <span className="text-text-secondary text-sm">
                  {metric.name}: {metric.value.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Response Time Distribution */}
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Response Time Distribution
          </h3>
          
          {loading ? (
            <div className="h-64 flex items-center justify-center">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={performanceData.responseTimeDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
                <XAxis 
                  dataKey="range" 
                  stroke="#848D97"
                  fontSize={12}
                  tick={{ fill: '#848D97' }}
                />
                <YAxis 
                  stroke="#848D97"
                  fontSize={12}
                  tick={{ fill: '#848D97' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#161B22', 
                    border: '1px solid #30363D',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="count" fill="#F5A524" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Model Usage Statistics */}
        <div className="bg-bg-surface border border-border-default rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Model Usage Statistics
          </h3>
          
          {loading ? (
            <div className="h-64 flex items-center justify-center">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : performanceData.modelUsage.length > 0 ? (
            <div className="space-y-4">
              {performanceData.modelUsage.map((model, index) => (
                <div key={model.model} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-text-primary font-medium">{model.model}</span>
                    <span className="text-text-secondary text-sm">
                      {model.requests} requests
                    </span>
                  </div>
                  <div className="w-full bg-bg-page rounded-full h-2">
                    <div 
                      className="h-2 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${(model.requests / Math.max(...performanceData.modelUsage.map(m => m.requests))) * 100}%`,
                        backgroundColor: pieColors[index % pieColors.length]
                      }}
                    />
                  </div>
                  <div className="text-text-secondary text-xs">
                    Avg: {formatDuration(model.avgTime)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center">
              <p className="text-text-secondary">No model usage data available</p>
            </div>
          )}
        </div>
      </div>

      {/* Performance Summary */}
      <div className="bg-bg-surface border border-border-default rounded-lg p-6">
        <h3 className="text-lg font-semibold text-text-primary mb-4">
          Performance Summary
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <Zap className="w-8 h-8 text-primary-500 mx-auto mb-2" />
            <h4 className="text-text-primary font-medium">Throughput</h4>
            <p className="text-text-secondary text-sm">
              {performanceData.requestsOverTime.length > 0 
                ? Math.round(performanceData.requestsOverTime.reduce((acc, curr) => acc + curr.requests, 0) / performanceData.requestsOverTime.length)
                : 0
              } req/min
            </p>
          </div>
          
          <div className="text-center">
            <Database className="w-8 h-8 text-semantic-warning mx-auto mb-2" />
            <h4 className="text-text-primary font-medium">Data Processed</h4>
            <p className="text-text-secondary text-sm">
              {performanceData.totalRequests * 1024} tokens
            </p>
          </div>
          
          <div className="text-center">
            <Server className="w-8 h-8 text-semantic-success mx-auto mb-2" />
            <h4 className="text-text-primary font-medium">System Health</h4>
            <p className="text-text-secondary text-sm">
              {performanceData.errorRate < 1 ? 'Excellent' : 
               performanceData.errorRate < 5 ? 'Good' : 'Needs Attention'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}