import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { formatDuration } from '../../lib/utils'

interface ChartData {
  time: string
  requests: number
  responseTime: number
}

interface RealTimeChartProps {
  data: ChartData[]
}

export function RealTimeChart({ data }: RealTimeChartProps) {
  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-text-secondary">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p>Waiting for data...</p>
        </div>
      </div>
    )
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-bg-surface border border-border-default rounded-lg p-3 shadow-lg">
          <p className="text-text-primary font-medium">{`Time: ${label}`}</p>
          <p className="text-text-secondary">
            <span className="text-primary-500">●</span> Requests: {payload[0]?.value || 0}
          </p>
          <p className="text-text-secondary">
            <span className="text-semantic-warning">●</span> Response Time: {formatDuration(payload[1]?.value || 0)}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
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
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="requests"
            stroke="#A855F7"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, stroke: '#A855F7', strokeWidth: 2, fill: '#0D1117' }}
          />
          <Line
            type="monotone"
            dataKey="responseTime"
            stroke="#F5A524"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, stroke: '#F5A524', strokeWidth: 2, fill: '#0D1117' }}
          />
        </LineChart>
      </ResponsiveContainer>
      
      {/* Legend */}
      <div className="flex justify-center space-x-6 mt-4">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-primary-500 rounded-full mr-2" />
          <span className="text-text-secondary text-sm">Requests</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-semantic-warning rounded-full mr-2" />
          <span className="text-text-secondary text-sm">Response Time</span>
        </div>
      </div>
    </div>
  )
}