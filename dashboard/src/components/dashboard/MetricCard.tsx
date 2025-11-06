import React from 'react'
import { LucideIcon, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '../../lib/utils'

interface MetricCardProps {
  title: string
  value: string | number
  icon: LucideIcon
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: number
  description?: string
}

export function MetricCard({ 
  title, 
  value, 
  icon: Icon, 
  trend = 'neutral', 
  trendValue, 
  description 
}: MetricCardProps) {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-semantic-success" />
      case 'down':
        return <TrendingDown className="w-4 h-4 text-semantic-error" />
      default:
        return <Minus className="w-4 h-4 text-text-secondary" />
    }
  }

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'text-semantic-success'
      case 'down':
        return 'text-semantic-error'
      default:
        return 'text-text-secondary'
    }
  }

  return (
    <div className="metric-card">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-text-secondary text-sm font-medium">{title}</p>
          <p className="text-2xl font-bold text-text-primary mt-2">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          
          {trendValue !== undefined && (
            <div className="flex items-center mt-2">
              {getTrendIcon()}
              <span className={cn('text-sm ml-1', getTrendColor())}>
                {trendValue}
              </span>
            </div>
          )}
          
          {description && (
            <p className="text-text-secondary text-xs mt-1">{description}</p>
          )}
        </div>
        
        <div className="p-3 bg-primary-500/10 rounded-lg">
          <Icon className="w-6 h-6 text-primary-500" />
        </div>
      </div>
    </div>
  )
}