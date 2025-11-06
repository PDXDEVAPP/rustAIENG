import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Brain, 
  MessageSquare, 
  Cloud, 
  BarChart3, 
  Settings,
  Menu,
  X
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { useState } from 'react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Models', href: '/models', icon: Brain },
  { name: 'Console', href: '/console', icon: MessageSquare },
  { name: 'Storage', href: '/storage', icon: Cloud },
  { name: 'Performance', href: '/performance', icon: BarChart3 },
  { name: 'Settings', href: '/settings', icon: Settings },
]

interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex h-screen">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        >
          <div className="absolute inset-0 bg-black/50" />
        </div>
      )}

      {/* Sidebar */}
      <div className={cn(
        "fixed inset-y-0 left-0 z-50 w-60 bg-bg-surface border-r border-border-default transform transition-transform duration-300 ease-out lg:translate-x-0 lg:static lg:inset-0",
        sidebarOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex items-center justify-between h-16 px-6 border-b border-border-default">
          <div className="flex items-center">
            <Brain className="w-8 h-8 text-primary-500" />
            <span className="ml-2 text-lg font-semibold text-text-primary">
              Rust Ollama
            </span>
          </div>
          <button
            className="lg:hidden text-text-secondary hover:text-text-primary"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <nav className="mt-6">
          <ul className="space-y-1 px-3">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href || 
                              (item.href === '/dashboard' && location.pathname === '/')
              const Icon = item.icon
              
              return (
                <li key={item.name}>
                  <Link
                    to={item.href}
                    className={cn(
                      "sidebar-nav-item",
                      isActive && "active"
                    )}
                    onClick={() => setSidebarOpen(false)}
                  >
                    <Icon className="w-5 h-5 mr-3" />
                    {item.name}
                  </Link>
                </li>
              )
            })}
          </ul>
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-border-default">
          <div className="text-xs text-text-secondary">
            <div>v0.2.0</div>
            <div className="mt-1">Rust Ollama Dashboard</div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top header */}
        <header className="h-16 bg-bg-surface border-b border-border-default flex items-center justify-between px-6">
          <button
            className="lg:hidden text-text-secondary hover:text-text-primary"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="w-6 h-6" />
          </button>
          
          <div className="flex-1 lg:flex lg:justify-end">
            <div className="flex items-center space-x-4">
              <div className="text-sm text-text-secondary">
                <span className="inline-flex items-center">
                  <span className="w-2 h-2 bg-semantic-success rounded-full mr-2"></span>
                  System Online
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-7xl mx-auto p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}