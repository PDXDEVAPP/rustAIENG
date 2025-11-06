import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { Dashboard } from './pages/Dashboard'
import { ModelManagement } from './pages/ModelManagement'
import { InferenceConsole } from './pages/InferenceConsole'
import { CloudStorage } from './pages/CloudStorage'
import { Performance } from './pages/Performance'
import { Settings } from './pages/Settings'
import { WebSocketProvider } from './contexts/WebSocketContext'
import './index.css'

function App() {
  return (
    <WebSocketProvider>
      <Router>
        <div className="min-h-screen bg-bg-page">
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/console" element={<InferenceConsole />} />
              <Route path="/storage" element={<CloudStorage />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
        </div>
      </Router>
    </WebSocketProvider>
  )
}

export default App