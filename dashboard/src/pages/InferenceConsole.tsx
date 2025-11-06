import React, { useState, useEffect, useRef } from 'react'
import { Send, Brain, Copy, ThumbsUp, ThumbsDown, RotateCcw } from 'lucide-react'
import { useWebSocket } from '../contexts/WebSocketContext'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  model?: string
  tokens?: number
  responseTime?: number
}

interface Model {
  id: string
  name: string
  status: 'loaded' | 'loading' | 'error'
}

export function InferenceConsole() {
  const { socket, isConnected } = useWebSocket()
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [availableModels, setAvailableModels] = useState<Model[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!socket) return

    socket.on('models_list', (models: Model[]) => {
      setAvailableModels(models.filter(m => m.status === 'loaded'))
      if (models.length > 0 && !selectedModel) {
        setSelectedModel(models[0].id)
      }
    })

    socket.on('chat_response', (data: {
      message: string
      model: string
      tokens: number
      responseTime: number
    }) => {
      const assistantMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: data.message,
        timestamp: new Date(),
        model: data.model,
        tokens: data.tokens,
        responseTime: data.responseTime,
      }
      
      setMessages(prev => [...prev, assistantMessage])
      setIsGenerating(false)
    })

    socket.on('stream_token', (data: { token: string }) => {
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1]
        if (lastMessage && lastMessage.type === 'assistant') {
          return prev.map((msg, index) => 
            index === prev.length - 1 
              ? { ...msg, content: msg.content + data.token }
              : msg
          )
        } else {
          const assistantMessage: Message = {
            id: Date.now().toString(),
            type: 'assistant',
            content: data.token,
            timestamp: new Date(),
            model: selectedModel,
          }
          return [...prev, assistantMessage]
        }
      })
    })

    // Request initial models list
    socket.emit('get_models')

    return () => {
      socket.off('models_list')
      socket.off('chat_response')
      socket.off('stream_token')
    }
  }, [socket, selectedModel])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || !socket || !selectedModel || isGenerating) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsGenerating(true)

    // Send message to backend
    socket.emit('chat_request', {
      message: inputValue.trim(),
      model: selectedModel,
      stream: true,
    })
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  const handleClearChat = () => {
    setMessages([])
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
    }
  }

  useEffect(() => {
    adjustTextareaHeight()
  }, [inputValue])

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Inference Console</h1>
          <p className="text-text-secondary mt-2">
            Test and interact with your language models in real-time
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="input-field min-w-[200px]"
          >
            <option value="">Select Model</option>
            {availableModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>

          <button 
            onClick={handleClearChat}
            className="btn-secondary"
            disabled={messages.length === 0}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Clear Chat
          </button>
        </div>
      </div>

      {/* Connection Status */}
      {!isConnected && (
        <div className="bg-semantic-error/10 border border-semantic-error rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-semantic-error rounded-full mr-3" />
            <span className="text-semantic-error font-medium">
              Disconnected from Ollama server
            </span>
          </div>
        </div>
      )}

      {/* Model Selection Warning */}
      {availableModels.length === 0 && isConnected && (
        <div className="bg-semantic-warning/10 border border-semantic-warning rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <Brain className="w-5 h-5 text-semantic-warning mr-3" />
            <span className="text-semantic-warning font-medium">
              No models loaded. Please load a model in the Model Management section first.
            </span>
          </div>
        </div>
      )}

      {/* Chat Interface */}
      <div className="flex-1 flex flex-col bg-bg-surface border border-border-default rounded-lg">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Brain className="w-16 h-16 text-text-secondary mx-auto mb-4" />
                <h3 className="text-text-primary font-semibold mb-2">Start a conversation</h3>
                <p className="text-text-secondary">
                  Select a model and send a message to begin testing your language model
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[70%] ${
                    message.type === 'user' ? 'chat-message-user' : 'chat-message-ai'
                  }`}
                >
                  <div className="whitespace-pre-wrap break-words">
                    {message.content}
                    {isGenerating && message.type === 'assistant' && (
                      <span className="inline-block w-2 h-5 bg-current animate-pulse ml-1" />
                    )}
                  </div>
                  
                  {message.type === 'assistant' && (
                    <div className="flex items-center justify-between mt-3 pt-2 border-t border-border-interactive/20">
                      <div className="flex items-center space-x-4 text-xs text-text-secondary">
                        {message.model && (
                          <span>{message.model}</span>
                        )}
                        {message.tokens && (
                          <span>{message.tokens} tokens</span>
                        )}
                        {message.responseTime && (
                          <span>{(message.responseTime / 1000).toFixed(2)}s</span>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <button
                          onClick={() => copyToClipboard(message.content)}
                          className="p-1 hover:bg-white/10 rounded text-text-secondary hover:text-text-primary transition-colors"
                        >
                          <Copy className="w-3 h-3" />
                        </button>
                        <button className="p-1 hover:bg-white/10 rounded text-text-secondary hover:text-semantic-success transition-colors">
                          <ThumbsUp className="w-3 h-3" />
                        </button>
                        <button className="p-1 hover:bg-white/10 rounded text-text-secondary hover:text-semantic-error transition-colors">
                          <ThumbsDown className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-border-default p-4">
          <form onSubmit={handleSubmit} className="flex items-end space-x-4">
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  selectedModel 
                    ? `Chat with ${availableModels.find(m => m.id === selectedModel)?.name}...` 
                    : 'Select a model to start chatting...'
                }
                className="w-full resize-none rounded-lg border border-border-default bg-bg-page p-3 text-text-primary placeholder-text-secondary focus:border-border-interactive focus:ring-1 focus:ring-border-interactive focus:outline-none"
                style={{ minHeight: '44px', maxHeight: '120px' }}
                disabled={!isConnected || !selectedModel || isGenerating}
              />
            </div>
            
            <button
              type="submit"
              disabled={!inputValue.trim() || !isConnected || !selectedModel || isGenerating}
              className="btn-primary p-3"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          
          <div className="flex items-center justify-between mt-2 text-xs text-text-secondary">
            <span>
              Press Enter to send, Shift+Enter for new line
            </span>
            {isGenerating && (
              <span className="flex items-center">
                <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse mr-2" />
                Generating response...
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}