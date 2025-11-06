# Rust Ollama Dashboard

A modern, professional web dashboard for managing your Rust-based Ollama LLM inference server. Built with React, TypeScript, and Tailwind CSS with real-time WebSocket integration.

## 🚀 Features

### **Modern UI/UX**
- **Dark Theme Design** - Optimized for developers and AI/ML professionals
- **Real-time Updates** - WebSocket integration for live monitoring
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Professional Aesthetics** - Clean, modern interface with smooth animations

### **Core Functionality**
- **📊 Dashboard** - Real-time system metrics and performance monitoring
- **🧠 Model Management** - Upload, configure, and manage LLM models
- **💬 Inference Console** - Interactive chat interface for testing models
- **☁️ Cloud Storage** - AWS S3 and Azure Blob Storage integration
- **📈 Performance Analytics** - Detailed performance metrics and charts
- **⚙️ Settings** - Comprehensive configuration management

### **Technical Features**
- **WebSocket Real-time Communication** - Live updates and chat
- **Advanced Charts** - Interactive data visualization with Recharts
- **Professional Component Library** - Consistent design system
- **TypeScript** - Full type safety and better developer experience
- **Tailwind CSS** - Utility-first styling with custom design system

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   React Dashboard│◄──────────────►│  Rust Ollama    │
│   (TypeScript)  │     HTTP/WS     │  Backend API    │
│                 │                 │                 │
│ • Real-time UI  │                 │ • Model Mgmt    │
│ • Charts        │                 │ • Inference     │
│ • File Browser  │                 │ • Cloud Storage │
│ • Settings      │                 │ • Monitoring    │
└─────────────────┘                 └─────────────────┘
```

## 📦 Installation & Setup

### **Prerequisites**
- Node.js 18+ and npm/pnpm
- Rust 1.91+ with Cargo
- Git

### **1. Clone and Setup Rust Backend**
```bash
# Clone the repository
git clone <your-repo>
cd <your-repo>

# Install Rust dependencies and build
cargo check
cargo build --release

# Start the Rust Ollama server
cargo run --release
```

The Rust server will start on `http://localhost:11435`

### **2. Setup React Dashboard**
```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install
# or
pnpm install

# Start the development server
npm run dev
# or
pnpm dev
```

The dashboard will be available at `http://localhost:3000`

## 🎯 Usage Guide

### **Dashboard Overview**
- **System Metrics** - CPU, memory, network usage in real-time
- **Model Status** - Currently loaded models and their performance
- **Request Analytics** - Live request rates and response times
- **Connection Status** - Real-time WebSocket connection health

### **Model Management**
- **Upload Models** - Drag & drop or browse to upload `.gguf`, `.bin`, `.safetensors` files
- **Model Configuration** - Set parameters, quantization, and performance options
- **Progress Tracking** - Real-time upload and loading progress indicators
- **Model Library** - Browse and download from model registries

### **Interactive Console**
- **Real-time Chat** - Test models with live conversation interface
- **Multiple Models** - Switch between different loaded models
- **Token Tracking** - Monitor token usage and response times
- **Export Conversations** - Save chat history for analysis

### **Cloud Storage Integration**
- **AWS S3** - Configure buckets, upload files, manage storage
- **Azure Blob** - Container management and file operations
- **Real-time Sync** - Live file browser with upload/download progress
- **Configuration** - Secure credential management

### **Performance Monitoring**
- **Request Analytics** - Historical and real-time request patterns
- **Resource Monitoring** - CPU, memory, disk, and network utilization
- **Response Time Analysis** - Distribution charts and performance trends
- **Model Usage Stats** - Per-model performance and utilization metrics

### **Configuration Management**
- **Server Settings** - Host, port, timeout, and CORS configuration
- **Model Settings** - Cache size, quantization, GPU acceleration
- **Security** - API keys, rate limiting, IP restrictions
- **Monitoring** - Metrics collection and distributed tracing

## 🎨 Design System

### **Color Palette**
- **Primary**: `#A855F7` (Vibrant Purple) - CTAs, active states
- **Background**: `#0D1117` (Deep Navy) - Main background
- **Surface**: `#161B22` (Navy Gray) - Cards and components
- **Success**: `#238636` - Healthy states and confirmations
- **Warning**: `#F5A524` - Caution states
- **Error**: `#DA3633` - Error states and warnings

### **Typography**
- **Font**: Inter (Google Fonts) - Professional, highly legible
- **Hierarchy**: Display (48px) → Heading (32px) → Sub-heading (24px) → Body (16px)
- **Code**: JetBrains Mono - Monospace for logs and technical content

### **Components**
- **Cards**: Elevated design with subtle shadows and hover effects
- **Buttons**: Primary (filled) and Secondary (outlined) variants
- **Inputs**: Consistent styling with focus states and validation
- **Navigation**: Sidebar with active states and smooth transitions

## 🔧 Development

### **Project Structure**
```
dashboard/
├── src/
│   ├── components/
│   │   ├── dashboard/       # Dashboard-specific components
│   │   └── layout/          # Layout components
│   ├── pages/               # Main page components
│   ├── contexts/            # React contexts (WebSocket, etc.)
│   ├── lib/                 # Utility functions
│   └── index.css            # Global styles and design system
├── public/                  # Static assets
└── package.json             # Dependencies and scripts
```

### **Available Scripts**
```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint

# Rust Backend
cargo check          # Check for compilation errors
cargo build          # Build debug version
cargo run            # Run development server
cargo test           # Run tests
```

### **Environment Configuration**
The dashboard connects to the Rust backend via WebSocket:
- **Backend URL**: `ws://localhost:11435/ws`
- **HTTP API**: `http://localhost:11435`
- **Environment variables** can be configured in the dashboard settings

## 🔐 Security Features

- **API Key Management** - Secure credential storage and rotation
- **CORS Configuration** - Configurable cross-origin policies
- **Rate Limiting** - Request throttling and abuse prevention
- **IP Whitelisting** - Access control by IP address
- **JWT Authentication** - Secure token-based authentication

## 📊 Performance Optimizations

- **Real-time Updates** - Efficient WebSocket communication
- **Lazy Loading** - On-demand component and data loading
- **Caching** - Strategic client-side caching for better UX
- **Responsive Charts** - Optimized data visualization
- **Progressive Enhancement** - Works without JavaScript for basic functionality

## 🌟 Future Enhancements

- [ ] **Multi-user Support** - User authentication and permissions
- [ ] **Advanced Analytics** - More detailed performance insights
- [ ] **Plugin System** - Extensible architecture for custom features
- [ ] **Mobile App** - Native mobile applications
- [ ] **Docker Support** - Containerized deployment
- [ ] **Monitoring Alerts** - Email/Slack notifications
- [ ] **Model Training** - Integrated fine-tuning capabilities
- [ ] **API Documentation** - Interactive API explorer

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]

---

**Built with ❤️ using React, TypeScript, Tailwind CSS, and Rust**