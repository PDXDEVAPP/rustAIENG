use clap::Parser;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use serde_json::json;
use std::io;
use tui::backend::{Backend, CrosstermBackend};
use tui::layout::{Constraint, Direction, Layout};
use tui::style::{Color, Modifier, Style};
use tui::text::{Span, Spans, Text};
use tui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Tabs};
use tui::Frame;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "ollama_tui")]
#[command(about = "Interactive TUI for Rust Ollama management")]
#[command(version = "0.2.0")]
struct Args {
    /// Server URL
    #[arg(short, long, default_value = "http://localhost:11434")]
    server_url: String,
    
    /// Auto-refresh interval (seconds)
    #[arg(short, long, default_value = "2")]
    refresh_interval: u64,
}

#[derive(Copy, Clone)]
enum MenuItem {
    Dashboard,
    Models,
    Performance,
    Logs,
    Settings,
}

impl MenuItem {
    fn title(&self) -> &'static str {
        match self {
            MenuItem::Dashboard => "Dashboard",
            MenuItem::Models => "Models",
            MenuItem::Performance => "Performance",
            MenuItem::Logs => "Logs",
            MenuItem::Settings => "Settings",
        }
    }
}

struct AppState {
    menu_selection: MenuItem,
    server_url: String,
    models: Vec<ModelInfo>,
    performance_metrics: PerformanceMetrics,
    logs: Vec<LogEntry>,
    is_connected: bool,
    error_message: Option<String>,
    chat_input: String,
    selected_model: Option<String>,
}

#[derive(Clone, Debug)]
struct ModelInfo {
    name: String,
    size: u64,
    modified_at: String,
    running: bool,
}

#[derive(Clone, Debug)]
struct PerformanceMetrics {
    cpu_usage: f32,
    memory_usage: f64,
    active_requests: u32,
    total_requests: u64,
    avg_response_time: f64,
    uptime_seconds: u64,
}

#[derive(Clone, Debug)]
struct LogEntry {
    timestamp: String,
    level: String,
    message: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            menu_selection: MenuItem::Dashboard,
            server_url: "http://localhost:11434".to_string(),
            models: Vec::new(),
            performance_metrics: PerformanceMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                active_requests: 0,
                total_requests: 0,
                avg_response_time: 0.0,
                uptime_seconds: 0,
            },
            logs: Vec::new(),
            is_connected: false,
            error_message: None,
            chat_input: String::new(),
            selected_model: None,
        }
    }
}

struct TuiApp {
    app_state: AppState,
}

impl TuiApp {
    fn new(server_url: String) -> Self {
        Self {
            app_state: AppState {
                server_url,
                ..Default::default()
            },
        }
    }

    async fn run<B: Backend>(&mut self, terminal: &mut tui::Terminal<B>) -> io::Result<()> {
        loop {
            // Draw UI
            terminal.draw(|f| {
                self.draw_ui(f);
            })?;

            // Handle input
            if event::poll(std::time::Duration::from_millis(250))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Char('h') => self.app_state.menu_selection = MenuItem::Dashboard,
                            KeyCode::Char('m') => self.app_state.menu_selection = MenuItem::Models,
                            KeyCode::Char('p') => self.app_state.menu_selection = MenuItem::Performance,
                            KeyCode::Char('l') => self.app_state.menu_selection = MenuItem::Logs,
                            KeyCode::Char('s') => self.app_state.menu_selection = MenuItem::Settings,
                            KeyCode::Tab => self.next_menu_item(),
                            KeyCode::BackTab => self.prev_menu_item(),
                            KeyCode::Enter => self.handle_enter(),
                            KeyCode::Esc => self.handle_escape(),
                            KeyCode::Char(c) => self.handle_char_input(c),
                            KeyCode::Backspace => self.handle_backspace(),
                            _ => {}
                        }
                    }
                }
            }

            // Refresh data
            self.refresh_data().await;
        }
    }

    fn next_menu_item(&mut self) {
        self.app_state.menu_selection = match self.app_state.menu_selection {
            MenuItem::Dashboard => MenuItem::Models,
            MenuItem::Models => MenuItem::Performance,
            MenuItem::Performance => MenuItem::Logs,
            MenuItem::Logs => MenuItem::Settings,
            MenuItem::Settings => MenuItem::Dashboard,
        };
    }

    fn prev_menu_item(&mut self) {
        self.app_state.menu_selection = match self.app_state.menu_selection {
            MenuItem::Dashboard => MenuItem::Settings,
            MenuItem::Models => MenuItem::Dashboard,
            MenuItem::Performance => MenuItem::Models,
            MenuItem::Logs => MenuItem::Performance,
            MenuItem::Settings => MenuItem::Logs,
        };
    }

    fn handle_enter(&mut self) {
        match self.app_state.menu_selection {
            MenuItem::Models => {
                // Select model for management
            }
            MenuItem::Settings => {
                // Apply settings
            }
            _ => {}
        }
    }

    fn handle_escape(&mut self) {
        self.app_state.chat_input.clear();
        self.app_state.selected_model = None;
    }

    fn handle_char_input(&mut self, c: char) {
        if self.app_state.menu_selection == MenuItem::Dashboard && !self.app_state.chat_input.is_empty() {
            self.app_state.chat_input.push(c);
        }
    }

    fn handle_backspace(&mut self) {
        if !self.app_state.chat_input.is_empty() {
            self.app_state.chat_input.pop();
        }
    }

    async fn refresh_data(&mut self) {
        // Fetch data from server
        if let Err(e) = self.fetch_server_data().await {
            warn!("Failed to fetch server data: {}", e);
            self.app_state.error_message = Some(e.to_string());
        } else {
            self.app_state.is_connected = true;
            self.app_state.error_message = None;
        }

        // Generate mock data for demonstration
        self.generate_mock_data();
    }

    async fn fetch_server_data(&self) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        
        // Check health
        let health_response = client
            .get(&format!("{}/health", self.app_state.server_url))
            .send()
            .await?;
            
        if !health_response.status().is_success() {
            return Err("Server not responding".into());
        }

        // Fetch models
        let models_response = client
            .post(&format!("{}/api/list", self.app_state.server_url))
            .json(&json!({}))
            .send()
            .await?;
            
        if models_response.status().is_success() {
            let models_data: serde_json::Value = models_response.json().await?;
            // Parse models data
        }

        Ok(())
    }

    fn generate_mock_data(&mut self) {
        // Mock models
        self.app_state.models = vec![
            ModelInfo {
                name: "llama3.2".to_string(),
                size: 2048000000,
                modified_at: "2024-01-15T10:30:00Z".to_string(),
                running: true,
            },
            ModelInfo {
                name: "mistral7b".to_string(),
                size: 1800000000,
                modified_at: "2024-01-14T15:20:00Z".to_string(),
                running: false,
            },
            ModelInfo {
                name: "codellama".to_string(),
                size: 2200000000,
                modified_at: "2024-01-13T09:15:00Z".to_string(),
                running: true,
            },
        ];

        // Mock performance metrics
        self.app_state.performance_metrics = PerformanceMetrics {
            cpu_usage: 45.2 + (rand::random::<f32>() * 10.0 - 5.0),
            memory_usage: 65.8 + (rand::random::<f32>() * 5.0 - 2.5),
            active_requests: 3 + (rand::random::<u32>() % 5),
            total_requests: 1247,
            avg_response_time: 850.0 + (rand::random::<f64>() * 200.0 - 100.0),
            uptime_seconds: 3600 + rand::random::<u64>() % 7200,
        };

        // Mock logs
        self.app_state.logs = vec![
            LogEntry {
                timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                level: "INFO".to_string(),
                message: "Model llama3.2 loaded successfully".to_string(),
            },
            LogEntry {
                timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                level: "DEBUG".to_string(),
                message: "Request processed in 245ms".to_string(),
            },
            LogEntry {
                timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                level: "WARN".to_string(),
                message: "High memory usage detected".to_string(),
            },
        ];
    }

    fn draw_ui<B: Backend>(&mut self, f: &mut Frame<B>) {
        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.size());

        // Draw header
        self.draw_header(f, chunks[0]);

        // Draw main content
        self.draw_content(f, chunks[1]);
    }

    fn draw_header<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let header_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        // Connection status
        let status_style = if self.app_state.is_connected {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Red)
        };

        let connection_status = Paragraph::new(Spans::from(vec![
            Span::styled("Rust Ollama TUI", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" - "),
            Span::styled(
                if self.app_state.is_connected { "Connected" } else { "Disconnected" },
                status_style,
            ),
        ]))
        .block(Block::default().borders(Borders::ALL).title("Connection"));

        f.render_widget(connection_status, header_chunks[0]);

        // Quick stats
        let stats_text = format!(
            "Models: {} | Requests: {} | CPU: {:.1}% | Memory: {:.1}%",
            self.app_state.models.len(),
            self.app_state.performance_metrics.total_requests,
            self.app_state.performance_metrics.cpu_usage,
            self.app_state.performance_metrics.memory_usage
        );

        let stats = Paragraph::new(stats_text)
            .style(Style::default())
            .block(Block::default().borders(Borders::ALL).title("Quick Stats"));

        f.render_widget(stats, header_chunks[1]);
    }

    fn draw_content<B: Backend>(&mut self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let vertical_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        // Draw tabs
        self.draw_tabs(f, vertical_chunks[0]);

        // Draw content based on selected tab
        match self.app_state.menu_selection {
            MenuItem::Dashboard => self.draw_dashboard(f, vertical_chunks[1]),
            MenuItem::Models => self.draw_models(f, vertical_chunks[1]),
            MenuItem::Performance => self.draw_performance(f, vertical_chunks[1]),
            MenuItem::Logs => self.draw_logs(f, vertical_chunks[1]),
            MenuItem::Settings => self.draw_settings(f, vertical_chunks[1]),
        }
    }

    fn draw_tabs<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let tabs = vec![
            MenuItem::Dashboard.title(),
            MenuItem::Models.title(),
            MenuItem::Performance.title(),
            MenuItem::Logs.title(),
            MenuItem::Settings.title(),
        ];

        let selected_tab = match self.app_state.menu_selection {
            MenuItem::Dashboard => 0,
            MenuItem::Models => 1,
            MenuItem::Performance => 2,
            MenuItem::Logs => 3,
            MenuItem::Settings => 4,
        };

        let tabs_widget = Tabs::new(tabs)
            .block(Block::default().borders(Borders::ALL).title("Navigation"))
            .select(selected_tab)
            .style(Style::default().fg(Color::Cyan))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD).bg(Color::Blue));

        f.render_widget(tabs_widget, area);
    }

    fn draw_dashboard<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left side - System overview
        let system_info = vec![
            Spans::from(Span::raw("System Overview")),
            Spans::from(Span::raw("")),
            Spans::from(Span::raw(format!("Uptime: {} hours", self.app_state.performance_metrics.uptime_seconds / 3600))),
            Spans::from(Span::raw(format!("Active Requests: {}", self.app_state.performance_metrics.active_requests))),
            Spans::from(Span::raw(format!("Avg Response Time: {:.0}ms", self.app_state.performance_metrics.avg_response_time))),
        ];

        let system_block = Paragraph::new(system_info)
            .block(Block::default().borders(Borders::ALL).title("System Information"))
            .style(Style::default());

        f.render_widget(system_block, chunks[0]);

        // Right side - Model status
        let running_models: Vec<_> = self.app_state.models
            .iter()
            .filter(|m| m.running)
            .collect();

        let model_info = vec![
            Spans::from(Span::raw("Model Status")),
            Spans::from(Span::raw("")),
            Spans::from(Span::raw(format!("Total Models: {}", self.app_state.models.len()))),
            Spans::from(Span::raw(format!("Running Models: {}", running_models.len()))),
            Spans::from(Span::raw("")),
        ];

        for model in &running_models {
            model_info.push(Span::raw(format!("• {} ({} MB)", model.name, model.size / 1024 / 1024)));
        }

        let model_block = Paragraph::new(model_info)
            .block(Block::default().borders(Borders::ALL).title("Models"))
            .scroll((0, 0));

        f.render_widget(model_block, chunks[1]);
    }

    fn draw_models<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        // Model list
        let model_items: Vec<ListItem> = self.app_state.models
            .iter()
            .map(|model| {
                let status = if model.running { "● Running" } else { "○ Stopped" };
                let content = format!("{} - {} MB - {} - {}", 
                    model.name, 
                    model.size / 1024 / 1024, 
                    status,
                    model.modified_at
                );
                ListItem::new(Span::raw(content))
            })
            .collect();

        let models_list = List::new(model_items)
            .block(Block::default().borders(Borders::ALL).title("Models"))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));

        f.render_widget(models_list, chunks[0]);

        // Model details/actions
        let details = vec![
            Spans::from(Span::raw("Actions")),
            Spans::from(Span::raw("")),
            Spans::from(Span::raw("r - Refresh")),
            Spans::from(Span::raw("p - Pull model")),
            Spans::from(Span::raw("s - Start/Stop")),
            Spans::from(Span::raw("d - Delete")),
            Spans::from(Span::raw("c - Copy model")),
        ];

        let actions = Paragraph::new(details)
            .block(Block::default().borders(Borders::ALL).title("Commands"));

        f.render_widget(actions, chunks[1]);
    }

    fn draw_performance<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Length(3), Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        // CPU Usage
        let cpu_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("CPU Usage"))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent(self.app_state.performance_metrics.cpu_usage as u16);

        f.render_widget(cpu_gauge, chunks[0]);

        // Memory Usage
        let memory_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Memory Usage"))
            .gauge_style(Style::default().fg(Color::Magenta))
            .percent(self.app_state.performance_metrics.memory_usage as u16);

        f.render_widget(memory_gauge, chunks[1]);

        // Response Time
        let response_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Avg Response Time"))
            .gauge_style(Style::default().fg(Color::Green))
            .percent(((self.app_state.performance_metrics.avg_response_time / 2000.0) * 100.0) as u16);

        f.render_widget(response_gauge, chunks[2]);

        // Detailed metrics
        let metrics_text = vec![
            Spans::from(Span::raw("Performance Details")),
            Spans::from(Span::raw("")),
            Spans::from(Span::raw(format!("Total Requests: {}", self.app_state.performance_metrics.total_requests))),
            Spans::from(Span::raw(format!("Active Requests: {}", self.app_state.performance_metrics.active_requests))),
            Spans::from(Span::raw(format!("Uptime: {} seconds", self.app_state.performance_metrics.uptime_seconds))),
        ];

        let metrics_block = Paragraph::new(metrics_text)
            .block(Block::default().borders(Borders::ALL).title("Metrics"));

        f.render_widget(metrics_block, chunks[3]);
    }

    fn draw_logs<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let log_items: Vec<ListItem> = self.app_state.logs
            .iter()
            .map(|log| {
                let style = match log.level.as_str() {
                    "ERROR" => Style::default().fg(Color::Red),
                    "WARN" => Style::default().fg(Color::Yellow),
                    "DEBUG" => Style::default().fg(Color::Blue),
                    _ => Style::default().fg(Color::White),
                };
                
                let content = format!("[{}] {}: {}", log.timestamp, log.level, log.message);
                ListItem::new(Span::styled(content, style))
            })
            .collect();

        let logs_list = List::new(log_items)
            .block(Block::default().borders(Borders::ALL).title("Logs"))
            .scroll((0, 0));

        f.render_widget(logs_list, area);
    }

    fn draw_settings<B: Backend>(&self, f: &mut Frame<B>, area: tui::layout::Rect) {
        let settings_text = vec![
            Spans::from(Span::raw("Settings")),
            Spans::from(Span::raw("")),
            Spans::from(Span::raw(format!("Server URL: {}", self.app_state.server_url))),
            Spans::from(Span::raw("")),
            Spans::from(Span::raw("Available Settings:")),
            Spans::from(Span::raw("• Server configuration")),
            Spans::from(Span::raw("• Performance tuning")),
            Spans::from(Span::raw("• Model preferences")),
            Spans::from(Span::raw("• Logging configuration")),
        ];

        let settings_block = Paragraph::new(settings_text)
            .block(Block::default().borders(Borders::ALL).title("Settings"))
            .scroll((0, 0));

        f.render_widget(settings_block, area);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = tui::Terminal::new(backend)?;

    // Create and run app
    let mut app = TuiApp::new(args.server_url);
    
    let result = app.run(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        println!("{:?}", err)
    }

    Ok(())
}