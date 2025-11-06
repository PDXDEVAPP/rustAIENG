// Build script for Rust Ollama
// This file can be used to set up any build-time dependencies or configuration

fn main() {
    // Configure CUDA support if available
    if cfg!(feature = "candle-cuda") {
        println!("cargo:rustc-cfg=cuda_enabled");
    }
    
    // Check for required system dependencies
    #[cfg(target_os = "linux")]
    {
        // Linux-specific setup
        println!("cargo:rustc-link-lib=cuda");
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS Metal support setup
        println!("cargo:rustc-link-lib=metal");
    }
    
    println!("cargo:rerun-if-changed=build.rs");
}