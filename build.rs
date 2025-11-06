// Build script for Rust Ollama
// This file can be used to set up any build-time dependencies or configuration

fn main() {
    // Check for required system dependencies
    #[cfg(target_os = "linux")]
    {
        // Linux-specific setup
        println!("cargo:rustc-link-lib=stdc++");
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS Metal support setup
        println!("cargo:rustc-link-lib=metal");
    }
    
    println!("cargo:rerun-if-changed=build.rs");
}