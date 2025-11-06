/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: {
          500: '#A855F7',
          glow: 'rgba(168, 85, 247, 0.35)',
        },
        bg: {
          page: '#0D1117',
          surface: '#161B22',
        },
        border: {
          default: '#30363D',
          interactive: '#8B5CF6',
        },
        text: {
          primary: '#E6EDF3',
          secondary: '#848D97',
        },
        semantic: {
          success: '#238636',
          warning: '#F5A524',
          error: '#DA3633',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
        'xxl': '48px',
        'xxxl': '64px',
      },
      borderRadius: {
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
      },
      boxShadow: {
        'interactive': '0 0 12px 0 rgba(168, 85, 247, 0.35)',
        'card-hover': '0 0 24px 0 rgba(0, 0, 0, 0.2)',
      },
      animation: {
        'slide-in': 'slideIn 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        slideIn: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}