/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: {
          light: '#F7F7F8',
          dark: '#0B0F19',
        },
        surface: {
          light: '#FFFFFF',
          dark: '#111827',
        },
        'text-primary': {
          light: '#111827',
          dark: '#F9FAFB',
        },
        'text-secondary': {
          light: '#6B7280',
          dark: '#9CA3AF',
        },
        border: {
          light: '#E5E7EB',
          dark: '#1F2937',
        },
        accent: '#2563EB',
        success: '#16A34A',
        warning: '#D97706',
        error: '#DC2626',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      fontSize: {
        'xs': ['12px', { lineHeight: '1.4' }],
        'sm': ['14px', { lineHeight: '1.4' }],
        'base': ['16px', { lineHeight: '1.6' }],
        'lg': ['20px', { lineHeight: '1.4' }],
      },
      spacing: {
        '18': '4.5rem',
      },
      borderRadius: {
        'btn': '10px',
        'panel': '14px',
        'pill': '999px',
      },
      boxShadow: {
        'panel': '0 8px 30px rgba(0,0,0,0.08)',
        'panel-dark': '0 8px 30px rgba(0,0,0,0.25)',
      },
      maxWidth: {
        'chat': '980px',
      },
    },
  },
  plugins: [],
}
