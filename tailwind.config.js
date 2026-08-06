/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        viqa: {
          bg: '#040711',
          bg2: '#080D1D',
          panel: 'rgba(10, 18, 40, 0.68)',
          border: 'rgba(90, 220, 255, 0.25)',
          cyan: '#58E6FF',
          violet: '#9F7AEA',
          gold: '#FFD76A',
          success: '#4ADE80',
          warning: '#FBBF24',
          error: '#FB7185',
          text: '#F8FAFC',
          muted: '#94A3B8',
        },
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(90, 220, 255, 0.18), 0 24px 80px rgba(0, 0, 0, 0.45)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'Inter', 'sans-serif'],
        mono: ['Orbitron', 'Inter', 'sans-serif'],
      },
      backgroundImage: {
        'viqa-radial': 'radial-gradient(circle at top, rgba(88, 230, 255, 0.18), transparent 35%), radial-gradient(circle at bottom right, rgba(159, 122, 234, 0.18), transparent 30%)',
      },
    },
  },
  plugins: [],
};