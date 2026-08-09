/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        viqa: {
          bg: '#0B1220',
          bg2: '#111827',
          panel: 'rgba(30, 41, 59, 0.82)',
          border: 'rgba(148, 163, 184, 0.20)',
          cyan: '#38BDF8',
          violet: '#818CF8',
          gold: '#FBBF24',
          success: '#34D399',
          warning: '#F59E0B',
          error: '#FB7185',
          text: '#F8FAFC',
          muted: '#94A3B8',
        },
      },
      boxShadow: {
        glow: '0 18px 48px rgba(2, 6, 23, 0.28)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'Inter', 'sans-serif'],
        mono: ['Orbitron', 'Inter', 'sans-serif'],
      },
      backgroundImage: {
        'viqa-radial': 'radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.10), transparent 35%), radial-gradient(circle at 80% 30%, rgba(129, 140, 248, 0.10), transparent 40%)',
      },
    },
  },
  plugins: [],
};
