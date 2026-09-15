export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0c1316', panel: '#121b1f', panel2: '#1a252a', line: '#25343a',
        ink: '#e9eff1', ink2: '#b0c1c7', ink3: '#7f9299',
        accent: '#52c4d6', good: '#67c998', warn: '#e3ab5e', crit: '#f2849b',
      },
      fontFamily: { mono: ['ui-monospace', 'Menlo', 'monospace'] },
    },
  },
}
