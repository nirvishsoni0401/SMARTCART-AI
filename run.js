const { spawn } = require('child_process');
const path = require('path');

console.log('====================================================');
console.log('            STARTING SMARTCART AI PLATFORM          ');
console.log('   Backend:  http://localhost:8000                  ');
console.log('   Docs:     http://localhost:8000/docs             ');
console.log('   Frontend: http://localhost:5173                  ');
console.log('   Admin:    admin@smartcart.ai / admin123          ');
console.log('====================================================');

const isWindows = process.platform === 'win32';
const backendPython = isWindows
  ? path.join(__dirname, 'backend', 'venv', 'Scripts', 'python.exe')
  : path.join(__dirname, 'backend', 'venv', 'bin', 'python');

// 1. Launch FastAPI Backend
const backend = spawn(backendPython, ['-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000', '--reload'], {
  cwd: path.join(__dirname, 'backend'),
  stdio: 'inherit',
  shell: true,
});

// 2. Launch Vite Frontend
const npmCmd = isWindows ? 'npm.cmd' : 'npm';
const frontend = spawn(npmCmd, ['run', 'dev'], {
  cwd: path.join(__dirname, 'frontend'),
  stdio: 'inherit',
  shell: true,
});

const cleanup = () => {
  console.log('\nShutting down SmartCart AI services...');
  backend.kill();
  frontend.kill();
  process.exit(0);
};

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

