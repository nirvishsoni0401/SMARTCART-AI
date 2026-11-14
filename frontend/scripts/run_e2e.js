import { spawn, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendDir = path.resolve(__dirname, '..');
const repoRoot = path.resolve(frontendDir, '..');

const BACKEND_PORT = 8008;
const FRONTEND_PORT = 5174;

const originalDb = path.join(repoRoot, 'backend', 'smartcart.db');
const tempDb = path.join(repoRoot, 'backend', `temp_e2e_${Date.now()}.db`);

function pollUrl(url, timeoutMs = 25000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function check() {
      http
        .get(url, (res) => {
          if (res.statusCode >= 200 && res.statusCode < 400) {
            resolve();
          } else if (Date.now() - start > timeoutMs) {
            reject(new Error(`Timeout waiting for ${url} (status: ${res.statusCode})`));
          } else {
            setTimeout(check, 300);
          }
        })
        .on('error', () => {
          if (Date.now() - start > timeoutMs) {
            reject(new Error(`Timeout waiting for ${url}`));
          } else {
            setTimeout(check, 300);
          }
        });
    }
    check();
  });
}

function killProcess(child) {
  if (!child || child.killed) return;
  try {
    if (process.platform === 'win32') {
      execSync(`taskkill /pid ${child.pid} /T /F`, { stdio: 'ignore' });
    } else {
      child.kill('SIGTERM');
    }
  } catch {}
}

async function run() {
  console.log('====================================================');
  console.log('SMARTCART AI — PLAYWRIGHT HEADLESS E2E SMOKE RUNNER');
  console.log('====================================================');

  // 1. Create temporary copy of database
  if (!fs.existsSync(originalDb)) {
    throw new Error(`Original database not found at ${originalDb}`);
  }
  console.log(`[1/5] Creating temporary database copy: ${path.basename(tempDb)}`);
  fs.copyFileSync(originalDb, tempDb);

  const pythonBin = fs.existsSync(path.join(repoRoot, 'backend', 'venv', 'Scripts', 'python.exe'))
    ? path.join(repoRoot, 'backend', 'venv', 'Scripts', 'python.exe')
    : 'python';

  const tempDbPosix = tempDb.replace(/\\/g, '/');
  const backendEnv = {
    ...process.env,
    DATABASE_URL: `sqlite:///${tempDbPosix}`,
    PORT: String(BACKEND_PORT),
  };

  let backendProc = null;
  let frontendProc = null;

  try {
    // 2. Start Backend against Temporary Database
    console.log(`[2/5] Starting FastAPI backend on http://127.0.0.1:${BACKEND_PORT} (isolated DB)...`);
    backendProc = spawn(
      pythonBin,
      ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)],
      {
        cwd: path.join(repoRoot, 'backend'),
        env: backendEnv,
        stdio: 'pipe',
      }
    );

    backendProc.stderr.on('data', (d) => {
      const str = d.toString();
      if (str.includes('ERROR:') || str.includes('Traceback')) {
        console.error(`[backend error]: ${str}`);
      }
    });

    await pollUrl(`http://127.0.0.1:${BACKEND_PORT}/products/count`);
    console.log('      Backend is healthy and responding.');

    // 3. Start Vite dev server on dedicated port directly with Node
    console.log(`[3/5] Starting Vite dev server on http://127.0.0.1:${FRONTEND_PORT}...`);
    const viteJsPath = path.join(frontendDir, 'node_modules', 'vite', 'bin', 'vite.js');
    const frontendEnv = {
      ...process.env,
      VITE_API_URL: `http://127.0.0.1:${BACKEND_PORT}`,
    };

    frontendProc = spawn(
      process.execPath,
      [viteJsPath, '--port', String(FRONTEND_PORT), '--strictPort', '--host', '127.0.0.1'],
      {
        cwd: frontendDir,
        env: frontendEnv,
        stdio: 'pipe',
      }
    );

    frontendProc.stdout.on('data', (d) => {
      const str = d.toString();
      if (str.includes('ready in')) {
        console.log(`      Vite dev server started: ${str.trim()}`);
      }
    });

    frontendProc.stderr.on('data', (d) => {
      const str = d.toString();
      if (str.includes('error') || str.includes('Error')) {
        console.error(`[vite error]: ${str}`);
      }
    });

    await pollUrl(`http://127.0.0.1:${FRONTEND_PORT}`);
    console.log('      Vite dev server is ready.');

    // 4. Run Playwright Smoke Tests
    console.log(`[4/5] Executing Playwright E2E smoke tests...`);
    const playwrightBin = path.join(frontendDir, 'node_modules', '.bin', process.platform === 'win32' ? 'playwright.cmd' : 'playwright');
    
    execSync(
      `"${playwrightBin}" test`,
      {
        cwd: frontendDir,
        env: {
          ...process.env,
          PLAYWRIGHT_BASE_URL: `http://127.0.0.1:${FRONTEND_PORT}`,
        },
        stdio: 'inherit',
      }
    );

    console.log('[5/5] Playwright E2E smoke tests completed successfully.');
  } finally {
    console.log('Cleaning up processes and temporary database...');
    if (frontendProc) killProcess(frontendProc);
    if (backendProc) killProcess(backendProc);

    if (fs.existsSync(tempDb)) {
      try {
        fs.unlinkSync(tempDb);
      } catch {
        setTimeout(() => {
          try { fs.unlinkSync(tempDb); } catch {}
        }, 1000);
      }
    }
    console.log('Cleanup finished.');
  }
}

run().catch((err) => {
  console.error('E2E Runner failed:', err);
  process.exit(1);
});
