// Process orchestration only: every example is compiled and served by fe web dev.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { createWriteStream, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const binary = process.env.FE_BIN || 'fe';
const port = Number(process.env.FE_DEMO_PORT || 38300);
const jobs = Number(process.env.FE_DEMO_JOBS || 2);
const names = readdirSync(join(root, 'fe/web'), { withFileTypes: true })
  .filter(d => d.isDirectory() && existsSync(join(root, 'fe/web', d.name, 'index.html')))
  .map(d => d.name).sort();
if (!Number.isInteger(jobs) || jobs < 1 || !Number.isInteger(port) || port < 1024 || port + names.length > 65535) {
  throw new Error('Invalid FE_DEMO_JOBS or FE_DEMO_PORT');
}
const rows = names.map((name, i) => ({ name, port: port + i + 1, status: 'queued', seconds: null }));
if (process.argv.includes('--list')) {
  console.log(JSON.stringify(rows, null, 2));
  process.exit(0);
}
const logRoot = resolve(process.env.FE_DEMO_LOG_DIR || join(root, 'scratch/dev-examples'), String(Date.now()));
mkdirSync(logRoot, { recursive: true });
const children = new Set();
let building = 0;
let cursor = 0;
let stopping = false;
const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const server = createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  if (req.url === '/status.json') {
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ binary, logRoot, rows }));
    return;
  }
  if (req.url !== '/') { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.end(`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <meta http-equiv="refresh" content="5"><title>Quilting examples</title>
    <style>body{font:16px system-ui;max-width:65rem;margin:3rem auto;padding:0 1rem;background:#18191c;color:#ddd}
    a{color:#afc9ed}td,th{text-align:left;padding:.5rem 1.2rem .5rem 0}small{color:#aaa}code{overflow-wrap:anywhere}</style>
    <h1>Quilting examples</h1><p>Independent <code>fe web dev</code> processes. ${jobs} concurrent initial builds.
    Open only the examples you want to run. This index refreshes every five seconds.</p>
    <table><thead><tr><th>Example</th><th>Server</th><th>Initial build</th></tr></thead><tbody>
    ${rows.map(r => `<tr><td>${r.status === 'serving' ? `<a target="_blank" rel="noopener" href="http://127.0.0.1:${r.port}/">${escape(r.name)}</a>` : escape(r.name)}</td>
    <td>${escape(r.status)}</td><td>${r.seconds === null ? '—' : r.seconds.toFixed(1) + 's'}</td></tr>`).join('')}
    </tbody></table><p><small>Serving means the initial build succeeded, not that browser rendering has been verified.
    Subsequent edits use each server's normal watcher and diagnostics.</small></p>
    <p>Logs: <code>${escape(logRoot)}</code></p>`);
});

function pump() {
  while (!stopping && building < jobs && cursor < rows.length) {
    const row = rows[cursor++];
    row.status = 'building';
    building++;
    const start = performance.now();
    const log = createWriteStream(join(logRoot, row.name + '.log'));
    const child = spawn(binary, ['web', 'dev', `fe/web/${row.name}/index.html`, '--host', '127.0.0.1', '--port', String(row.port), '--color', 'never'],
      { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
    children.add(child);
    let released = false;
    let tail = '';
    const release = () => {
      if (released) return;
      released = true;
      row.seconds = (performance.now() - start) / 1000;
      building--;
      setImmediate(pump);
    };
    const output = chunk => {
      log.write(chunk);
      tail = (tail + chunk.toString()).slice(-4096);
      if (!released && tail.includes('serving Fe HTML development site at')) {
        row.status = 'serving';
        release();
        console.log(`${row.name}: http://127.0.0.1:${row.port}/`);
      }
    };
    child.stdout.on('data', output);
    child.stderr.on('data', output);
    child.on('error', error => { row.status = 'failed'; log.write(String(error)); release(); });
    child.on('close', (code, signal) => {
      children.delete(child);
      row.status = stopping ? 'stopped' : `exited (${signal || code})`;
      release();
      log.end();
      console.log(`${row.name}: ${row.status}`);
    });
  }
}
function stop() {
  if (stopping) return;
  stopping = true;
  server.close();
  for (const child of children) child.kill('SIGTERM');
}
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
server.on('error', error => { console.error(error); process.exitCode = 1; stop(); });
server.listen(port, '127.0.0.1', () => {
  console.log(`Examples: http://127.0.0.1:${port}/\nLogs: ${logRoot}\nFe: ${binary}`);
  pump();
});
