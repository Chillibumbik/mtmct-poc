import { type ChildProcess, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const REPO = path.resolve(import.meta.dirname, '..', '..')
const API = 'http://127.0.0.1:8000/api/system'

async function up(url: string): Promise<boolean> {
  try {
    return (await fetch(url, { signal: AbortSignal.timeout(1500) })).ok
  } catch {
    return false
  }
}

/**
 * Starts api + worker on an isolated MTMCT_HOME (fresh db/cache/outputs; configs, data, models are symlinked).
 * If an API is already listening on :8000 (e.g. `make dev`) it is reused as is and the run lands in its database.
 */
export default async function globalSetup() {
  if (await up(API)) return
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'mtmct-e2e-'))
  for (const d of ['configs', 'data', 'models']) fs.symlinkSync(path.join(REPO, d), path.join(home, d))
  const env = { ...process.env, MTMCT_HOME: home }
  const children: ChildProcess[] = ['api', 'worker'].map((cmd) =>
    spawn('uv', ['run', 'mtmct', cmd], { cwd: REPO, env, stdio: 'inherit', detached: true }),
  )
  for (let i = 0; i < 60 && !(await up(API)); i++) await new Promise((r) => setTimeout(r, 1000))
  if (!(await up(API))) {
    children.forEach((c) => c.pid && process.kill(-c.pid))
    throw new Error('API did not start on 127.0.0.1:8000')
  }
  return () => {
    // SIGTERM to the whole process group (uv + python child)
    children.forEach((c) => c.pid && process.kill(-c.pid, 'SIGTERM'))
    fs.rmSync(home, { recursive: true, force: true })
  }
}
