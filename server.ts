import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import multer from 'multer';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';

const app = express();
app.set('trust proxy', 1);

/* =========================================================
   CONFIGURACIÓN GENERAL
   ========================================================= */

const PORT = Number(process.env.PORT) || 3000;
const modelName = process.env.GEMINI_MODEL || 'gemini-3-flash-preview';
const API_TOKEN = process.env.API_TOKEN || '';
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const PDF_RENDER_TIMEOUT_MS = 20_000;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE },
});

/* =========================================================
   CORS / BODY
   ========================================================= */

app.use(
  cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key'],
  })
);
app.options('/{*splat}', cors());
app.use(express.json({ limit: '60mb' }));
app.use(express.text({ type: ['text/plain'], limit: '60mb' }));

/* =========================================================
   GEMINI API KEYS
   ========================================================= */

const apiKeys = [
  process.env.GEMINI_API_KEY,
  process.env.GEMINI_API_KEY_1,
  process.env.GEMINI_API_KEY_2,
  process.env.GEMINI_API_KEY_3,
  process.env.GEMINI_API_KEY_4,
  process.env.GEMINI_API_KEY_5,
  process.env.GEMINI_API_KEY_6,
  process.env.GEMINI_API_KEY_7,
  process.env.GEMINI_API_KEY_8,
  process.env.GEMINI_API_KEY_9,
  process.env.GEMINI_API_KEY_10,
]
  .map((key) => (typeof key === 'string' ? key.trim() : ''))
  .filter((key): key is string => key.length > 0)
  .filter((key, index, all) => all.indexOf(key) === index);

if (apiKeys.length === 0) {
  console.error(
    'ERROR: Gemini no está configurado. Usá GEMINI_API_KEY o GEMINI_API_KEY_1...GEMINI_API_KEY_10.'
  );
} else {
  console.log(`Gemini: ${apiKeys.length} API key(s) detectada(s).`);
}

interface KeyState {
  ai: GoogleGenAI;
  blockedUntil: number;
}

const keyStates: KeyState[] = apiKeys.map((key) => ({
  ai: new GoogleGenAI({ apiKey: key }),
  blockedUntil: 0,
}));

let currentKeyIndex = 0;

function getAvailableKey(): KeyState | null {
  if (keyStates.length === 0) return null;
  const now = Date.now();

  for (let i = 0; i < keyStates.length; i++) {
    const index = (currentKeyIndex + i) % keyStates.length;
    const keyState = keyStates[index];
    if (keyState.blockedUntil <= now) {
      currentKeyIndex = (index + 1) % keyStates.length;
      return keyState;
    }
  }
  return null;
}

function blockKey(keyState: KeyState, retryAfterMs: number) {
  keyState.blockedUntil = Date.now() + retryAfterMs;
}

function getRetryAfterMs(error: unknown): number {
  const message = error instanceof Error ? error.message : String(error);
  const match = message.match(/retry in\s+([\d.]+)s/i);
  if (match) {
    const seconds = Number.parseFloat(match[1]);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.ceil(seconds * 1000);
    }
  }
  return 30_000;
}

function isRateLimitError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  const normalized = message.toLowerCase();
  return (
    normalized.includes('429') ||
    normalized.includes('too_many_requests') ||
    normalized.includes('ratelimiterror') ||
    normalized.includes('rate limit') ||
    normalized.includes('quota exceeded') ||
    normalized.includes('resource exhausted') ||
    normalized.includes('resource_exhausted')
  );
}

async function runGemini<T>(
  operation: (ai: GoogleGenAI) => Promise<T>
): Promise<T> {
  if (keyStates.length === 0) {
    throw new Error('No hay ninguna API key de Gemini configurada en el servidor.');
  }

  const attemptedKeys = new Set<KeyState>();

  for (let attempt = 0; attempt < keyStates.length; attempt++) {
    const keyState = getAvailableKey();
    if (!keyState) {
      throw new Error('Todas las API keys de Gemini están temporalmente limitadas.');
    }
    if (attemptedKeys.has(keyState)) break;
    attemptedKeys.add(keyState);

    const keyNumber = keyStates.indexOf(keyState) + 1;

    try {
      console.log(`Gemini: usando API key ${keyNumber}`);
      return await operation(keyState.ai);
    } catch (error: unknown) {
      if (!isRateLimitError(error)) throw error;

      const retryAfterMs = getRetryAfterMs(error);
      console.warn(
        `Gemini: API key ${keyNumber} alcanzó el límite. Bloqueando durante ${Math.ceil(
          retryAfterMs / 1000
        )}s.`
      );
      blockKey(keyState, retryAfterMs);
    }
  }

  throw new Error('Todas las API keys de Gemini alcanzaron el límite.');
}

/* =========================================================
   HELPERS
   ========================================================= */

function getRequestBody(req: Request): Record<string, any> {
  let body: unknown = req.body;
  if (typeof body === 'string') {
    if (body.trim().length === 0) return {};
    try {
      body = JSON.parse(body);
    } catch {
      return {};
    }
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) return {};
  return body as Record<string, any>;
}

function parseGeminiJson(text: string): any {
  const cleaned = text
    .trim()
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const objectMatch = cleaned.match(/\{[\s\S]*\}/);
    if (objectMatch) return JSON.parse(objectMatch[0]);

    const arrayMatch = cleaned.match(/\[[\s\S]*\]/);
    if (arrayMatch) return JSON.parse(arrayMatch[0]);

    throw new Error('Gemini devolvió un JSON inválido.');
  }
}

function auth(req: Request, res: Response, next: NextFunction) {
  if (!API_TOKEN) return next();

  const sent =
    req.headers['x-api-key'] ||
    (req.headers.authorization || '').replace(/^Bearer\s+/i, '');

  if (sent !== API_TOKEN) {
    return res.status(401).json({ error: 'no autorizado' });
  }
  return next();
}

function getParamString(value: unknown): string {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return String(value[0] || '');
  return '';
}

/* =========================================================
   GITHUB STORAGE
   ========================================================= */

function ghConfig() {
  const owner = process.env.GITHUB_OWNER;
  const repo = process.env.GITHUB_REPO;
  const branch = process.env.GITHUB_BRANCH || 'main';
  const token = process.env.GITHUB_TOKEN;

  if (!owner || !repo || !token) {
    throw new Error('Faltan GITHUB_OWNER, GITHUB_REPO o GITHUB_TOKEN en Render.');
  }

  return { owner, repo, branch, token };
}

function validateGithubPath(path: string): string {
  const normalized = path.replace(/^\/+/, '').replace(/\\/g, '/');
  if (!normalized || normalized.includes('..') || normalized.startsWith('.git/')) {
    throw new Error('Ruta de archivo inválida.');
  }
  return normalized;
}

function encodePathForGithub(path: string): string {
  return path.split('/').map((part) => encodeURIComponent(part)).join('/');
}

function encodePathB64(path: string): string {
  return Buffer.from(path, 'utf8')
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function decodePathB64(value: string): string {
  let normalized = value.replace(/-/g, '+').replace(/_/g, '/');
  while (normalized.length % 4 !== 0) normalized += '=';
  return validateGithubPath(Buffer.from(normalized, 'base64').toString('utf8'));
}

function githubHeaders(token: string) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'Content-Type': 'application/json',
    'User-Agent': 'torchill-api',
  };
}

async function getGithubFile(path: string) {
  const { owner, repo, branch, token } = ghConfig();
  const encodedPath = encodePathForGithub(path);
  const url =
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/${encodedPath}` +
    `?ref=${encodeURIComponent(branch)}`;

  const response = await fetch(url, { headers: githubHeaders(token) });
  if (!response.ok) {
    if (response.status === 404) return null;
    const text = await response.text();
    throw new Error(`GitHub GET error ${response.status}: ${text}`);
  }
  return response.json();
}

async function uploadToGithub(path: string, data: Buffer) {
  const { owner, repo, branch, token } = ghConfig();
  const safePath = validateGithubPath(path);

  if (data.length > MAX_FILE_SIZE) {
    throw new Error('El archivo supera el límite máximo de 20 MB.');
  }

  const existing = await getGithubFile(safePath);
  const url =
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/${encodePathForGithub(safePath)}`;

  const body: Record<string, any> = {
    message: `Torchill receipt upload: ${safePath}`,
    content: data.toString('base64'),
    branch,
  };

  if (existing && typeof existing.sha === 'string') body.sha = existing.sha;

  const response = await fetch(url, {
    method: 'PUT',
    headers: githubHeaders(token),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`GitHub upload error ${response.status}: ${text}`);
  }

  const result = await response.json();
  return {
    path: safePath,
    sha: result.content?.sha || null,
    branch,
    owner,
    repo,
    pathB64: encodePathB64(safePath),
  };
}

function mimeForPath(path: string): string {
  const lower = String(path || '').toLowerCase();
  if (lower.endsWith('.pdf')) return 'application/pdf';
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.gif')) return 'image/gif';
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  return 'application/octet-stream';
}

/* =========================================================
   PDF -> PNG AISLADO EN PROCESO HIJO

   IMPORTANTE:
   - El proceso principal NO renderiza PDFs con PDF.js.
   - Si pdfjs-dist / @napi-rs/canvas falla nativamente, solamente muere
     el proceso hijo.
   - El caller puede hacer fallback al PDF original.
   ========================================================= */

const PDF_WORKER_SCRIPT = String.raw`
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');
const { createCanvas } = require('@napi-rs/canvas');

class SafeCanvasFactory {
  create(width, height) {
    const canvas = createCanvas(Math.max(1, Math.ceil(width)), Math.max(1, Math.ceil(height)));
    const context = canvas.getContext('2d');
    return { canvas, context };
  }
  reset(canvasAndContext, width, height) {
    // @napi-rs/canvas puede fallar si PDF.js intenta mutar/destruir
    // referencias nativas compartidas. Creamos un canvas nuevo.
    const canvas = createCanvas(Math.max(1, Math.ceil(width)), Math.max(1, Math.ceil(height)));
    canvasAndContext.canvas = canvas;
    canvasAndContext.context = canvas.getContext('2d');
  }
  destroy(canvasAndContext) {
    // NO hacer canvas.width = 0 ni canvas.height = 0.
    // Ese patrón es justamente el que puede disparar InvalidArg en N-API.
    canvasAndContext.context = null;
    canvasAndContext.canvas = null;
  }
}

(async () => {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  const input = Buffer.concat(chunks);
  let pdf = null;

  try {
    const factory = new SafeCanvasFactory();
    const task = pdfjsLib.getDocument({
      data: new Uint8Array(input),
      isEvalSupported: false,
      useSystemFonts: false,
      CanvasFactory: SafeCanvasFactory,
    });

    pdf = await task.promise;
    if (pdf.numPages < 1) throw new Error('El PDF no contiene páginas.');

    const page = await pdf.getPage(1);
    const base = page.getViewport({ scale: 1 });
    const maxSide = 1600;
    const scale = Math.min(2, maxSide / Math.max(base.width, base.height));
    const viewport = page.getViewport({ scale });

    const holder = factory.create(viewport.width, viewport.height);
    await page.render({
      canvasContext: holder.context,
      viewport,
      canvasFactory: factory,
    }).promise;

    const png = holder.canvas.toBuffer('image/png');
    process.stdout.write(png);
    factory.destroy(holder);
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exitCode = 2;
  } finally {
    try { if (pdf) await pdf.destroy(); } catch (_) {}
  }
})();
`;

async function pdfFirstPageToPngSafe(buffer: Buffer): Promise<Buffer | null> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['-e', PDF_WORKER_SCRIPT], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: process.env,
    });

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    let settled = false;

    const finish = (result: Buffer | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    const timer = setTimeout(() => {
      console.warn('PDF -> PNG: timeout; se usará PDF nativo.');
      try { child.kill('SIGKILL'); } catch {}
      finish(null);
    }, PDF_RENDER_TIMEOUT_MS);

    child.stdout.on('data', (chunk) => stdoutChunks.push(Buffer.from(chunk)));
    child.stderr.on('data', (chunk) => stderrChunks.push(Buffer.from(chunk)));

    child.on('error', (error) => {
      console.warn('PDF -> PNG: no se pudo iniciar worker:', error.message);
      finish(null);
    });

    child.on('close', (code, signal) => {
      if (settled) return;

      const stderr = Buffer.concat(stderrChunks).toString('utf8').trim();
      const png = Buffer.concat(stdoutChunks);

      if (code === 0 && png.length > 8 && png.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) {
        console.log(`PDF -> PNG: conversión exitosa (${png.length} bytes).`);
        return finish(png);
      }

      console.warn(
        `PDF -> PNG: worker falló (code=${code}, signal=${signal || 'none'}). ` +
        `Fallback a PDF nativo.${stderr ? ` Detalle: ${stderr.slice(0, 700)}` : ''}`
      );
      return finish(null);
    });

    child.stdin.on('error', () => {});
    child.stdin.end(buffer);
  });
}

/* =========================================================
   RECEIPT HELPERS
   ========================================================= */

function getMimeType(fileUrl: string, contentType: string): string {
  const detected = contentType.split(';')[0].trim().toLowerCase();
  if (detected && detected !== 'application/octet-stream') return detected;

  const lowerUrl = fileUrl.toLowerCase();
  if (lowerUrl.includes('.pdf')) return 'application/pdf';
  if (lowerUrl.includes('.png')) return 'image/png';
  if (lowerUrl.includes('.webp')) return 'image/webp';
  if (lowerUrl.includes('.gif')) return 'image/gif';
  return 'image/jpeg';
}

async function downloadReceipt(fileUrl: string): Promise<{ buffer: Buffer; mimeType: string }> {
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(fileUrl);
  } catch {
    throw new Error('fileUrl no es una URL válida.');
  }

  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    throw new Error('fileUrl debe utilizar HTTP o HTTPS.');
  }

  const fileRes = await fetch(fileUrl);
  if (!fileRes.ok) {
    throw new Error(`No se pudo descargar el archivo (${fileRes.status}).`);
  }

  const contentLength = fileRes.headers.get('content-length');
  if (contentLength && Number(contentLength) > MAX_FILE_SIZE) {
    throw new Error('El archivo supera el límite máximo de 20 MB.');
  }

  if (!fileRes.body) throw new Error('La respuesta no contiene datos.');

  const reader = fileRes.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        total += value.length;
        if (total > MAX_FILE_SIZE) {
          try { await reader.cancel(); } catch {}
          throw new Error('El archivo supera el límite máximo de 20 MB.');
        }
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }

  const buffer = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
  const mimeType = getMimeType(
    fileUrl,
    fileRes.headers.get('content-type') || ''
  );

  return { buffer, mimeType };
}

/* =========================================================
   TAROT
   ========================================================= */

interface TarotFile {
  name: string;
  sha: string;
  size: number;
}

let tarotDirectoryCache: { files: TarotFile[]; timestamp: number } | null = null;
const TAROT_DIRECTORY_TTL = 10 * 60 * 1000;
const tarotOriginalCache = new Map<
  number,
  { buffer: Buffer; timestamp: number; mimeType: string; width: number; height: number; sha: string }
>();
const TAROT_ORIGINAL_TTL = 5 * 60 * 1000;

async function listTarotFiles(forceRefresh = false): Promise<TarotFile[]> {
  if (
    !forceRefresh &&
    tarotDirectoryCache &&
    Date.now() - tarotDirectoryCache.timestamp < TAROT_DIRECTORY_TTL
  ) {
    return tarotDirectoryCache.files;
  }

  const { owner, repo, branch, token } = ghConfig();
  const url =
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/tarot?ref=${encodeURIComponent(branch)}`;

  const response = await fetch(url, { headers: githubHeaders(token) });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`No se pudo listar /tarot (${response.status}): ${text}`);
  }

  const data = await response.json();
  if (!Array.isArray(data)) throw new Error('/tarot no es un directorio válido.');

  const files: TarotFile[] = data
    .filter(
      (item: any) =>
        item?.type === 'file' &&
        typeof item.name === 'string' &&
        /\.(jpg|jpeg|png|webp)$/i.test(item.name)
    )
    .map((item: any) => ({
      name: item.name,
      sha: String(item.sha || ''),
      size: Number(item.size || 0),
    }));

  tarotDirectoryCache = { files, timestamp: Date.now() };
  console.log(`Tarot: ${files.length} imágenes encontradas.`);
  return files;
}

function findTarotFile(number: number, files: TarotFile[]): TarotFile | null {
  const regex = new RegExp(`juli[\\s_-]*0*${number}(?=\\D|$)`, 'i');
  return files.find((file) => regex.test(file.name)) || null;
}

async function fetchTarotOriginal(number: number, file: TarotFile) {
  const cached = tarotOriginalCache.get(number);
  if (
    cached &&
    cached.sha === file.sha &&
    Date.now() - cached.timestamp < TAROT_ORIGINAL_TTL
  ) {
    return cached;
  }

  const { owner, repo, branch, token } = ghConfig();
  const url =
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/tarot/${encodeURIComponent(file.name)}` +
    `?ref=${encodeURIComponent(branch)}`;

  const response = await fetch(url, {
    headers: { ...githubHeaders(token), Accept: 'application/vnd.github.raw' },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`No se pudo descargar ${file.name}: ${text}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  let width = 0;
  let height = 0;

  try {
    const image = await loadImage(buffer);
    width = image.width;
    height = image.height;
  } catch (error) {
    console.warn(`Tarot: no se pudieron leer dimensiones de ${file.name}`, error);
  }

  const entry = {
    buffer,
    timestamp: Date.now(),
    mimeType: mimeForPath(file.name),
    width,
    height,
    sha: file.sha,
  };

  tarotOriginalCache.set(number, entry);
  return entry;
}

/* =========================================================
   HEALTH
   ========================================================= */

app.get('/health', (_req: Request, res: Response) => {
  return res.status(200).json({
    ok: true,
    service: 'torchill-api',
    model: modelName,
    gemini: apiKeys.length > 0,
    geminiConfigured: apiKeys.length > 0,
    geminiKeys: apiKeys.length,
    githubConfigured: Boolean(
      process.env.GITHUB_OWNER && process.env.GITHUB_REPO && process.env.GITHUB_TOKEN
    ),
    pdfModes: ['auto', 'image', 'native'],
  });
});

app.get('/api/gemini/project-copy', (_req: Request, res: Response) => {
  return res.status(200).json({
    status: 'ok',
    service: 'torchill-api',
    model: modelName,
    geminiConfigured: apiKeys.length > 0,
    geminiKeys: apiKeys.length,
    message: 'Torchill API funcionando correctamente.',
  });
});

/* =========================================================
   UPLOAD
   ========================================================= */

app.post('/upload', auth, upload.single('file'), async (req: Request, res: Response) => {
  try {
    const uploadedFile = (req as any).file as { buffer?: Buffer; originalname?: string; mimetype?: string } | undefined;
    let buffer: Buffer | null = uploadedFile?.buffer || null;
    let originalName = uploadedFile?.originalname || '';
    let mimeType = uploadedFile?.mimetype || '';
    const body = getRequestBody(req);

    if (!buffer) {
      const dataBase64 =
        typeof body.dataBase64 === 'string'
          ? body.dataBase64
          : typeof body.base64 === 'string'
            ? body.base64
            : '';

      if (dataBase64) {
        const clean = dataBase64.replace(/^data:[^;]+;base64,/i, '');
        buffer = Buffer.from(clean, 'base64');
        originalName =
          typeof body.fileName === 'string'
            ? body.fileName
            : `receipt.${typeof body.ext === 'string' ? body.ext : 'jpg'}`;
        mimeType = typeof body.mimeType === 'string' ? body.mimeType : '';
      }
    }

    if (!buffer) {
      return res.status(400).json({
        error:
          'Falta el archivo. Usá multipart/form-data con campo "file" o JSON con "dataBase64"/"base64".',
      });
    }

    if (buffer.length > MAX_FILE_SIZE) {
      return res.status(413).json({ error: 'El archivo supera el límite máximo de 20 MB.' });
    }

    const safeName = String(originalName || 'receipt').replace(/[^a-zA-Z0-9._-]/g, '_');
    const extension = safeName.includes('.') ? safeName.split('.').pop()?.toLowerCase() : '';
    const finalExtension =
      extension ||
      (mimeType === 'application/pdf'
        ? 'pdf'
        : mimeType === 'image/png'
          ? 'png'
          : mimeType === 'image/webp'
            ? 'webp'
            : 'jpg');

    let requestedPath = typeof body.path === 'string' ? body.path : '';
    if (!requestedPath) {
      const now = new Date();
      const ym = `${now.getUTCFullYear()}${String(now.getUTCMonth() + 1).padStart(2, '0')}`;
      requestedPath = `receipts/${ym}/${Date.now()}-${crypto.randomUUID()}.${finalExtension}`;
    }

    requestedPath = validateGithubPath(requestedPath);
    const result = await uploadToGithub(requestedPath, buffer);
    const receiptUrl = `/receipt/${result.pathB64}`;

    return res.status(200).json({
      ok: true,
      ...result,
      mimeType: mimeType || mimeForPath(result.path),
      size: buffer.length,
      url: receiptUrl,
      receiptUrl,
      fileUrl: `${req.protocol}://${req.get('host')}${receiptUrl}`,
      absoluteUrl: `${req.protocol}://${req.get('host')}${receiptUrl}`,
    });
  } catch (error: unknown) {
    console.error('Upload error:', error);
    const message = error instanceof Error ? error.message : String(error);
    return res.status(500).json({
      error: 'No se pudo subir el archivo.',
      details: message,
    });
  }
});

/* =========================================================
   RECEIPT PRIVADO
   ========================================================= */

app.get('/receipt/:pathB64', async (req: Request, res: Response) => {
  try {
    const path = decodePathB64(getParamString(req.params.pathB64));
    if (!path.startsWith('receipts/')) {
      return res.status(403).send('ruta no permitida');
    }

    const { owner, repo, branch, token } = ghConfig();
    const url =
      `https://api.github.com/repos/${encodeURIComponent(owner)}` +
      `/${encodeURIComponent(repo)}/contents/${encodePathForGithub(path)}` +
      `?ref=${encodeURIComponent(branch)}`;

    const response = await fetch(url, {
      headers: { ...githubHeaders(token), Accept: 'application/vnd.github.raw' },
    });

    if (!response.ok) return res.status(response.status).send('not found');

    const buffer = Buffer.from(await response.arrayBuffer());
    res.setHeader('Content-Type', mimeForPath(path));
    res.setHeader('Cache-Control', 'private, max-age=3600');
    res.setHeader('Content-Length', String(buffer.length));
    return res.send(buffer);
  } catch (error) {
    console.error('Receipt error:', error);
    return res.status(500).send('error');
  }
});

/* =========================================================
   TAROT ORIGINAL /tarot/:n
   ========================================================= */

app.get('/tarot/:n', async (req: Request, res: Response) => {
  try {
    const raw = getParamString(req.params.n);
    if (!/^\d+$/.test(raw)) return res.status(400).send('n inválido');

    const number = Number.parseInt(raw, 10);
    const files = await listTarotFiles();
    const match = findTarotFile(number, files);
    if (!match) return res.status(404).send(`carta no encontrada: ${number}`);

    const original = await fetchTarotOriginal(number, match);
    res.setHeader('Content-Type', original.mimeType);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Length', String(original.buffer.length));
    res.setHeader('ETag', `"${match.sha}"`);
    return res.send(original.buffer);
  } catch (error) {
    console.error('Tarot original error:', error);
    return res.status(500).send('error');
  }
});

/* =========================================================
   TAROT RESPONSIVE /tarot/:w/:n
   ========================================================= */

app.get('/tarot/:w/:n', async (req: Request, res: Response) => {
  try {
    const requestedWidth = Number.parseInt(getParamString(req.params.w), 10);
    const number = Number.parseInt(getParamString(req.params.n), 10);

    if (
      !Number.isFinite(requestedWidth) || requestedWidth <= 0 ||
      !Number.isFinite(number) || number <= 0
    ) {
      return res.status(400).send('params inválidos');
    }

    const width = Math.min(requestedWidth, 4096);
    const files = await listTarotFiles();
    const match = findTarotFile(number, files);
    if (!match) return res.status(404).send(`carta no encontrada: ${number}`);

    const original = await fetchTarotOriginal(number, match);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Vary', 'Accept');
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    res.setHeader('ETag', `"${match.sha}-${width}"`);

    if (!original.width || width >= original.width) {
      res.setHeader('Content-Type', original.mimeType);
      res.setHeader('Content-Length', String(original.buffer.length));
      res.setHeader('X-Tarot-Original-Width', String(original.width || 0));
      res.setHeader('X-Tarot-Output-Width', String(original.width || 0));
      return res.send(original.buffer);
    }

    const image = await loadImage(original.buffer);
    const outputWidth = width;
    const outputHeight = Math.max(1, Math.round(image.height * (outputWidth / image.width)));
    const canvas = createCanvas(outputWidth, outputHeight);
    const context = canvas.getContext('2d');
    context.imageSmoothingEnabled = true;
    try { (context as any).imageSmoothingQuality = 'high'; } catch {}
    context.drawImage(image, 0, 0, outputWidth, outputHeight);

    const accept = String(req.headers.accept || '');
    if (accept.includes('image/webp')) {
      const encoded = await canvas.encode('webp', 92);
      res.setHeader('Content-Type', 'image/webp');
      res.setHeader('Content-Length', String(encoded.byteLength));
      res.setHeader('X-Tarot-Original-Width', String(original.width));
      res.setHeader('X-Tarot-Output-Width', String(outputWidth));
      return res.send(encoded);
    }

    const encoded = await canvas.encode('jpeg', 92);
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Content-Length', String(encoded.byteLength));
    res.setHeader('X-Tarot-Original-Width', String(original.width));
    res.setHeader('X-Tarot-Output-Width', String(outputWidth));
    return res.send(encoded);
  } catch (error) {
    console.error('Tarot resize error:', error);
    return res.status(500).send('error');
  }
});

/* =========================================================
   TAROT MANIFEST
   ========================================================= */

app.get('/tarot-manifest', async (_req: Request, res: Response) => {
  try {
    const { owner, repo, branch, token } = ghConfig();
    const files = await listTarotFiles(true);
    let commitSha: string | null = null;

    try {
      const response = await fetch(
        `https://api.github.com/repos/${encodeURIComponent(owner)}` +
        `/${encodeURIComponent(repo)}/branches/${encodeURIComponent(branch)}`,
        { headers: githubHeaders(token) }
      );
      if (response.ok) {
        const data = await response.json();
        commitSha = data?.commit?.sha || null;
      }
    } catch (error) {
      console.warn('Commit SHA error:', error);
    }

    const cards = files
      .map((file) => {
        const match = file.name.match(/juli[\s_-]*0*(\d+)\b/i);
        if (!match) return null;
        return {
          juli: Number.parseInt(match[1], 10),
          sha: file.sha,
          size: file.size,
          name: file.name,
        };
      })
      .filter(Boolean)
      .sort((a: any, b: any) => a.juli - b.juli);

    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.json({ ok: true, commitSha, branch, count: cards.length, cards });
  } catch (error: unknown) {
    console.error('Tarot manifest error:', error);
    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

/* =========================================================
   GEMINI PROJECT COPY
   ========================================================= */

app.post('/api/gemini/project-copy', async (req: Request, res: Response) => {
  try {
    if (apiKeys.length === 0) {
      return res.status(503).json({ error: 'El servicio Gemini no está configurado.' });
    }

    const body = getRequestBody(req);
    const { action, text, language, title, segments } = body;

    if (Object.keys(body).length === 0) {
      return res.status(400).json({
        error: 'La solicitud no contiene un body JSON válido.',
        expected: 'Para proyectos: {"title":"...","text":"..."}',
      });
    }

    if (action === 'translate') {
      if (!Array.isArray(segments)) {
        return res.status(400).json({ error: '"segments" debe ser un array.' });
      }
      if (segments.length === 0) {
        return res.status(400).json({ error: '"segments" no puede estar vacío.' });
      }

      const targetLanguage = language === 'en' ? 'inglés' : 'español';
      const prompt = `
Actúa como traductor profesional especializado en diseño gráfico y comunicación visual.
Traduce los siguientes textos al ${targetLanguage}.

Reglas:
- Mantén exactamente el significado.
- No inventes información.
- No elimines información.
- Conserva el orden de los textos.
- Mantén un tono profesional.
- La traducción debe sonar natural.
- No agregues explicaciones.

Textos:
${JSON.stringify(segments)}

Devuelve exclusivamente JSON válido con esta estructura:
{"translations":["texto traducido 1","texto traducido 2"]}
`;

      const interaction: any = await runGemini<any>((ai) =>
        ai.interactions.create({
          model: modelName,
          input: prompt,
          response_format: {
            type: 'text',
            mime_type: 'application/json',
            schema: {
              type: 'object',
              properties: {
                translations: { type: 'array', items: { type: 'string' } },
              },
              required: ['translations'],
            },
          },
        })
      );

      const output = interaction.output_text?.trim();
      if (!output) throw new Error('Gemini devolvió una respuesta vacía.');
      return res.status(200).json(parseGeminiJson(output));
    }

    if (typeof title !== 'string' || title.trim().length === 0) {
      return res.status(400).json({ error: 'Falta el campo "title".' });
    }
    if (typeof text !== 'string' || text.trim().length === 0) {
      return res.status(400).json({ error: 'Falta el campo "text".' });
    }

    const prompt = `
Actúa como director de arte y editor especializado en portfolios profesionales de diseño gráfico.

Proyecto:
${title}

Texto original:
${text}

Objetivos:
- Mejorar la claridad.
- Mejorar la redacción.
- Mantener la intención original.
- Utilizar lenguaje profesional.
- Evitar frases publicitarias genéricas.
- No inventar información.
- No agregar datos inexistentes.
- Mantener el contenido apropiado para un portfolio de diseño.

Devuelve exclusivamente JSON válido con esta estructura:
{
  "lead":"string",
  "discipline":"string",
  "sections":[{"title":"string","summary":"string"}],
  "imageAlts":["string"]
}
`;

    const interaction: any = await runGemini<any>((ai) =>
      ai.interactions.create({
        model: modelName,
        input: prompt,
        response_format: {
          type: 'text',
          mime_type: 'application/json',
          schema: {
            type: 'object',
            properties: {
              lead: { type: 'string' },
              discipline: { type: 'string' },
              sections: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    summary: { type: 'string' },
                  },
                  required: ['title', 'summary'],
                },
              },
              imageAlts: { type: 'array', items: { type: 'string' } },
            },
            required: ['lead', 'discipline', 'sections', 'imageAlts'],
          },
        },
      })
    );

    const output = interaction.output_text?.trim();
    if (!output) throw new Error('Gemini devolvió una respuesta vacía.');
    return res.status(200).json(parseGeminiJson(output));
  } catch (error: unknown) {
    console.error('Gemini project-copy error:', error);

    if (isRateLimitError(error)) {
      return res.status(429).json({
        error: 'Todas las API keys de Gemini están temporalmente limitadas.',
        retryable: true,
      });
    }

    const message = error instanceof Error ? error.message : String(error);
    return res.status(500).json({
      error: 'Error procesando la solicitud con Gemini.',
      details: message,
    });
  }
});

/* =========================================================
   GENERATE TEXT
   ========================================================= */

app.post('/generate-text', auth, async (req: Request, res: Response) => {
  try {
    if (apiKeys.length === 0) {
      return res.status(503).json({ error: 'El servicio Gemini no está configurado.' });
    }

    const body = getRequestBody(req);
    const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';
    if (!prompt) return res.status(400).json({ error: 'prompt requerido' });

    const response: any = await runGemini<any>((ai) =>
      ai.models.generateContent({
        model: modelName,
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        config: {
          temperature: Number.isFinite(Number(body.temperature))
            ? Math.max(0, Math.min(2, Number(body.temperature)))
            : 0.85,
          maxOutputTokens: Number.isFinite(Number(body.maxOutputTokens))
            ? Math.max(1, Math.min(8192, Number(body.maxOutputTokens)))
            : 2400,
          topP: 0.95,
        },
      })
    );

    const text = response.text?.trim() || '';
    if (!text) throw new Error('Gemini devolvió una respuesta vacía.');

    return res.status(200).json({ ok: true, text, model: modelName });
  } catch (error: unknown) {
    console.error('Generate text error:', error);
    if (isRateLimitError(error)) {
      return res.status(429).json({
        error: 'Todas las API keys de Gemini están temporalmente limitadas.',
        retryable: true,
      });
    }
    return res.status(500).json({
      error: 'Error generando texto.',
      details: error instanceof Error ? error.message : String(error),
    });
  }
});

/* =========================================================
   ANALYZE RECEIPT

   pdfMode:
   - auto   (default): intenta PNG; si falla, manda PDF original.
   - image: intenta PNG; si falla, manda PDF original.
   - native: manda el PDF original directamente.
   ========================================================= */

app.post('/analyze-receipt', auth, async (req: Request, res: Response) => {
  try {
    if (apiKeys.length === 0) {
      return res.status(503).json({ error: 'El servicio Gemini no está configurado.' });
    }

    const body = getRequestBody(req);
    const fileUrl = typeof body.fileUrl === 'string' ? body.fileUrl.trim() : '';
    const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';
    const requestedPdfMode = typeof body.pdfMode === 'string' ? body.pdfMode.toLowerCase() : 'auto';
    const pdfMode = ['auto', 'image', 'native'].includes(requestedPdfMode)
      ? requestedPdfMode
      : 'auto';

    if (!fileUrl) return res.status(400).json({ error: 'fileUrl es requerido.' });
    if (!prompt) return res.status(400).json({ error: 'prompt es requerido.' });

    const downloaded = await downloadReceipt(fileUrl);
    let dataBuffer = downloaded.buffer;
    let mimeType = downloaded.mimeType;
    let pdfProcessing: 'not-pdf' | 'native' | 'image' | 'native-fallback' = 'not-pdf';

    const isPdf =
      mimeType === 'application/pdf' ||
      fileUrl.toLowerCase().split('?')[0].endsWith('.pdf');

    console.log(
      `Receipt: archivo descargado (${downloaded.buffer.length} bytes, ${downloaded.mimeType}, pdfMode=${pdfMode})`
    );

    if (isPdf) {
      if (pdfMode === 'native') {
        pdfProcessing = 'native';
        console.log('Receipt PDF: envío nativo a Gemini.');
      } else {
        const png = await pdfFirstPageToPngSafe(downloaded.buffer);
        if (png) {
          dataBuffer = png;
          mimeType = 'image/png';
          pdfProcessing = 'image';
          console.log('Receipt PDF: usando primera página convertida a PNG.');
        } else {
          dataBuffer = downloaded.buffer;
          mimeType = 'application/pdf';
          pdfProcessing = 'native-fallback';
          console.warn('Receipt PDF: conversión falló; usando PDF original en Gemini.');
        }
      }
    }

    const base64 = dataBuffer.toString('base64');
    const geminiPrompt = `
${prompt}

IMPORTANTE:
- Analiza cuidadosamente el comprobante.
- Extrae únicamente información visible o claramente inferible.
- No inventes datos.
- Si un campo no puede determinarse, utiliza null.
- Devuelve ÚNICAMENTE JSON válido.
`;

    const response: any = await runGemini<any>((ai) =>
      ai.models.generateContent({
        model: modelName,
        contents: [
          {
            role: 'user',
            parts: [
              { text: geminiPrompt },
              { inlineData: { mimeType, data: base64 } },
            ],
          },
        ],
        config: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      })
    );

    const text = response.text?.trim() || '';
    if (!text) throw new Error('Gemini devolvió una respuesta vacía.');

    const parsed = parseGeminiJson(text);
    return res.status(200).json({
      ok: true,
      detected: parsed,
      raw: text,
      inputMimeType: downloaded.mimeType,
      sentMimeType: mimeType,
      pdfMode,
      pdfProcessing,
    });
  } catch (error: unknown) {
    console.error('Analyze receipt error:', error);

    if (isRateLimitError(error)) {
      return res.status(429).json({
        error: 'Todas las API keys de Gemini están temporalmente limitadas.',
        retryable: true,
      });
    }

    const message = error instanceof Error ? error.message : String(error);
    return res.status(500).json({
      error: 'Error analizando el comprobante.',
      details: message,
    });
  }
});

/* =========================================================
   404 / ERROR HANDLER
   ========================================================= */

app.use((_req: Request, res: Response) => {
  return res.status(404).json({ error: 'Endpoint no encontrado.' });
});

app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
  console.error('Unhandled server error:', error);
  return res.status(500).json({
    error: 'Error interno del servidor.',
    details: error instanceof Error ? error.message : String(error),
  });
});

/* =========================================================
   START
   ========================================================= */

app.listen(PORT, '0.0.0.0', () => {
  console.log('======================================');
  console.log('Torchill API iniciada');
  console.log(`Puerto: ${PORT}`);
  console.log(`Gemini configurado: ${apiKeys.length > 0}`);
  console.log(`Gemini API keys disponibles: ${apiKeys.length}`);
  console.log(`Modelo: ${modelName}`);
  console.log(
    `GitHub configurado: ${Boolean(
      process.env.GITHUB_OWNER && process.env.GITHUB_REPO && process.env.GITHUB_TOKEN
    )}`
  );
  console.log('PDF modes: auto | image | native');
  console.log('--------------------------------------');
  console.log('GET  /health');
  console.log('GET  /api/gemini/project-copy');
  console.log('POST /api/gemini/project-copy');
  console.log('POST /upload');
  console.log('GET  /receipt/:pathB64');
  console.log('GET  /tarot/:n');
  console.log('GET  /tarot/:w/:n');
  console.log('GET  /tarot-manifest');
  console.log('POST /analyze-receipt');
  console.log('POST /generate-text');
  console.log('======================================');
});
