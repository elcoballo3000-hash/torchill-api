import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import multer from 'multer';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import {
  S3Client,
  PutObjectCommand,
  HeadObjectCommand,
  GetObjectCommand,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

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
app.use(express.json({
  limit: '60mb',
  verify: (req: Request, _res: Response, buffer: Buffer) => {
    (req as any).rawBody = Buffer.from(buffer);
  },
}));
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


interface AppClientRateLimitEntry {
  count: number;
  resetAt: number;
}

const appClientRateLimit = new Map<string, AppClientRateLimitEntry>();
const APP_CLIENT_RATE_LIMIT_PER_MINUTE = 40;

function getAppClientOrigin(req: Request): string {
  return String(req.headers.origin || '').trim().replace(/\/+$/, '');
}

function getAppClientIp(req: Request): string {
  const forwarded = String(req.headers['x-forwarded-for'] || '')
    .split(',')[0]
    .trim();

  return forwarded || req.ip || req.socket.remoteAddress || 'unknown';
}

function consumeAppClientRateLimit(req: Request): boolean {
  const ip = getAppClientIp(req);
  const now = Date.now();
  const current = appClientRateLimit.get(ip);

  if (!current || current.resetAt <= now) {
    appClientRateLimit.set(ip, {
      count: 1,
      resetAt: now + 60_000,
    });
    return true;
  }

  current.count += 1;

  if (appClientRateLimit.size > 5000) {
    for (const [key, value] of appClientRateLimit.entries()) {
      if (value.resetAt <= now) appClientRateLimit.delete(key);
    }
  }

  return current.count <= APP_CLIENT_RATE_LIMIT_PER_MINUTE;
}

function appClientOrAuth(req: Request, res: Response, next: NextFunction) {
  const sent =
    req.headers['x-api-key'] ||
    (req.headers.authorization || '').replace(/^Bearer\s+/i, '');

  if (API_TOKEN && sent === API_TOKEN) {
    return next();
  }

  const origin = getAppClientOrigin(req);

  if (!origin || !isTarotClientOriginAllowed(origin)) {
    return res.status(401).json({
      error: 'no autorizado',
      details: 'Origen cliente no permitido.',
    });
  }

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Vary', 'Origin');

  if (!consumeAppClientRateLimit(req)) {
    return res.status(429).json({
      error: 'Demasiadas solicitudes. Reintentá en unos segundos.',
    });
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
   CLOUDFLARE R2 PRIVADO — STAR TAROT
   ========================================================= */

const TAROT_R2_WIDTHS = [320, 640, 960, 1280] as const;
const TAROT_SIGNED_URL_TTL_SECONDS = 10 * 60;
const MAX_SIGNED_CARDS_PER_REQUEST = 24;

// Endpoint público-controlado para el frontend de Base44.
// No usa API_TOKEN en el navegador.
const TAROT_CLIENT_SIGNED_URL_TTL_SECONDS = 5 * 60;
const TAROT_CLIENT_MAX_CARDS_PER_REQUEST = 12;
const TAROT_CLIENT_RATE_LIMIT_PER_MINUTE = 30;
const TAROT_CLIENT_MIN_CARD = 3;
const TAROT_CLIENT_MAX_CARD = 80;

interface ClientRateLimitEntry {
  count: number;
  resetAt: number;
}

const tarotClientRateLimit = new Map<string, ClientRateLimitEntry>();

function getTarotClientAllowedOrigins(): string[] {
  return String(process.env.TAROT_CLIENT_ORIGINS || '')
    .split(',')
    .map((value) => value.trim().replace(/\/+$/, ''))
    .filter(Boolean);
}

function isTarotClientOriginAllowed(origin: string): boolean {
  const normalized = origin.trim().replace(/\/+$/, '');
  if (!normalized) return false;

  const allowedOrigins = getTarotClientAllowedOrigins();

  // Por seguridad el endpoint queda deshabilitado hasta configurar
  // TAROT_CLIENT_ORIGINS en Render.
  if (allowedOrigins.length === 0) return false;

  return allowedOrigins.includes(normalized);
}

function getTarotClientIp(req: Request): string {
  const forwarded = String(req.headers['x-forwarded-for'] || '')
    .split(',')[0]
    .trim();

  return forwarded || req.ip || req.socket.remoteAddress || 'unknown';
}

function consumeTarotClientRateLimit(req: Request): {
  allowed: boolean;
  remaining: number;
  resetAt: number;
} {
  const ip = getTarotClientIp(req);
  const now = Date.now();
  const windowMs = 60_000;

  const current = tarotClientRateLimit.get(ip);

  if (!current || current.resetAt <= now) {
    const entry = {
      count: 1,
      resetAt: now + windowMs,
    };
    tarotClientRateLimit.set(ip, entry);

    return {
      allowed: true,
      remaining: TAROT_CLIENT_RATE_LIMIT_PER_MINUTE - 1,
      resetAt: entry.resetAt,
    };
  }

  current.count += 1;

  // Limpieza liviana para que el Map no crezca indefinidamente.
  if (tarotClientRateLimit.size > 5000) {
    for (const [key, value] of tarotClientRateLimit.entries()) {
      if (value.resetAt <= now) tarotClientRateLimit.delete(key);
    }
  }

  return {
    allowed: current.count <= TAROT_CLIENT_RATE_LIMIT_PER_MINUTE,
    remaining: Math.max(
      0,
      TAROT_CLIENT_RATE_LIMIT_PER_MINUTE - current.count
    ),
    resetAt: current.resetAt,
  };
}

interface R2Config {
  accountId: string;
  accessKeyId: string;
  secretAccessKey: string;
  bucket: string;
}

let cachedR2Client: S3Client | null = null;
let cachedR2ClientSignature = '';

function r2Config(): R2Config {
  const accountId = process.env.R2_ACCOUNT_ID?.trim() || '';
  const accessKeyId = process.env.R2_ACCESS_KEY_ID?.trim() || '';
  const secretAccessKey = process.env.R2_SECRET_ACCESS_KEY?.trim() || '';
  const bucket = process.env.R2_BUCKET?.trim() || '';

  if (!accountId || !accessKeyId || !secretAccessKey || !bucket) {
    throw new Error(
      'R2 no está configurado. Revisá R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY y R2_BUCKET.'
    );
  }

  return { accountId, accessKeyId, secretAccessKey, bucket };
}

function getR2Client(): S3Client {
  const cfg = r2Config();
  const signature = `${cfg.accountId}:${cfg.accessKeyId}`;

  if (cachedR2Client && cachedR2ClientSignature === signature) {
    return cachedR2Client;
  }

  cachedR2Client = new S3Client({
    region: 'auto',
    endpoint: `https://${cfg.accountId}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: cfg.accessKeyId,
      secretAccessKey: cfg.secretAccessKey,
    },
  });
  cachedR2ClientSignature = signature;

  return cachedR2Client;
}

function tarotR2Key(number: number, width: number | 'original'): string {
  const filename = `juli_${String(number).padStart(2, '0')}.webp`;

  if (width === 'original') {
    return `tarot/original/${filename}`;
  }

  return `tarot/${width}/${filename}`;
}

function tarotBackR2Key(width: number | 'original'): string {
  if (width === 'original') return 'tarot/original/dorso.webp';
  return `tarot/${width}/dorso.webp`;
}

async function r2HasSameSourceSha(key: string, sourceSha: string): Promise<boolean> {
  const { bucket } = r2Config();

  try {
    const result = await getR2Client().send(
      new HeadObjectCommand({
        Bucket: bucket,
        Key: key,
      })
    );

    return String(result.Metadata?.source_sha || '') === sourceSha;
  } catch (error: any) {
    const status = error?.$metadata?.httpStatusCode;
    if (
      status === 404 ||
      error?.name === 'NotFound' ||
      error?.name === 'NoSuchKey'
    ) {
      return false;
    }

    throw error;
  }
}

async function encodeTarotVariants(
  source: Buffer
): Promise<Array<{ width: number | 'original'; buffer: Buffer }>> {
  const image = await loadImage(source);
  const variants: Array<{ width: number | 'original'; buffer: Buffer }> = [];

  const renderWebp = async (requestedWidth: number, quality: number): Promise<Buffer> => {
    const outputWidth = Math.max(1, Math.min(requestedWidth, image.width));
    const outputHeight = Math.max(
      1,
      Math.round(image.height * (outputWidth / image.width))
    );

    const canvas = createCanvas(outputWidth, outputHeight);
    const context = canvas.getContext('2d');
    context.imageSmoothingEnabled = true;
    try {
      (context as any).imageSmoothingQuality = 'high';
    } catch {}

    context.drawImage(image, 0, 0, outputWidth, outputHeight);
    return Buffer.from(await canvas.encode('webp', quality));
  };

  for (const width of TAROT_R2_WIDTHS) {
    variants.push({
      width,
      buffer: await renderWebp(width, 88),
    });
  }

  const originalCanvas = createCanvas(image.width, image.height);
  const originalContext = originalCanvas.getContext('2d');
  originalContext.drawImage(image, 0, 0, image.width, image.height);

  variants.push({
    width: 'original',
    buffer: Buffer.from(await originalCanvas.encode('webp', 92)),
  });

  return variants;
}

async function putTarotVariantToR2(
  number: number,
  width: number | 'original',
  buffer: Buffer,
  sourceSha: string
): Promise<string> {
  const { bucket } = r2Config();
  const key = tarotR2Key(number, width);

  await getR2Client().send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: buffer,
      ContentType: 'image/webp',
      CacheControl: 'private, max-age=600',
      Metadata: {
        source_sha: sourceSha,
        juli: String(number),
        width: String(width),
      },
    })
  );

  return key;
}

async function putTarotBackVariantToR2(
  width: number | 'original',
  buffer: Buffer,
  sourceSha: string
): Promise<string> {
  const { bucket } = r2Config();
  const key = tarotBackR2Key(width);

  await getR2Client().send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: buffer,
      ContentType: 'image/webp',
      CacheControl: 'private, max-age=600',
      Metadata: {
        source_sha: sourceSha,
        asset: 'dorso',
        width: String(width),
      },
    })
  );

  return key;
}

async function syncTarotBackToR2(
  files: TarotFile[],
  force = false
) {
  const file = findTarotBackFile(files);

  if (!file) {
    return {
      asset: 'dorso',
      ok: false,
      skipped: false,
      error: 'No se encontró dorso.jpg/jpeg/png/webp dentro de GitHub /tarot.',
    };
  }

  const probeKey = tarotBackR2Key(640);

  if (!force && await r2HasSameSourceSha(probeKey, file.sha)) {
    return {
      asset: 'dorso',
      ok: true,
      skipped: true,
      sourceName: file.name,
      sourceSha: file.sha,
    };
  }

  const source = await fetchTarotFileBuffer(file);
  const variants = await encodeTarotVariants(source);
  const uploaded = [];

  for (const variant of variants) {
    const key = await putTarotBackVariantToR2(
      variant.width,
      variant.buffer,
      file.sha
    );

    uploaded.push({
      width: variant.width,
      key,
      bytes: variant.buffer.length,
    });
  }

  return {
    asset: 'dorso',
    ok: true,
    skipped: false,
    sourceName: file.name,
    sourceSha: file.sha,
    variants: uploaded,
  };
}

async function syncTarotCardToR2(
  number: number,
  files: TarotFile[],
  force = false
) {
  const file = findTarotFile(number, files);

  if (!file) {
    return {
      juli: number,
      ok: false,
      skipped: false,
      error: 'Carta no encontrada dentro de GitHub /tarot.',
    };
  }

  // Con revisar 640 alcanza: todas las variantes se escriben juntas.
  const probeKey = tarotR2Key(number, 640);

  if (!force && await r2HasSameSourceSha(probeKey, file.sha)) {
    return {
      juli: number,
      ok: true,
      skipped: true,
      sourceName: file.name,
      sourceSha: file.sha,
    };
  }

  const original = await fetchTarotOriginal(number, file);
  const variants = await encodeTarotVariants(original.buffer);

  const uploaded = [];

  for (const variant of variants) {
    const key = await putTarotVariantToR2(
      number,
      variant.width,
      variant.buffer,
      file.sha
    );

    uploaded.push({
      width: variant.width,
      key,
      bytes: variant.buffer.length,
    });
  }

  return {
    juli: number,
    ok: true,
    skipped: false,
    sourceName: file.name,
    sourceSha: file.sha,
    variants: uploaded,
  };
}

async function syncTarotNumbersToR2(
  numbers: number[],
  force = false
) {
  const uniqueNumbers = [...new Set(numbers)]
    .filter((value) => Number.isInteger(value) && value > 0)
    .sort((a, b) => a - b);

  const files = await listTarotFiles(true);
  const results: any[] = [];

  // Intencionalmente secuencial: evita picos fuertes de RAM/CPU en Render.
  for (const number of uniqueNumbers) {
    try {
      results.push(await syncTarotCardToR2(number, files, force));
    } catch (error: unknown) {
      results.push({
        juli: number,
        ok: false,
        skipped: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return results;
}

async function createTarotSignedUrl(
  number: number,
  width: number | 'original',
  expiresIn: number
): Promise<string> {
  const { bucket } = r2Config();

  return getSignedUrl(
    getR2Client(),
    new GetObjectCommand({
      Bucket: bucket,
      Key: tarotR2Key(number, width),
    }),
    { expiresIn }
  );
}

async function createTarotBackSignedUrl(
  width: number | 'original',
  expiresIn: number
): Promise<string> {
  const { bucket } = r2Config();

  return getSignedUrl(
    getR2Client(),
    new GetObjectCommand({
      Bucket: bucket,
      Key: tarotBackR2Key(width),
    }),
    { expiresIn }
  );
}

function verifyGithubWebhookSignature(req: Request): boolean {
  const secret = process.env.GITHUB_WEBHOOK_SECRET?.trim() || '';
  if (!secret) return false;

  const receivedSignature = String(
    req.headers['x-hub-signature-256'] || ''
  );

  if (!receivedSignature.startsWith('sha256=')) return false;

  const rawBody = (req as any).rawBody as Buffer | undefined;
  if (!rawBody) return false;

  const expectedSignature =
    'sha256=' +
    crypto
      .createHmac('sha256', secret)
      .update(rawBody)
      .digest('hex');

  const receivedBuffer = Buffer.from(receivedSignature);
  const expectedBuffer = Buffer.from(expectedSignature);

  if (receivedBuffer.length !== expectedBuffer.length) return false;

  return crypto.timingSafeEqual(receivedBuffer, expectedBuffer);
}

function githubPushChangedTarotBack(body: any): boolean {
  const commits = Array.isArray(body?.commits) ? body.commits : [];

  for (const commit of commits) {
    for (const group of ['added', 'modified', 'removed']) {
      const paths = Array.isArray(commit?.[group]) ? commit[group] : [];

      for (const path of paths) {
        if (typeof path !== 'string') continue;
        if (!/^tarot\//i.test(path)) continue;

        const filename = path.split('/').pop() || '';
        if (isTarotBackFilename(filename)) return true;
      }
    }
  }

  return false;
}

function extractTarotNumbersFromGithubPush(body: any): number[] {
  const changedPaths = new Set<string>();

  const commits = Array.isArray(body?.commits) ? body.commits : [];

  for (const commit of commits) {
    for (const group of ['added', 'modified', 'removed']) {
      const paths = Array.isArray(commit?.[group]) ? commit[group] : [];

      for (const path of paths) {
        if (typeof path === 'string') changedPaths.add(path);
      }
    }
  }

  const numbers = new Set<number>();

  for (const path of changedPaths) {
    // MUY IMPORTANTE:
    // receipts/* y cualquier otra carpeta del mismo repo quedan ignoradas.
    if (!/^tarot\//i.test(path)) continue;

    const filename = path.split('/').pop() || '';
    const number = getTarotNumberFromFilename(filename);
    if (number !== null) numbers.add(number);
  }

  return [...numbers].sort((a, b) => a - b);
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

function getTarotNumberFromFilename(filename: string): number | null {
  // Formato real actual:
  // "tarot juli-1.jpg"
  // "tarot juli-10.jpg"
  // "tarot juli-78.webp"
  //
  // También tolera espacios, "_" y ceros iniciales:
  // "tarot_juli_01.jpg", "juli 001.png", etc.
  const match = filename.match(
    /(?:^|[\s_-])juli[\s_-]*0*(\d+)(?=\D|$)/i
  );

  if (!match) return null;

  const number = Number.parseInt(match[1], 10);
  return Number.isInteger(number) && number > 0 ? number : null;
}

function findTarotFile(number: number, files: TarotFile[]): TarotFile | null {
  return (
    files.find((file) => getTarotNumberFromFilename(file.name) === number) ||
    null
  );
}

function isTarotBackFilename(filename: string): boolean {
  return /^dorso\.(jpg|jpeg|png|webp)$/i.test(filename.trim());
}

function findTarotBackFile(files: TarotFile[]): TarotFile | null {
  return files.find((file) => isTarotBackFilename(file.name)) || null;
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

async function fetchTarotFileBuffer(file: TarotFile): Promise<Buffer> {
  const { owner, repo, branch, token } = ghConfig();

  const url =
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/tarot/${encodeURIComponent(file.name)}` +
    `?ref=${encodeURIComponent(branch)}`;

  const response = await fetch(url, {
    headers: {
      ...githubHeaders(token),
      Accept: 'application/vnd.github.raw',
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`No se pudo descargar ${file.name}: ${body}`);
  }

  return Buffer.from(await response.arrayBuffer());
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
    r2Configured: Boolean(
      process.env.R2_ACCOUNT_ID &&
      process.env.R2_ACCESS_KEY_ID &&
      process.env.R2_SECRET_ACCESS_KEY &&
      process.env.R2_BUCKET
    ),
    r2Bucket: process.env.R2_BUCKET || null,
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

app.post('/upload', appClientOrAuth, upload.single('file'), async (req: Request, res: Response) => {
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

app.get('/tarot/:n', auth, async (req: Request, res: Response) => {
  try {
    const raw = getParamString(req.params.n);
    if (!/^\d+$/.test(raw)) return res.status(400).send('n inválido');

    const number = Number.parseInt(raw, 10);
    const files = await listTarotFiles();
    const match = findTarotFile(number, files);
    if (!match) return res.status(404).send(`carta no encontrada: ${number}`);

    const original = await fetchTarotOriginal(number, match);
    res.setHeader('Content-Type', original.mimeType);
    res.setHeader('Cache-Control', 'private, no-store');
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

app.get('/tarot/:w/:n', auth, async (req: Request, res: Response) => {
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
    res.setHeader('Cache-Control', 'private, no-store');
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

app.get('/tarot-manifest', auth, async (_req: Request, res: Response) => {
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
        const number = getTarotNumberFromFilename(file.name);
        if (number === null) return null;
        return {
          juli: number,
          sha: file.sha,
          size: file.size,
          name: file.name,
        };
      })
      .filter(Boolean)
      .sort((a: any, b: any) => a.juli - b.juli);

    res.setHeader('Cache-Control', 'private, no-store');
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
   TAROT PRIVADO EN CLOUDFLARE R2
   ========================================================= */

// Primera sincronización:
// POST /api/tarot/sync
//
// Solo algunas cartas:
// POST /api/tarot/sync?cards=1,2,17
//
// Forzar regeneración:
// POST /api/tarot/sync?cards=17&force=1
app.post('/api/tarot/sync', auth, async (req: Request, res: Response) => {
  try {
    r2Config();

    const files = await listTarotFiles(true);
    const body = getRequestBody(req);

    const cardsValue =
      typeof req.query.cards === 'string'
        ? req.query.cards
        : typeof body.cards === 'string'
          ? body.cards
          : '';

    const numbers = cardsValue
      ? cardsValue
          .split(',')
          .map((value: string) => Number.parseInt(value.trim(), 10))
          .filter((value: number) => Number.isInteger(value) && value > 0)
      : files
          .map((file) => {
            const number = getTarotNumberFromFilename(file.name);
            return number ?? NaN;
          })
          .filter((value) => Number.isInteger(value) && value > 0);

    const force =
      String(req.query.force || '') === '1' ||
      body.force === true ||
      String(body.force || '') === '1';

    const backRequested =
      String(req.query.back || '') === '1' ||
      body.back === true ||
      String(body.back || '') === '1';

    const results = await syncTarotNumbersToR2(numbers, force);

    let backResult: any = null;

    if (backRequested) {
      try {
        backResult = await syncTarotBackToR2(files, force);
      } catch (error: unknown) {
        backResult = {
          asset: 'dorso',
          ok: false,
          skipped: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }

    const synced = results.filter(
      (item) => item.ok && !item.skipped
    ).length;

    const skipped = results.filter(
      (item) => item.ok && item.skipped
    ).length;

    const failed = results.filter(
      (item) => !item.ok
    ).length;

    const backFailed = backResult && !backResult.ok ? 1 : 0;

    return res.status(failed + backFailed > 0 ? 207 : 200).json({
      ok: failed === 0 && backFailed === 0,
      requested: [...new Set(numbers)].length,
      synced,
      skipped,
      failed,
      back: backResult,
      results,
    });
  } catch (error: unknown) {
    console.error('Tarot R2 sync error:', error);

    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

// Base44 NO debe tener credenciales R2.
// Su backend pide URLs temporales a este endpoint.
app.get('/api/tarot/urls', auth, async (req: Request, res: Response) => {
  try {
    r2Config();

    const requestedCards = String(req.query.cards || '')
      .split(',')
      .map((value) => Number.parseInt(value.trim(), 10))
      .filter((value) => Number.isInteger(value) && value > 0);

    const cards = [...new Set(requestedCards)];

    if (cards.length === 0) {
      return res.status(400).json({
        ok: false,
        error: 'cards es requerido. Ejemplo: ?cards=1,2,3&widths=320,640',
      });
    }

    if (cards.length > MAX_SIGNED_CARDS_PER_REQUEST) {
      return res.status(400).json({
        ok: false,
        error: `Máximo ${MAX_SIGNED_CARDS_PER_REQUEST} cartas por solicitud.`,
      });
    }

    const widthTokens = String(req.query.widths || '320,640')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean);

    const widths: Array<number | 'original'> = [];

    for (const token of widthTokens) {
      if (token === 'original') {
        widths.push('original');
        continue;
      }

      const width = Number.parseInt(token, 10);

      if ((TAROT_R2_WIDTHS as readonly number[]).includes(width)) {
        widths.push(width);
      }
    }

    const uniqueWidths = [...new Set(widths)];

    if (uniqueWidths.length === 0) {
      return res.status(400).json({
        ok: false,
        error: 'widths inválido. Usá 320,640,960,1280,original.',
      });
    }

    const requestedExpiry = Number.parseInt(
      String(req.query.expires || ''),
      10
    );

    const expiresIn = Number.isFinite(requestedExpiry)
      ? Math.max(60, Math.min(900, requestedExpiry))
      : TAROT_SIGNED_URL_TTL_SECONDS;

    const result: Record<string, Record<string, string>> = {};

    for (const card of cards) {
      result[String(card)] = {};

      for (const width of uniqueWidths) {
        result[String(card)][String(width)] =
          await createTarotSignedUrl(card, width, expiresIn);
      }
    }

    res.setHeader('Cache-Control', 'private, no-store');

    return res.status(200).json({
      ok: true,
      expiresIn,
      expiresAt: new Date(
        Date.now() + expiresIn * 1000
      ).toISOString(),
      cards: result,
    });
  } catch (error: unknown) {
    console.error('Tarot signed URLs error:', error);

    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});


// ================================================================
// FRONTEND BASE44 SIN BACKEND FUNCTIONS / SIN INTEGRATION CREDITS
//
// Este endpoint NO recibe API_TOKEN desde el navegador.
// Seguridad aplicada:
// - Origin permitido configurado en TAROT_CLIENT_ORIGINS.
// - Máximo 12 cartas por request.
// - Solamente cartas 3..80.
// - Solamente 320/640/960/1280 (nunca "original").
// - URLs R2 firmadas por 5 minutos.
// - Rate limit por IP.
// - Solo lectura; no lista/escribe/borra objetos.
//
// IMPORTANTE:
// CORS ayuda a limitar el uso desde navegadores, pero no convierte
// este endpoint en autenticación fuerte. Una persona técnicamente
// avanzada podría imitar un Origin fuera del navegador. Por eso
// también existen límites, expiración corta y un alcance mínimo.
// ================================================================
app.get('/api/tarot/client-urls', async (req: Request, res: Response) => {
  try {
    r2Config();

    const origin = String(req.headers.origin || '').trim();
    const allowedOrigins = getTarotClientAllowedOrigins();

    if (allowedOrigins.length === 0) {
      return res.status(503).json({
        ok: false,
        error:
          'TAROT_CLIENT_ORIGINS no está configurado en Render. ' +
          'Agregá la URL exacta de tu app Base44.',
      });
    }

    if (!origin || !isTarotClientOriginAllowed(origin)) {
      return res.status(403).json({
        ok: false,
        error: 'Origen no permitido.',
      });
    }

    // Sobrescribe el CORS global "*" específicamente para esta ruta.
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Methods', 'GET');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    const limit = consumeTarotClientRateLimit(req);
    res.setHeader(
      'X-RateLimit-Limit',
      String(TAROT_CLIENT_RATE_LIMIT_PER_MINUTE)
    );
    res.setHeader('X-RateLimit-Remaining', String(limit.remaining));
    res.setHeader(
      'X-RateLimit-Reset',
      String(Math.ceil(limit.resetAt / 1000))
    );

    if (!limit.allowed) {
      res.setHeader(
        'Retry-After',
        String(Math.max(1, Math.ceil((limit.resetAt - Date.now()) / 1000)))
      );

      return res.status(429).json({
        ok: false,
        error: 'Demasiadas solicitudes. Reintentá en unos segundos.',
      });
    }

    const requestedCards = String(req.query.cards || '')
      .split(',')
      .map((value: string) => Number.parseInt(value.trim(), 10))
      .filter(
        (value: number) =>
          Number.isInteger(value) &&
          value >= TAROT_CLIENT_MIN_CARD &&
          value <= TAROT_CLIENT_MAX_CARD
      );

    const cards = [...new Set(requestedCards)];

    const includeBack =
      String(req.query.back || '') === '1' ||
      String(req.query.includeBack || '') === '1';

    if (cards.length === 0 && !includeBack) {
      return res.status(400).json({
        ok: false,
        error:
          'cards es requerido salvo que uses back=1. ' +
          'Ejemplo: ?cards=10&widths=640 o ?back=1&widths=640',
      });
    }

    if (cards.length > TAROT_CLIENT_MAX_CARDS_PER_REQUEST) {
      return res.status(400).json({
        ok: false,
        error:
          `Máximo ${TAROT_CLIENT_MAX_CARDS_PER_REQUEST} cartas ` +
          'por solicitud desde el frontend.',
      });
    }

    const widthTokens = String(req.query.widths || '320,640')
      .split(',')
      .map((value: string) => value.trim())
      .filter(Boolean);

    const widths: number[] = [];

    for (const token of widthTokens) {
      // "original" queda expresamente prohibido en el endpoint cliente.
      if (token.toLowerCase() === 'original') continue;

      const width = Number.parseInt(token, 10);

      if ((TAROT_R2_WIDTHS as readonly number[]).includes(width)) {
        widths.push(width);
      }
    }

    const uniqueWidths = [...new Set(widths)];

    if (uniqueWidths.length === 0) {
      return res.status(400).json({
        ok: false,
        error:
          'widths inválido. Desde el frontend solo se permiten ' +
          '320,640,960,1280.',
      });
    }

    const result: Record<string, Record<string, string>> = {};

    for (const card of cards) {
      result[String(card)] = {};

      for (const width of uniqueWidths) {
        result[String(card)][String(width)] =
          await createTarotSignedUrl(
            card,
            width,
            TAROT_CLIENT_SIGNED_URL_TTL_SECONDS
          );
      }
    }

    res.setHeader('Cache-Control', 'private, no-store');

    const back: Record<string, string> = {};

    if (includeBack) {
      for (const width of uniqueWidths) {
        back[String(width)] = await createTarotBackSignedUrl(
          width,
          TAROT_CLIENT_SIGNED_URL_TTL_SECONDS
        );
      }
    }

    return res.status(200).json({
      ok: true,
      expiresIn: TAROT_CLIENT_SIGNED_URL_TTL_SECONDS,
      expiresAt: new Date(
        Date.now() + TAROT_CLIENT_SIGNED_URL_TTL_SECONDS * 1000
      ).toISOString(),
      cards: result,
      back: includeBack ? back : undefined,
    });
  } catch (error: unknown) {
    console.error('Tarot client signed URLs error:', error);

    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.get('/api/tarot/manifest', auth, async (_req: Request, res: Response) => {
  try {
    const { branch } = ghConfig();
    r2Config();

    const files = await listTarotFiles(true);

    const cards = files
      .map((file) => {
        const number = getTarotNumberFromFilename(file.name);
        if (number === null) return null;

        return {
          juli: number,
          name: file.name,
          sha: file.sha,
          size: file.size,
        };
      })
      .filter(Boolean)
      .sort((a: any, b: any) => a.juli - b.juli);

    res.setHeader('Cache-Control', 'private, no-store');

    return res.status(200).json({
      ok: true,
      branch,
      sourceRepo: 'torchill-receipts',
      sourceFolder: 'tarot',
      count: cards.length,
      variants: [...TAROT_R2_WIDTHS, 'original'],
      cards,
    });
  } catch (error: unknown) {
    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

// GitHub webhook.
// Los pushes que solo cambian receipts/* son ignorados.
app.post('/api/github/webhook', async (req: Request, res: Response) => {
  try {
    if (!verifyGithubWebhookSignature(req)) {
      return res.status(401).json({
        ok: false,
        error: 'Firma de GitHub inválida.',
      });
    }

    const githubEvent = String(
      req.headers['x-github-event'] || ''
    );

    if (githubEvent === 'ping') {
      return res.status(200).json({
        ok: true,
        event: 'ping',
      });
    }

    if (githubEvent !== 'push') {
      return res.status(200).json({
        ok: true,
        ignored: true,
        event: githubEvent,
      });
    }

    const branch = process.env.GITHUB_BRANCH || 'main';
    const pushedRef = String((req.body as any)?.ref || '');

    if (
      pushedRef &&
      pushedRef !== `refs/heads/${branch}`
    ) {
      return res.status(200).json({
        ok: true,
        ignored: true,
        reason: `El push no corresponde a la rama ${branch}.`,
      });
    }

    const cards = extractTarotNumbersFromGithubPush(req.body);
    const backChanged = githubPushChangedTarotBack(req.body);

    if (cards.length === 0 && !backChanged) {
      return res.status(200).json({
        ok: true,
        ignored: true,
        reason:
          'El push no modificó cartas juli_XX ni dorso dentro de /tarot. ' +
          'receipts/ no dispara sincronización.',
      });
    }

    const results =
      cards.length > 0
        ? await syncTarotNumbersToR2(cards, true)
        : [];

    let backResult: any = null;

    if (backChanged) {
      try {
        const files = await listTarotFiles(true);
        backResult = await syncTarotBackToR2(files, true);
      } catch (error: unknown) {
        backResult = {
          asset: 'dorso',
          ok: false,
          skipped: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }

    const failed = results.filter((item) => !item.ok).length;
    const backFailed = backResult && !backResult.ok ? 1 : 0;

    return res.status(failed + backFailed > 0 ? 207 : 200).json({
      ok: failed === 0 && backFailed === 0,
      event: 'push',
      cards,
      backChanged,
      back: backResult,
      failed,
      results,
    });
  } catch (error: unknown) {
    console.error('GitHub webhook error:', error);

    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

/* =========================================================
   GEMINI PROJECT COPY
   ========================================================= */

app.post('/api/gemini/project-copy', appClientOrAuth, async (req: Request, res: Response) => {
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

app.post('/generate-text', appClientOrAuth, async (req: Request, res: Response) => {
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

app.post('/analyze-receipt', appClientOrAuth, async (req: Request, res: Response) => {
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
  console.log(
    `R2 configurado: ${Boolean(
      process.env.R2_ACCOUNT_ID &&
      process.env.R2_ACCESS_KEY_ID &&
      process.env.R2_SECRET_ACCESS_KEY &&
      process.env.R2_BUCKET
    )}`
  );
  console.log(`R2 bucket: ${process.env.R2_BUCKET || 'NO CONFIGURADO'}`);
  console.log(
    `Tarot client origins: ${
      getTarotClientAllowedOrigins().length > 0
        ? getTarotClientAllowedOrigins().join(', ')
        : 'NO CONFIGURADO'
    }`
  );
  console.log('Tarot filename pattern: tarot juli-N.ext');
  console.log('PDF modes: auto | image | native');
  console.log('--------------------------------------');
  console.log('GET  /health');
  console.log('GET  /api/gemini/project-copy');
  console.log('POST /api/gemini/project-copy [BASE44 CLIENT OR API_TOKEN]');
  console.log('POST /upload [BASE44 CLIENT OR API_TOKEN]');
  console.log('GET  /receipt/:pathB64');
  console.log('GET  /tarot/:n');
  console.log('GET  /tarot/:w/:n');
  console.log('GET  /tarot-manifest [PRIVATE]');
  console.log('GET  /api/tarot/manifest [PRIVATE]');
  console.log('GET  /api/tarot/urls [PRIVATE]');
  console.log('GET  /api/tarot/client-urls [BASE44 CLIENT]');
  console.log('Tarot dorso support: enabled (back=1)');
  console.log('POST /api/tarot/sync [PRIVATE]');
  console.log('POST /api/github/webhook [SIGNED GITHUB]');
  console.log('POST /analyze-receipt [BASE44 CLIENT OR API_TOKEN]');
  console.log('POST /generate-text [BASE44 CLIENT OR API_TOKEN]');
  console.log('======================================');
});
