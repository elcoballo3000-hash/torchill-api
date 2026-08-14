import express, {
  Request,
  Response,
  NextFunction,
} from 'express';

import cors from 'cors';
import multer from 'multer';
import crypto from 'node:crypto';

import { GoogleGenAI } from '@google/genai';

import {
  createCanvas,
  DOMMatrix,
  Path2D,
  ImageData,
} from 'canvas';

/* =========================================================
   TIPOS
   ========================================================= */

type JsonObject = Record<string, any>;

interface KeyState {
  ai: GoogleGenAI;
  blockedUntil: number;
  keyNumber: number;
}

interface DownloadedFile {
  buffer: Buffer;
  mimeType: string;
}

/* =========================================================
   PDF.JS
   =========================================================
   Se carga dinámicamente para evitar problemas entre
   CommonJS / ESM con las versiones modernas de pdfjs-dist.
   ========================================================= */

let pdfjsPromise: Promise<any> | null = null;

async function getPdfJs(): Promise<any> {
  if (!pdfjsPromise) {
    pdfjsPromise = new Function(
      'return import("pdfjs-dist/legacy/build/pdf.mjs")'
    )();
  }

  return pdfjsPromise;
}

/* =========================================================
   PDFJS + CANVAS
   ========================================================= */

(globalThis as any).DOMMatrix = DOMMatrix;
(globalThis as any).Path2D = Path2D;
(globalThis as any).ImageData = ImageData;

/* =========================================================
   EXPRESS
   ========================================================= */

const app = express();

/* =========================================================
   CONFIGURACIÓN
   ========================================================= */

const PORT =
  Number(process.env.PORT) || 3000;

const modelName =
  process.env.GEMINI_MODEL ||
  'gemini-3-flash-preview';

const API_TOKEN =
  process.env.API_TOKEN?.trim() || '';

const MAX_UPLOAD_SIZE =
  20 * 1024 * 1024;

const MAX_RECEIPT_FILE_SIZE =
  MAX_UPLOAD_SIZE;

const MAX_BODY_SIZE =
  '1mb';

/* =========================================================
   MULTER
   ========================================================= */

const upload = multer({
  storage: multer.memoryStorage(),

  limits: {
    fileSize: MAX_UPLOAD_SIZE,
    files: 1,
  },

  fileFilter: (
    _req,
    file,
    cb
  ) => {
    const allowed = [
      'application/pdf',
      'image/png',
      'image/jpeg',
      'image/webp',
      'image/gif',
      'image/svg+xml',
    ];

    if (
      allowed.includes(
        file.mimetype.toLowerCase()
      )
    ) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Tipo de archivo no permitido: ${file.mimetype}`
        )
      );
    }
  },
});

/* =========================================================
   CORS
   ========================================================= */

app.use(
  cors({
    origin: '*',
    methods: [
      'GET',
      'POST',
      'OPTIONS',
    ],
    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'X-API-Key',
    ],
  })
);

/* =========================================================
   OPTIONS
   ========================================================= */

app.options(
  '*',
  cors()
);

/* =========================================================
   BODY PARSERS
   ========================================================= */

app.use(
  express.json({
    limit: MAX_BODY_SIZE,
  })
);

app.use(
  express.text({
    type: ['text/plain'],
    limit: MAX_BODY_SIZE,
  })
);

/* =========================================================
   GEMINI API KEYS
   ========================================================= */

const apiKeys = [
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
  .filter(
    (
      key
    ): key is string =>
      typeof key === 'string' &&
      key.trim().length > 0
  )
  .map(
    (key) => key.trim()
  );

if (
  apiKeys.length === 0
) {
  console.error(
    'ERROR: No hay ninguna GEMINI_API_KEY configurada.'
  );
} else {
  console.log(
    `Gemini: ${apiKeys.length} API keys configuradas.`
  );
}

/* =========================================================
   CREAR CLIENTES GEMINI
   ========================================================= */

const keyStates: KeyState[] =
  apiKeys.map(
    (
      key,
      index
    ) => ({
      ai: new GoogleGenAI({
        apiKey: key,
      }),

      blockedUntil: 0,

      keyNumber:
        index + 1,
    })
  );

let currentKeyIndex = 0;

/* =========================================================
   ERROR -> STRING
   ========================================================= */

function errorMessage(
  error: unknown
): string {
  if (
    error instanceof Error
  ) {
    return error.message;
  }

  return String(error);
}

/* =========================================================
   DETECTAR RATE LIMIT
   ========================================================= */

function isRateLimitError(
  error: unknown
): boolean {
  const message =
    errorMessage(
      error
    ).toLowerCase();

  return (
    message.includes('429') ||
    message.includes(
      'too_many_requests'
    ) ||
    message.includes(
      'ratelimiterror'
    ) ||
    message.includes(
      'rate limit'
    ) ||
    message.includes(
      'quota exceeded'
    ) ||
    message.includes(
      'resource exhausted'
    ) ||
    message.includes(
      'resource_exhausted'
    ) ||
    message.includes(
      'too many requests'
    )
  );
}

/* =========================================================
   RETRY AFTER
   ========================================================= */

function getRetryAfterMs(
  error: unknown
): number {
  const message =
    errorMessage(
      error
    );

  const patterns = [
    /retry in\s+([\d.]+)\s*s/i,
    /retry after\s+([\d.]+)\s*s/i,
    /retryDelay["']?\s*[:=]\s*["']?([\d.]+)s/i,
    /seconds["']?\s*[:=]\s*["']?([\d.]+)/i,
  ];

  for (
    const pattern of patterns
  ) {
    const match =
      message.match(
        pattern
      );

    if (match) {
      const seconds =
        Number.parseFloat(
          match[1]
        );

      if (
        Number.isFinite(
          seconds
        ) &&
        seconds > 0
      ) {
        return Math.min(
          Math.ceil(
            seconds * 1000
          ),
          10 * 60 * 1000
        );
      }
    }
  }

  return 30_000;
}

/* =========================================================
   OBTENER KEY DISPONIBLE
   ========================================================= */

function getAvailableKey():
  KeyState | null {
  if (
    keyStates.length === 0
  ) {
    return null;
  }

  const now =
    Date.now();

  for (
    let i = 0;
    i < keyStates.length;
    i++
  ) {
    const index =
      (
        currentKeyIndex +
        i
      ) %
      keyStates.length;

    const keyState =
      keyStates[index];

    if (
      keyState.blockedUntil <=
      now
    ) {
      currentKeyIndex =
        (
          index + 1
        ) %
        keyStates.length;

      return keyState;
    }
  }

  return null;
}

/* =========================================================
   BLOQUEAR KEY
   ========================================================= */

function blockKey(
  keyState: KeyState,
  retryAfterMs: number
) {
  keyState.blockedUntil =
    Date.now() +
    retryAfterMs;
}

/* =========================================================
   GEMINI CON ROTACIÓN DE KEYS
   ========================================================= */

async function runGemini<T>(
  operation: (
    ai: GoogleGenAI
  ) => Promise<T>
): Promise<T> {
  if (
    keyStates.length === 0
  ) {
    throw new Error(
      'No hay ninguna API key de Gemini configurada en el servidor.'
    );
  }

  const attemptedKeys =
    new Set<number>();

  let lastRateLimitError:
    unknown = null;

  for (
    let attempt = 0;
    attempt < keyStates.length;
    attempt++
  ) {
    const keyState =
      getAvailableKey();

    if (!keyState) {
      throw new Error(
        'Todas las API keys de Gemini están temporalmente limitadas.'
      );
    }

    if (
      attemptedKeys.has(
        keyState.keyNumber
      )
    ) {
      break;
    }

    attemptedKeys.add(
      keyState.keyNumber
    );

    try {
      console.log(
        `Gemini: usando API key ${keyState.keyNumber}`
      );

      const result =
        await operation(
          keyState.ai
        );

      return result;
    } catch (
      error: unknown
    ) {
      if (
        !isRateLimitError(
          error
        )
      ) {
        throw error;
      }

      lastRateLimitError =
        error;

      const retryAfterMs =
        getRetryAfterMs(
          error
        );

      blockKey(
        keyState,
        retryAfterMs
      );

      console.warn(
        `Gemini: API key ${keyState.keyNumber} limitada durante ${Math.ceil(
          retryAfterMs / 1000
        )}s.`
      );
    }
  }

  if (
    lastRateLimitError
  ) {
    throw lastRateLimitError;
  }

  throw new Error(
    'Todas las API keys de Gemini alcanzaron el límite.'
  );
}

/* =========================================================
   BODY NORMALIZADO
   ========================================================= */

function getRequestBody(
  req: Request
): JsonObject {
  let body: unknown =
    req.body;

  if (
    typeof body === 'string'
  ) {
    if (
      body.trim().length === 0
    ) {
      return {};
    }

    try {
      body =
        JSON.parse(
          body
        );
    } catch {
      return {};
    }
  }

  if (
    !body ||
    typeof body !== 'object' ||
    Array.isArray(body)
  ) {
    return {};
  }

  return body as JsonObject;
}

/* =========================================================
   PARAM STRING
   ========================================================= */

function getParamString(
  value:
    | string
    | string[]
    | undefined
): string {
  if (
    typeof value === 'string'
  ) {
    return value;
  }

  if (
    Array.isArray(value)
  ) {
    return value[0] || '';
  }

  return '';
}

/* =========================================================
   PARSEAR JSON GEMINI
   ========================================================= */

function parseGeminiJson(
  text: string
): any {
  let cleaned =
    text.trim();

  cleaned =
    cleaned.replace(
      /^```json\s*/i,
      ''
    );

  cleaned =
    cleaned.replace(
      /^```\s*/i,
      ''
    );

  cleaned =
    cleaned.replace(
      /\s*```$/i,
      ''
    );

  cleaned =
    cleaned.trim();

  try {
    return JSON.parse(
      cleaned
    );
  } catch {}

  const objectStart =
    cleaned.indexOf('{');

  const objectEnd =
    cleaned.lastIndexOf('}');

  if (
    objectStart !== -1 &&
    objectEnd > objectStart
  ) {
    const candidate =
      cleaned.slice(
        objectStart,
        objectEnd + 1
      );

    try {
      return JSON.parse(
        candidate
      );
    } catch {}
  }

  const arrayStart =
    cleaned.indexOf('[');

  const arrayEnd =
    cleaned.lastIndexOf(']');

  if (
    arrayStart !== -1 &&
    arrayEnd > arrayStart
  ) {
    const candidate =
      cleaned.slice(
        arrayStart,
        arrayEnd + 1
      );

    try {
      return JSON.parse(
        candidate
      );
    } catch {}
  }

  throw new Error(
    'Gemini devolvió un JSON inválido.'
  );
}

/* =========================================================
   AUTH
   ========================================================= */

function auth(
  req: Request,
  res: Response,
  next: NextFunction
) {
  /*
   * Si API_TOKEN no está configurado,
   * se permite la solicitud.
   */

  if (!API_TOKEN) {
    return next();
  }

  const authorization =
    req.headers.authorization ||
    '';

  const bearerToken =
    authorization.replace(
      /^Bearer\s+/i,
      ''
    ).trim();

  const headerToken =
    typeof req.headers[
      'x-api-key'
    ] === 'string'
      ? req.headers[
          'x-api-key'
        ].trim()
      : '';

  const sentToken =
    headerToken ||
    bearerToken;

  if (
    !sentToken ||
    sentToken !== API_TOKEN
  ) {
    return res.status(401).json({
      error:
        'no autorizado',
    });
  }

  return next();
}

/* =========================================================
   GITHUB CONFIG
   ========================================================= */

function ghConfig() {
  const owner =
    process.env.GITHUB_OWNER?.trim();

  const repo =
    process.env.GITHUB_REPO?.trim();

  const branch =
    process.env.GITHUB_BRANCH?.trim() ||
    'main';

  const token =
    process.env.GITHUB_TOKEN?.trim();

  if (
    !owner ||
    !repo ||
    !token
  ) {
    throw new Error(
      'Faltan GITHUB_OWNER, GITHUB_REPO o GITHUB_TOKEN en Render.'
    );
  }

  return {
    owner,
    repo,
    branch,
    token,
  };
}

/* =========================================================
   VALIDAR PATH GITHUB
   ========================================================= */

function validateGithubPath(
  path: string
): string {
  let normalized =
    path
      .replace(
        /^\/+/,
        ''
      )
      .replace(
        /\\/g,
        '/'
      );

  normalized =
    normalized.replace(
      /\/+/g,
      '/'
    );

  if (
    !normalized ||
    normalized.includes('..') ||
    normalized.startsWith(
      '.git/'
    ) ||
    normalized === '.git'
  ) {
    throw new Error(
      'Ruta de archivo inválida.'
    );
  }

  return normalized;
}

/* =========================================================
   ENCODE PATH GITHUB
   ========================================================= */

function encodePathForGithub(
  path: string
): string {
  return path
    .split('/')
    .map(
      (part) =>
        encodeURIComponent(
          part
        )
    )
    .join('/');
}

/* =========================================================
   BASE64 URL SAFE
   ========================================================= */

function encodePathB64(
  path: string
): string {
  return Buffer.from(
    path,
    'utf8'
  )
    .toString('base64')
    .replace(
      /\+/g,
      '-'
    )
    .replace(
      /\//g,
      '_'
    )
    .replace(
      /=+$/g,
      ''
    );
}

/* =========================================================
   DECODE BASE64 URL SAFE
   ========================================================= */

function decodePathB64(
  value: string
): string {
  if (
    !/^[A-Za-z0-9_-]+$/.test(
      value
    )
  ) {
    throw new Error(
      'pathB64 inválido.'
    );
  }

  let normalized =
    value
      .replace(
        /-/g,
        '+'
      )
      .replace(
        /_/g,
        '/'
      );

  while (
    normalized.length % 4 !==
    0
  ) {
    normalized += '=';
  }

  const decoded =
    Buffer.from(
      normalized,
      'base64'
    ).toString(
      'utf8'
    );

  return validateGithubPath(
    decoded
  );
}

/* =========================================================
   GITHUB HEADERS
   ========================================================= */

function githubHeaders(
  token: string
) {
  return {
    Authorization:
      `Bearer ${token}`,

    Accept:
      'application/vnd.github+json',

    'X-GitHub-Api-Version':
      '2022-11-28',

    'Content-Type':
      'application/json',
  };
}

/* =========================================================
   GITHUB GET
   ========================================================= */

async function getGithubFile(
  path: string
) {
  const {
    owner,
    repo,
    branch,
    token,
  } = ghConfig();

  const safePath =
    validateGithubPath(
      path
    );

  const encodedPath =
    encodePathForGithub(
      safePath
    );

  const url =
    `https://api.github.com/repos/${encodeURIComponent(
      owner
    )}/${encodeURIComponent(
      repo
    )}/contents/${encodedPath}?ref=${encodeURIComponent(
      branch
    )}`;

  const response =
    await fetch(
      url,
      {
        method: 'GET',
        headers:
          githubHeaders(
            token
          ),
      }
    );

  if (
    response.status ===
    404
  ) {
    return null;
  }

  if (
    !response.ok
  ) {
    const text =
      await response.text();

    throw new Error(
      `GitHub GET error ${response.status}: ${text}`
    );
  }

  return response.json();
}

/* =========================================================
   GITHUB UPLOAD
   ========================================================= */

async function uploadToGithub(
  path: string,
  data: Buffer
) {
  const {
    owner,
    repo,
    branch,
    token,
  } = ghConfig();

  const safePath =
    validateGithubPath(
      path
    );

  if (
    data.length >
    MAX_UPLOAD_SIZE
  ) {
    throw new Error(
      'El archivo supera el límite máximo de 20 MB.'
    );
  }

  const existing =
    await getGithubFile(
      safePath
    );

  const url =
    `https://api.github.com/repos/${encodeURIComponent(
      owner
    )}/${encodeURIComponent(
      repo
    )}/contents/${encodePathForGithub(
      safePath
    )}`;

  const body: JsonObject = {
    message:
      `Torchill receipt upload: ${safePath}`,

    content:
      data.toString(
        'base64'
      ),

    branch,
  };

  if (
    existing &&
    typeof existing.sha ===
      'string'
  ) {
    body.sha =
      existing.sha;
  }

  const response =
    await fetch(
      url,
      {
        method: 'PUT',

        headers:
          githubHeaders(
            token
          ),

        body:
          JSON.stringify(
            body
          ),
      }
    );

  if (
    !response.ok
  ) {
    const text =
      await response.text();

    throw new Error(
      `GitHub upload error ${response.status}: ${text}`
    );
  }

  const result =
    await response.json();

  return {
    path: safePath,

    sha:
      result.content?.sha ||
      null,

    branch,

    owner,

    repo,

    pathB64:
      encodePathB64(
        safePath
      ),
  };
}

/* =========================================================
   MIME
   ========================================================= */

function mimeForPath(
  path: string
): string {
  const lower =
    path.toLowerCase();

  if (
    lower.endsWith('.pdf')
  ) {
    return 'application/pdf';
  }

  if (
    lower.endsWith('.png')
  ) {
    return 'image/png';
  }

  if (
    lower.endsWith('.webp')
  ) {
    return 'image/webp';
  }

  if (
    lower.endsWith('.gif')
  ) {
    return 'image/gif';
  }

  if (
    lower.endsWith('.svg')
  ) {
    return 'image/svg+xml';
  }

  if (
    lower.endsWith('.jpg') ||
    lower.endsWith('.jpeg')
  ) {
    return 'image/jpeg';
  }

  return 'application/octet-stream';
}

/* =========================================================
   MIME DESDE URL
   ========================================================= */

function getMimeType(
  fileUrl: string,
  contentType: string
): string {
  const normalized =
    contentType
      .split(';')[0]
      .trim()
      .toLowerCase();

  if (
    normalized &&
    normalized !==
      'application/octet-stream'
  ) {
    return normalized;
  }

  try {
    const parsed =
      new URL(
        fileUrl
      );

    return mimeForPath(
      parsed.pathname
    );
  } catch {
    return 'application/octet-stream';
  }
}

/* =========================================================
   DESCARGAR ARCHIVO GITHUB
   ========================================================= */

async function downloadGithubFile(
  path: string
): Promise<DownloadedFile> {
  const {
    owner,
    repo,
    branch,
    token,
  } = ghConfig();

  const safePath =
    validateGithubPath(
      path
    );

  const encodedPath =
    encodePathForGithub(
      safePath
    );

  const url =
    `https://api.github.com/repos/${encodeURIComponent(
      owner
    )}/${encodeURIComponent(
      repo
    )}/contents/${encodedPath}?ref=${encodeURIComponent(
      branch
    )}`;

  const response =
    await fetch(
      url,
      {
        headers:
          githubHeaders(
            token
          ),
      }
    );

  if (
    response.status ===
    404
  ) {
    throw new Error(
      'Archivo no encontrado en GitHub.'
    );
  }

  if (
    !response.ok
  ) {
    const text =
      await response.text();

    throw new Error(
      `GitHub download error ${response.status}: ${text}`
    );
  }

  const result =
    await response.json();

  if (
    result.type !== 'file' ||
    typeof result.content !==
      'string'
  ) {
    throw new Error(
      'GitHub no devolvió un archivo válido.'
    );
  }

  const cleanBase64 =
    result.content.replace(
      /\s/g,
      ''
    );

  const buffer =
    Buffer.from(
      cleanBase64,
      'base64'
    );

  if (
    buffer.length >
    MAX_RECEIPT_FILE_SIZE
  ) {
    throw new Error(
      'El archivo supera el límite máximo de 20 MB.'
    );
  }

  return {
    buffer,

    mimeType:
      mimeForPath(
        safePath
      ),
  };
}

/* =========================================================
   PDF -> PNG
   ========================================================= */

async function pdfFirstPageToPng(
  buffer: Buffer
): Promise<Buffer> {
  const pdfjsLib =
    await getPdfJs();

  const loadingTask =
    pdfjsLib.getDocument({
      data:
        new Uint8Array(
          buffer
        ),

      isEvalSupported:
        false,

      useSystemFonts:
        true,

      disableFontFace:
        false,
    });

  const pdf =
    await loadingTask.promise;

  try {
    if (
      pdf.numPages < 1
    ) {
      throw new Error(
        'El PDF no contiene páginas.'
      );
    }

    const page =
      await pdf.getPage(1);

    const baseViewport =
      page.getViewport({
        scale: 1,
      });

    const maxSide =
      1600;

    const scale =
      Math.min(
        2,
        maxSide /
          Math.max(
            baseViewport.width,
            baseViewport.height
          )
      );

    const viewport =
      page.getViewport({
        scale,
      });

    const width =
      Math.max(
        1,
        Math.ceil(
          viewport.width
        )
      );

    const height =
      Math.max(
        1,
        Math.ceil(
          viewport.height
        )
      );

    const canvas =
      createCanvas(
        width,
        height
      );

    const context =
      canvas.getContext(
        '2d'
      );

    context.imageSmoothingEnabled =
      true;

    await page.render({
      canvasContext:
        context as any,

      viewport,

      intent:
        'display',
    }).promise;

    return canvas.toBuffer(
      'image/png'
    );
  } finally {
    try {
      await pdf.destroy();
    } catch {}
  }
}

/* =========================================================
   DESCARGAR RECEIPT EXTERNO
   ========================================================= */

async function downloadReceipt(
  fileUrl: string
): Promise<DownloadedFile> {
  let parsedUrl: URL;

  try {
    parsedUrl =
      new URL(
        fileUrl
      );
  } catch {
    throw new Error(
      'fileUrl no es una URL válida.'
    );
  }

  if (
    parsedUrl.protocol !==
      'http:' &&
    parsedUrl.protocol !==
      'https:'
  ) {
    throw new Error(
      'fileUrl debe utilizar HTTP o HTTPS.'
    );
  }

  const fileRes =
    await fetch(
      parsedUrl,
      {
        redirect: 'follow',
      }
    );

  if (
    !fileRes.ok
  ) {
    throw new Error(
      `No se pudo descargar el archivo (${fileRes.status}).`
    );
  }

  const contentLength =
    fileRes.headers.get(
      'content-length'
    );

  if (
    contentLength &&
    Number(contentLength) >
      MAX_RECEIPT_FILE_SIZE
  ) {
    throw new Error(
      'El archivo supera el límite máximo de 20 MB.'
    );
  }

  if (
    !fileRes.body
  ) {
    throw new Error(
      'La respuesta no contiene datos.'
    );
  }

  const reader =
    fileRes.body.getReader();

  const chunks: Buffer[] =
    [];

  let total = 0;

  try {
    while (true) {
      const {
        done,
        value,
      } =
        await reader.read();

      if (done) {
        break;
      }

      if (
        value &&
        value.length > 0
      ) {
        total +=
          value.length;

        if (
          total >
          MAX_RECEIPT_FILE_SIZE
        ) {
          try {
            await reader.cancel();
          } catch {}

          throw new Error(
            'El archivo supera el límite máximo de 20 MB.'
          );
        }

        chunks.push(
          Buffer.from(
            value
          )
        );
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {}
  }

  const buffer =
    Buffer.concat(
      chunks
    );

  return {
    buffer,

    mimeType:
      getMimeType(
        fileUrl,
        fileRes.headers.get(
          'content-type'
        ) || ''
      ),
  };
}

/* =========================================================
   HEALTH
   ========================================================= */

app.get(
  '/health',
  (
    _req: Request,
    res: Response
  ) => {
    return res.status(200).json({
      ok: true,

      service:
        'torchill-api',

      model:
        modelName,

      gemini:
        apiKeys.length > 0,

      geminiConfigured:
        apiKeys.length > 0,

      geminiKeys:
        apiKeys.length,

      githubConfigured:
        Boolean(
          process.env.GITHUB_OWNER &&
          process.env.GITHUB_REPO &&
          process.env.GITHUB_TOKEN
        ),
    });
  }
);

/* =========================================================
   PROJECT COPY HEALTH
   ========================================================= */

app.get(
  '/api/gemini/project-copy',
  (
    _req: Request,
    res: Response
  ) => {
    return res.status(200).json({
      status: 'ok',

      service:
        'torchill-api',

      model:
        modelName,

      geminiConfigured:
        apiKeys.length > 0,

      geminiKeys:
        apiKeys.length,

      message:
        'Torchill API funcionando correctamente.',
    });
  }
);

/* =========================================================
   GET RECEIPT
   ========================================================= */

app.get(
  '/receipt/:pathB64',
  async (
    req: Request,
    res: Response
  ) => {
    try {
      const pathB64 =
        getParamString(
          req.params.pathB64
        );

      if (!pathB64) {
        return res.status(400).json({
          error:
            'Falta el parámetro pathB64.',
        });
      }

      const path =
        decodePathB64(
          pathB64
        );

      const {
        buffer,
        mimeType,
      } =
        await downloadGithubFile(
          path
        );

      res.setHeader(
        'Content-Type',
        mimeType
      );

      res.setHeader(
        'Content-Length',
        String(
          buffer.length
        )
      );

      res.setHeader(
        'Cache-Control',
        'public, max-age=3600'
      );

      return res
        .status(200)
        .send(buffer);
    } catch (
      error: unknown
    ) {
      console.error(
        'Receipt download error:',
        error
      );

      return res.status(404).json({
        error:
          'No se pudo obtener el archivo.',

        details:
          errorMessage(
            error
          ),
      });
    }
  }
);

/* =========================================================
   UPLOAD
   ========================================================= */

app.post(
  '/upload',
  auth,
  upload.single('file'),
  async (
    req: Request,
    res: Response
  ) => {
    try {
      let buffer:
        | Buffer
        | null =
        req.file?.buffer ||
        null;

      let originalName =
        req.file?.originalname ||
        '';

      let mimeType =
        req.file?.mimetype ||
        '';

      const body =
        getRequestBody(
          req
        );

      /* =====================================================
         BASE64 FALLBACK
         ===================================================== */

      if (!buffer) {
        const dataBase64 =
          typeof body.dataBase64 ===
          'string'
            ? body.dataBase64.trim()
            : '';

        if (
          dataBase64
        ) {
          const clean =
            dataBase64.replace(
              /^data:[^;]+;base64,/i,
              ''
            );

          try {
            buffer =
              Buffer.from(
                clean,
                'base64'
              );
          } catch {
            return res.status(400).json({
              error:
                'dataBase64 inválido.',
            });
          }

          originalName =
            typeof body.fileName ===
            'string'
              ? body.fileName
              : typeof body.filename ===
                'string'
                ? body.filename
                : 'receipt';

          mimeType =
            typeof body.mimeType ===
            'string'
              ? body.mimeType
              : typeof body.mime ===
                'string'
                ? body.mime
                : typeof body.contentType ===
                  'string'
                  ? body.contentType
                  : '';
        }
      }

      if (!buffer) {
        return res.status(400).json({
          error:
            'Falta el archivo. Usá multipart/form-data con campo "file" o JSON con "dataBase64".',
        });
      }

      if (
        buffer.length >
        MAX_UPLOAD_SIZE
      ) {
        return res.status(413).json({
          error:
            'El archivo supera el límite máximo de 20 MB.',
        });
      }

      /* =====================================================
         DETECTAR EXTENSIÓN
         ===================================================== */

      const requestedExt =
        typeof body.ext ===
        'string'
          ? body.ext
              .trim()
              .replace(
                /^\./,
                ''
              )
              .toLowerCase()
          : '';

      const normalizedMime =
        mimeType
          .split(';')[0]
          .trim()
          .toLowerCase();

      let finalExtension =
        '';

      if (
        normalizedMime ===
        'application/pdf'
      ) {
        finalExtension =
          'pdf';
      } else if (
        normalizedMime ===
        'image/png'
      ) {
        finalExtension =
          'png';
      } else if (
        normalizedMime ===
        'image/webp'
      ) {
        finalExtension =
          'webp';
      } else if (
        normalizedMime ===
        'image/gif'
      ) {
        finalExtension =
          'gif';
      } else if (
        normalizedMime ===
          'image/jpeg' ||
        normalizedMime ===
          'image/jpg'
      ) {
        finalExtension =
          'jpg';
      } else if (
        normalizedMime ===
        'image/svg+xml'
      ) {
        finalExtension =
          'svg';
      } else if (
        requestedExt
      ) {
        finalExtension =
          requestedExt;
      } else if (
        originalName.includes('.')
      ) {
        finalExtension =
          originalName
            .split('.')
            .pop()
            ?.toLowerCase() ||
          '';
      }

      /* =====================================================
         DETECTAR PDF REAL POR BYTES
         ===================================================== */

      const isPdf =
        buffer.length >= 5 &&
        buffer
          .subarray(
            0,
            5
          )
          .toString(
            'ascii'
          ) ===
          '%PDF-';

      if (isPdf) {
        finalExtension =
          'pdf';

        mimeType =
          'application/pdf';
      }

      /* =====================================================
         NORMALIZAR EXTENSION
         ===================================================== */

      if (
        finalExtension ===
          'jpeg' ||
        finalExtension ===
          'jpe'
      ) {
        finalExtension =
          'jpg';
      }

      const allowedExtensions = [
        'pdf',
        'png',
        'jpg',
        'webp',
        'gif',
        'svg',
      ];

      if (
        !allowedExtensions.includes(
          finalExtension
        )
      ) {
        finalExtension =
          'jpg';
      }

      /* =====================================================
         MIME FINAL
         ===================================================== */

      const finalMimeType =
        mimeForPath(
          `file.${finalExtension}`
        );

      if (
        finalMimeType !==
        'application/octet-stream'
      ) {
        mimeType =
          finalMimeType;
      }

      /* =====================================================
         PATH GITHUB
         ===================================================== */

      let requestedPath =
        typeof body.path ===
        'string'
          ? body.path.trim()
          : '';

      if (
        !requestedPath
      ) {
        requestedPath =
          `receipts/${Date.now()}-${crypto.randomUUID()}.${finalExtension}`;
      } else {
        requestedPath =
          validateGithubPath(
            requestedPath
          );

        const hasExtension =
          /\.[a-zA-Z0-9]+$/.test(
            requestedPath
          );

        if (
          !hasExtension
        ) {
          requestedPath =
            `${requestedPath}.${finalExtension}`;
        } else {
          requestedPath =
            requestedPath.replace(
              /\.[a-zA-Z0-9]+$/,
              `.${finalExtension}`
            );
        }
      }

      requestedPath =
        validateGithubPath(
          requestedPath
        );

      console.log(
        'Upload:',
        JSON.stringify({
          originalName,
          originalMime:
            mimeType,
          finalExtension,
          finalMimeType,
          size:
            buffer.length,
          path:
            requestedPath,
          pdfDetected:
            isPdf,
        })
      );

      /* =====================================================
         GITHUB
         ===================================================== */

      const result =
        await uploadToGithub(
          requestedPath,
          buffer
        );

      /* =====================================================
         URL
         ===================================================== */

      const receiptUrl =
        `/receipt/${result.pathB64}`;

      const host =
        req.get('host');

      const absoluteUrl =
        host
          ? `${req.protocol}://${host}${receiptUrl}`
          : receiptUrl;

      /* =====================================================
         RESPONSE
         ===================================================== */

      return res.status(200).json({
        ok: true,

        ...result,

        mimeType:
          mimeForPath(
            result.path
          ),

        size:
          buffer.length,

        url:
          receiptUrl,

        receiptUrl,

        absoluteUrl,
      });
    } catch (
      error: unknown
    ) {
      console.error(
        'Upload error:',
        error
      );

      const message =
        errorMessage(
          error
        );

      if (
        message.includes(
          'File too large'
        )
      ) {
        return res.status(413).json({
          error:
            'El archivo supera el límite máximo de 20 MB.',
        });
      }

      return res.status(500).json({
        error:
          'No se pudo subir el archivo.',

        details:
          message,
      });
    }
  }
);

/* =========================================================
   GEMINI PROJECT COPY
   ========================================================= */

app.post(
  '/api/gemini/project-copy',
  async (
    req: Request,
    res: Response
  ) => {
    try {
      if (
        apiKeys.length ===
        0
      ) {
        return res.status(503).json({
          error:
            'El servicio Gemini no está configurado.',
        });
      }

      const body =
        getRequestBody(
          req
        );

      const {
        action,
        text,
        language,
        title,
        segments,
      } = body;

      if (
        Object.keys(
          body
        ).length === 0
      ) {
        return res.status(400).json({
          error:
            'La solicitud no contiene un body JSON válido.',
        });
      }

      /* =====================================================
         TRANSLATE
         ===================================================== */

      if (
        action ===
        'translate'
      ) {
        if (
          !Array.isArray(
            segments
          )
        ) {
          return res.status(400).json({
            error:
              '"segments" debe ser un array.',
          });
        }

        if (
          segments.length === 0
        ) {
          return res.status(400).json({
            error:
              '"segments" no puede estar vacío.',
          });
        }

        const targetLanguage =
          language === 'en'
            ? 'inglés'
            : language === 'zh'
              ? 'chino'
              : language === 'pt'
                ? 'portugués'
                : language === 'fr'
                  ? 'francés'
                  : language === 'it'
                    ? 'italiano'
                    : 'español';

        const prompt = `
Actúa como traductor profesional especializado
en diseño gráfico y comunicación visual.

Traduce los siguientes textos al ${targetLanguage}.

REGLAS:
- Mantén exactamente el significado.
- No inventes información.
- No elimines información.
- Conserva el orden.
- Mantén el tono profesional.
- La traducción debe sonar natural.
- No agregues explicaciones.
- Devuelve un elemento por cada texto recibido.

TEXTOS:

${JSON.stringify(
  segments
)}

Devuelve exclusivamente JSON válido.
`;

        const interaction =
          await runGemini(
            (ai) =>
              ai.interactions.create(
                {
                  model:
                    modelName,

                  input:
                    prompt,

                  response_format:
                    {
                      type:
                        'text',

                      mime_type:
                        'application/json',

                      schema: {
                        type:
                          'object',

                        properties: {
                          translations:
                            {
                              type:
                                'array',

                              items: {
                                type:
                                  'string',
                              },
                            },
                        },

                        required: [
                          'translations',
                        ],
                      },
                    },
                } as any
              )
          );

        const output =
          interaction.output_text
            ?.trim();

        if (
          !output
        ) {
          throw new Error(
            'Gemini devolvió una respuesta vacía.'
          );
        }

        const result =
          parseGeminiJson(
            output
          );

        return res
          .status(200)
          .json(result);
      }

      /* =====================================================
         PROJECT / PORTFOLIO
         ===================================================== */

      if (
        typeof title !==
          'string' ||
        title.trim()
          .length === 0
      ) {
        return res.status(400).json({
          error:
            'Falta el campo "title".',
        });
      }

      if (
        typeof text !==
          'string' ||
        text.trim()
          .length === 0
      ) {
        return res.status(400).json({
          error:
            'Falta el campo "text".',
        });
      }

      const prompt = `
Actúa como director de arte y editor
especializado en portfolios profesionales
de diseño gráfico.

PROYECTO:
${title}

TEXTO ORIGINAL:
${text}

OBJETIVOS:
- Mejorar claridad.
- Mejorar redacción.
- Mantener intención original.
- Utilizar lenguaje profesional.
- Evitar frases publicitarias genéricas.
- No inventar información.
- No agregar datos inexistentes.
- Mantener el contenido apropiado
  para un portfolio profesional.

Devuelve exclusivamente JSON válido.
`;

      const interaction =
        await runGemini(
          (ai) =>
            ai.interactions.create(
              {
                model:
                  modelName,

                input:
                  prompt,

                response_format:
                  {
                    type:
                      'text',

                    mime_type:
                      'application/json',

                    schema: {
                      type:
                        'object',

                      properties: {
                        lead: {
                          type:
                            'string',
                        },

                        discipline: {
                          type:
                            'string',
                        },

                        sections: {
                          type:
                            'array',

                          items: {
                            type:
                              'object',

                            properties: {
                              title: {
                                type:
                                  'string',
                              },

                              summary: {
                                type:
                                  'string',
                              },
                            },

                            required: [
                              'title',
                              'summary',
                            ],
                          },
                        },

                        imageAlts: {
                          type:
                            'array',

                          items: {
                            type:
                              'string',
                          },
                        },
                      },

                      required: [
                        'lead',
                        'discipline',
                        'sections',
                        'imageAlts',
                      ],
                    },
                  },
              } as any
            )
        );

      const output =
        interaction.output_text
          ?.trim();

      if (
        !output
      ) {
        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      const result =
        parseGeminiJson(
          output
        );

      return res
        .status(200)
        .json(result);
    } catch (
      error: unknown
    ) {
      console.error(
        'Gemini project-copy error:',
        error
      );

      if (
        isRateLimitError(
          error
        )
      ) {
        return res.status(429).json({
          error:
            'Todas las API keys de Gemini están temporalmente limitadas.',

          retryable:
            true,
        });
      }

      return res.status(500).json({
        error:
          'Error procesando la solicitud con Gemini.',

        details:
          errorMessage(
            error
          ),
      });
    }
  }
);

/* =========================================================
   ANALYZE RECEIPT
   ========================================================= */

app.post(
  '/analyze-receipt',
  auth,
  async (
    req: Request,
    res: Response
  ) => {
    try {
      if (
        apiKeys.length ===
        0
      ) {
        return res.status(503).json({
          error:
            'El servicio Gemini no está configurado.',
        });
      }

      const body =
        getRequestBody(
          req
        );

      const {
        fileUrl,
        prompt,
      } = body;

      if (
        typeof fileUrl !==
          'string' ||
        fileUrl.trim()
          .length === 0
      ) {
        return res.status(400).json({
          error:
            'fileUrl es requerido.',
        });
      }

      if (
        typeof prompt !==
          'string' ||
        prompt.trim()
          .length === 0
      ) {
        return res.status(400).json({
          error:
            'prompt es requerido.',
        });
      }

      /* =====================================================
         DESCARGAR
         ===================================================== */

      let {
        buffer,
        mimeType,
      } =
        await downloadReceipt(
          fileUrl.trim()
        );

      console.log(
        `Receipt: archivo descargado (${buffer.length} bytes, ${mimeType})`
      );

      /* =====================================================
         DETECTAR PDF POR BYTES
         ===================================================== */

      const looksLikePdf =
        buffer.length >= 5 &&
        buffer
          .subarray(
            0,
            5
          )
          .toString(
            'ascii'
          ) ===
          '%PDF-';

      if (
        looksLikePdf
      ) {
        mimeType =
          'application/pdf';
      }

      /* =====================================================
         PDF -> PNG
         ===================================================== */

      if (
        mimeType ===
          'application/pdf' ||
        looksLikePdf
      ) {
        console.log(
          'Receipt: PDF detectado. Convirtiendo primera página a PNG...'
        );

        try {
          const png =
            await pdfFirstPageToPng(
              buffer
            );

          buffer =
            png;

          mimeType =
            'image/png';

          console.log(
            `Receipt: PDF convertido a PNG (${buffer.length} bytes)`
          );
        } catch (
          pdfError: unknown
        ) {
          console.error(
            'Receipt PDF conversion error:',
            pdfError
          );

          return res.status(422).json({
            error:
              'No se pudo convertir el PDF a imagen.',

            details:
              errorMessage(
                pdfError
              ),
          });
        }
      }

      /* =====================================================
         VALIDAR MIME FINAL
         ===================================================== */

      const supportedMimeTypes = [
        'image/png',
        'image/jpeg',
        'image/webp',
        'image/gif',
      ];

      if (
        !supportedMimeTypes.includes(
          mimeType
        )
      ) {
        return res.status(415).json({
          error:
            `Tipo de archivo no soportado para análisis: ${mimeType}`,
        });
      }

      /* =====================================================
         BASE64
         ===================================================== */

      const base64 =
        buffer.toString(
          'base64'
        );

      /* =====================================================
         GEMINI
         ===================================================== */

      const geminiPrompt = `
${prompt}

INSTRUCCIONES IMPORTANTES:
- Analiza cuidadosamente el comprobante.
- Extrae únicamente información visible o claramente inferible.
- No inventes datos.
- Si un campo no puede determinarse, utiliza null.
- Devuelve exclusivamente JSON válido.
- No incluyas markdown.
- No incluyas explicaciones fuera del JSON.
`;

      const response =
        await runGemini(
          (ai) =>
            ai.models.generateContent(
              {
                model:
                  modelName,

                contents: [
                  {
                    role:
                      'user',

                    parts: [
                      {
                        text:
                          geminiPrompt,
                      },

                      {
                        inlineData:
                          {
                            mimeType,
                            data:
                              base64,
                          },
                      },
                    ],
                  },
                ],

                config: {
                  responseMimeType:
                    'application/json',

                  temperature:
                    0.1,
                },
              }
            )
        );

      const outputText =
        response.text?.trim() ||
        '';

      if (
        !outputText
      ) {
        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      /* =====================================================
         JSON
         ===================================================== */

      const parsed =
        parseGeminiJson(
          outputText
        );

      return res.status(200).json({
        ok: true,

        detected:
          parsed,

        raw:
          outputText,
      });
    } catch (
      error: unknown
    ) {
      console.error(
        'Analyze receipt error:',
        error
      );

      if (
        isRateLimitError(
          error
        )
      ) {
        return res.status(429).json({
          error:
            'Todas las API keys de Gemini están temporalmente limitadas.',

          retryable:
            true,
        });
      }

      return res.status(500).json({
        error:
          'Error analizando el comprobante.',

        details:
          errorMessage(
            error
          ),
      });
    }
  }
);

/* =========================================================
   MULTER ERROR HANDLER
   ========================================================= */

app.use(
  (
    error: any,
    _req: Request,
    res: Response,
    next: NextFunction
  ) => {
    if (
      error instanceof multer.MulterError
    ) {
      if (
        error.code ===
        'LIMIT_FILE_SIZE'
      ) {
        return res.status(413).json({
          error:
            'El archivo supera el límite máximo de 20 MB.',
        });
      }

      return res.status(400).json({
        error:
          `Error de subida: ${error.message}`,
      });
    }

    if (
      error instanceof Error &&
      error.message.includes(
        'Tipo de archivo no permitido'
      )
    ) {
      return res.status(415).json({
        error:
          error.message,
      });
    }

    return next(
      error
    );
  }
);

/* =========================================================
   404
   ========================================================= */

app.use(
  (
    _req: Request,
    res: Response
  ) => {
    return res.status(404).json({
      error:
        'Endpoint no encontrado.',
    });
  }
);

/* =========================================================
   ERROR GLOBAL
   ========================================================= */

app.use(
  (
    error: unknown,
    _req: Request,
    res: Response,
    _next: NextFunction
  ) => {
    console.error(
      'Unhandled server error:',
      error
    );

    return res.status(500).json({
      error:
        'Error interno del servidor.',

      details:
        errorMessage(
          error
        ),
    });
  }
);

/* =========================================================
   START
   ========================================================= */

app.listen(
  PORT,
  () => {
    console.log(
      '=========================================='
    );

    console.log(
      'TORCHILL API'
    );

    console.log(
      '=========================================='
    );

    console.log(
      `Servidor: http://localhost:${PORT}`
    );

    console.log(
      `Puerto: ${PORT}`
    );

    console.log(
      `Modelo Gemini: ${modelName}`
    );

    console.log(
      `Gemini API keys: ${apiKeys.length}`
    );

    console.log(
      `GitHub configurado: ${
        Boolean(
          process.env.GITHUB_OWNER &&
          process.env.GITHUB_REPO &&
          process.env.GITHUB_TOKEN
        )
      }`
    );

    console.log(
      '------------------------------------------'
    );

    console.log(
      'GET  /health'
    );

    console.log(
      'GET  /receipt/:pathB64'
    );

    console.log(
      'POST /upload'
    );

    console.log(
      'POST /analyze-receipt'
    );

    console.log(
      'POST /api/gemini/project-copy'
    );

    console.log(
      '=========================================='
    );
  }
);
