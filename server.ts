import express, {
  Request,
  Response,
  NextFunction,
} from 'express';

import cors from 'cors';

import {
  GoogleGenAI,
} from '@google/genai';

import multer from 'multer';

import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.js';

import {
  createCanvas,
  DOMMatrix,
  ImageData,
} from 'canvas';

import * as napiCanvas from '@napi-rs/canvas';

import crypto from 'node:crypto';

/* =========================================================
   TIPOS
   ========================================================= */

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
   PDF.JS / CANVAS
   ========================================================= */

(globalThis as any).DOMMatrix = DOMMatrix;
(globalThis as any).ImageData = ImageData;

if (
  !(globalThis as any).Path2D &&
  (napiCanvas as any).Path2D
) {
  (globalThis as any).Path2D =
    (napiCanvas as any).Path2D;
}

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

/* =========================================================
   MULTER
   ========================================================= */

const upload = multer({
  storage: multer.memoryStorage(),

  limits: {
    fileSize:
      MAX_UPLOAD_SIZE,
  },
});

/* =========================================================
   CORS
   ========================================================= */

app.use(
  cors({
    origin: true,

    methods: [
      'GET',
      'POST',
      'PUT',
      'DELETE',
      'OPTIONS',
    ],

    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'X-API-Key',
    ],

    credentials: false,

    maxAge: 86400,
  })
);

/*
 * IMPORTANTE:
 *
 * NO usar:
 *
 * app.options('*', ...)
 *
 * porque Express 5 / path-to-regexp
 * produce:
 *
 * Missing parameter name at index 1: *
 */

/* =========================================================
   BODY PARSERS
   ========================================================= */

app.use(
  express.json({
    limit: '25mb',
  })
);

app.use(
  express.text({
    type: [
      'text/plain',
    ],
    limit: '25mb',
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
  .map((key) =>
    typeof key === 'string'
      ? key.trim()
      : ''
  )
  .filter(
    (key): key is string =>
      key.length > 0
  );

/* =========================================================
   LOG GEMINI
   ========================================================= */

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
   KEY MANAGER
   ========================================================= */

const keyStates: KeyState[] =
  apiKeys.map(
    (
      key,
      index
    ) => ({
      ai:
        new GoogleGenAI({
          apiKey: key,
        }),

      blockedUntil: 0,

      keyNumber:
        index + 1,
    })
  );

let currentKeyIndex = 0;

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

    const state =
      keyStates[index];

    if (
      state.blockedUntil <=
      now
    ) {
      currentKeyIndex =
        (
          index + 1
        ) %
        keyStates.length;

      return state;
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
): void {

  keyState.blockedUntil =
    Date.now() +
    Math.max(
      retryAfterMs,
      1000
    );
}

/* =========================================================
   RETRY AFTER
   ========================================================= */

function getRetryAfterMs(
  error: unknown
): number {

  const message =
    error instanceof Error
      ? error.message
      : String(error);

  const patterns = [
    /retry in\s+([\d.]+)s/i,
    /retryDelay[^0-9]*([\d.]+)s/i,
    /retry-after[^0-9]*([\d.]+)/i,
  ];

  for (
    const pattern of patterns
  ) {

    const match =
      message.match(
        pattern
      );

    if (
      match
    ) {

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
        return Math.ceil(
          seconds * 1000
        );
      }
    }
  }

  return 30_000;
}

/* =========================================================
   RATE LIMIT
   ========================================================= */

function isRateLimitError(
  error: unknown
): boolean {

  const message =
    error instanceof Error
      ? error.message
      : String(error);

  const normalized =
    message.toLowerCase();

  return (
    normalized.includes(
      '429'
    ) ||
    normalized.includes(
      'too_many_requests'
    ) ||
    normalized.includes(
      'ratelimit'
    ) ||
    normalized.includes(
      'rate limit'
    ) ||
    normalized.includes(
      'quota exceeded'
    ) ||
    normalized.includes(
      'resource exhausted'
    ) ||
    normalized.includes(
      'resource_exhausted'
    )
  );
}

/* =========================================================
   EJECUTAR GEMINI CON ROTACIÓN
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
    new Set<KeyState>();

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
        keyState
      )
    ) {
      break;
    }

    attemptedKeys.add(
      keyState
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

      const retryAfterMs =
        getRetryAfterMs(
          error
        );

      console.warn(
        `Gemini: API key ${keyState.keyNumber} limitada durante ${Math.ceil(
          retryAfterMs / 1000
        )}s.`
      );

      blockKey(
        keyState,
        retryAfterMs
      );
    }
  }

  throw new Error(
    'Todas las API keys de Gemini alcanzaron el límite.'
  );
}

/* =========================================================
   NORMALIZAR BODY
   ========================================================= */

function getRequestBody(
  req: Request
): Record<string, any> {

  let body: unknown =
    req.body;

  if (
    typeof body ===
    'string'
  ) {

    if (
      body.trim().length ===
      0
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
    typeof body !==
      'object' ||
    Array.isArray(body)
  ) {
    return {};
  }

  return body as Record<
    string,
    any
  >;
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
    typeof value ===
    'string'
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
   PARSE GEMINI JSON
   ========================================================= */

function parseGeminiJson(
  text: string
): any {

  const cleaned =
    text
      .trim()
      .replace(
        /^```json\s*/i,
        ''
      )
      .replace(
        /^```\s*/i,
        ''
      )
      .replace(
        /\s*```$/i,
        ''
      )
      .trim();

  try {

    return JSON.parse(
      cleaned
    );

  } catch {}

  const objectMatch =
    cleaned.match(
      /\{[\s\S]*\}/
    );

  if (
    objectMatch
  ) {

    try {

      return JSON.parse(
        objectMatch[0]
      );

    } catch {}
  }

  const arrayMatch =
    cleaned.match(
      /\[[\s\S]*\]/
    );

  if (
    arrayMatch
  ) {

    try {

      return JSON.parse(
        arrayMatch[0]
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
   * Si API_TOKEN no existe,
   * no bloqueamos la petición.
   */

  if (
    !API_TOKEN
  ) {
    return next();
  }

  const authorization =
    req.headers.authorization ||
    '';

  const apiKeyHeader =
    req.headers[
      'x-api-key'
    ];

  const sent =
    typeof apiKeyHeader ===
    'string'
      ? apiKeyHeader
      : authorization.replace(
          /^Bearer\s+/i,
          ''
        );

  if (
    sent !== API_TOKEN
  ) {

    return res
      .status(401)
      .json({
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

  const normalized =
    path
      .replace(
        /^\/+/,
        ''
      )
      .replace(
        /\\/g,
        '/'
      )
      .trim();

  if (
    !normalized ||
    normalized.includes(
      '..'
    ) ||
    normalized.startsWith(
      '.git/'
    ) ||
    normalized.includes(
      '\0'
    )
  ) {

    throw new Error(
      'Ruta de archivo inválida.'
    );
  }

  return normalized;
}

/* =========================================================
   ENCODE GITHUB PATH
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
   PATH BASE64
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

function decodePathB64(
  value: string
): string {

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
    normalized.length %
      4 !==
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

    'User-Agent':
      'torchill-api',
  };
}

/* =========================================================
   GITHUB GET FILE
   ========================================================= */

async function getGithubFile(
  path: string
) {

  const {
    owner,
    repo,
    branch,
    token,
  } =
    ghConfig();

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
   UPLOAD GITHUB
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
  } =
    ghConfig();

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

  const body: Record<
    string,
    any
  > = {

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

    path:
      safePath,

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
   MIME DESDE URL / HEADER
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
   DOWNLOAD GITHUB FILE
   ========================================================= */

async function downloadGithubFile(
  path: string
): Promise<DownloadedFile> {

  const {
    owner,
    repo,
    branch,
    token,
  } =
    ghConfig();

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
    result.type !==
      'file' ||
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

  const loadingTask =
    pdfjsLib.getDocument({

      data:
        new Uint8Array(
          buffer
        ),

      isEvalSupported:
        false,

      useSystemFonts:
        false,
    });

  const pdf =
    await loadingTask.promise;

  try {

    if (
      pdf.numPages <
      1
    ) {

      throw new Error(
        'El PDF no contiene páginas.'
      );
    }

    const page =
      await pdf.getPage(
        1
      );

    const base =
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
            base.width,
            base.height
          )
      );

    const viewport =
      page.getViewport({
        scale,
      });

    const canvas =
      createCanvas(
        Math.ceil(
          viewport.width
        ),
        Math.ceil(
          viewport.height
        )
      );

    const context =
      canvas.getContext(
        '2d'
      );

    await page.render({

      canvasContext:
        context as any,

      viewport,

    }).promise;

    return canvas.toBuffer(
      'image/png'
    );

  } finally {

    await pdf.destroy();
  }
}

/* =========================================================
   DOWNLOAD RECEIPT FROM URL
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
      fileUrl
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

  const chunks:
    Uint8Array[] = [];

  let total = 0;

  try {

    while (
      true
    ) {

      const {
        done,
        value,
      } =
        await reader.read();

      if (
        done
      ) {
        break;
      }

      if (
        value
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
          value
        );
      }
    }

  } finally {

    reader.releaseLock();
  }

  const buffer =
    Buffer.concat(
      chunks.map(
        (chunk) =>
          Buffer.from(
            chunk
          )
      )
    );

  const mimeType =
    getMimeType(
      fileUrl,
      fileRes.headers.get(
        'content-type'
      ) || ''
    );

  return {
    buffer,
    mimeType,
  };
}

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

      if (
        !pathB64
      ) {

        return res
          .status(400)
          .json({
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

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res
        .status(404)
        .json({

          error:
            'No se pudo obtener el archivo.',

          details:
            message,
        });
    }
  }
);

/* =========================================================
   HEALTH
   ========================================================= */

app.get(
  '/health',
  (
    _req: Request,
    res: Response
  ) => {

    return res
      .status(200)
      .json({

        ok: true,

        service:
          'torchill-api',

        model:
          modelName,

        gemini:
          apiKeys.length >
          0,

        geminiConfigured:
          apiKeys.length >
          0,

        geminiKeys:
          apiKeys.length,

        githubConfigured:
          Boolean(
            process.env.GITHUB_OWNER &&
            process.env.GITHUB_REPO &&
            process.env.GITHUB_TOKEN
          ),

        generateText:
          true,

        node:
          process.version,

        timestamp:
          new Date().toISOString(),
      });
  }
);

/* =========================================================
   GEMINI PROJECT COPY HEALTH
   ========================================================= */

app.get(
  '/api/gemini/project-copy',
  (
    _req: Request,
    res: Response
  ) => {

    return res
      .status(200)
      .json({

        status:
          'ok',

        service:
          'torchill-api',

        model:
          modelName,

        geminiConfigured:
          apiKeys.length >
          0,

        geminiKeys:
          apiKeys.length,

        message:
          'Torchill API funcionando correctamente.',
      });
  }
);

/* =========================================================
   GENERATE TEXT
   =========================================================
   
   NUEVO ENDPOINT
   
   POST /generate-text

   Acepta por ejemplo:

   {
     "prompt": "Escribe una interpretación...",
     "language": "español"
   }

   También acepta:

   {
     "text": "...",
     "prompt": "...",
     "language": "...",
     "context": "...",
     "systemPrompt": "...",
     "temperature": 0.7,
     "maxOutputTokens": 2000
   }

   Respuesta:

   {
     "ok": true,
     "text": "...",
     "model": "...",
     "keyNumber": 1
   }

   ========================================================= */

app.post(
  '/generate-text',
  async (
    req: Request,
    res: Response
  ) => {

    try {

      if (
        apiKeys.length ===
        0
      ) {

        return res
          .status(503)
          .json({

            ok: false,

            error:
              'El servicio Gemini no está configurado.',
          });
      }

      const body =
        getRequestBody(
          req
        );

      const prompt =
        typeof body.prompt ===
        'string'
          ? body.prompt.trim()
          : '';

      const text =
        typeof body.text ===
        'string'
          ? body.text.trim()
          : '';

      const language =
        typeof body.language ===
        'string'
          ? body.language.trim()
          : '';

      const context =
        typeof body.context ===
        'string'
          ? body.context.trim()
          : '';

      const systemPrompt =
        typeof body.systemPrompt ===
        'string'
          ? body.systemPrompt.trim()
          : '';

      /*
       * Permitir que Base44 mande solamente "prompt"
       * o solamente "text".
       */

      const userInput =
        prompt ||
        text;

      if (
        !userInput
      ) {

        return res
          .status(400)
          .json({

            ok: false,

            error:
              'Falta "prompt" o "text".',

            expected:
              '{"prompt":"Texto que debe procesar Gemini"}',
          });
      }

      /*
       * Temperatura segura.
       */

      let temperature =
        Number(
          body.temperature
        );

      if (
        !Number.isFinite(
          temperature
        )
      ) {
        temperature =
          0.7;
      }

      temperature =
        Math.max(
          0,
          Math.min(
            2,
            temperature
          )
        );

      /*
       * Máximo de tokens.
       */

      let maxOutputTokens =
        Number(
          body.maxOutputTokens
        );

      if (
        !Number.isFinite(
          maxOutputTokens
        )
      ) {
        maxOutputTokens =
          4096;
      }

      maxOutputTokens =
        Math.max(
          1,
          Math.min(
            8192,
            Math.floor(
              maxOutputTokens
            )
          )
        );

      /*
       * Construcción del prompt.
       */

      let finalPrompt =
        '';

      if (
        systemPrompt
      ) {

        finalPrompt +=
          `${systemPrompt}\n\n`;
      }

      if (
        language
      ) {

        finalPrompt +=
          `Idioma de respuesta: ${language}\n\n`;
      }

      if (
        context
      ) {

        finalPrompt +=
          `Contexto:\n${context}\n\n`;
      }

      finalPrompt +=
        userInput;

      console.log(
        'Generate text:',
        {
          language:
            language ||
            'no especificado',

          promptLength:
            finalPrompt.length,

          temperature,

          maxOutputTokens,
        }
      );

      /*
       * Gemini.
       */

      const response =
        await runGemini(
          (ai) =>
            ai.models.generateContent(
              {
                model:
                  modelName,

                contents:
                  [
                    {
                      role:
                        'user',

                      parts:
                        [
                          {
                            text:
                              finalPrompt,
                          },
                        ],
                    },
                  ],

                config:
                  {
                    temperature,

                    maxOutputTokens,
                  },
              }
            )
        );

      const generatedText =
        response.text
          ?.trim() ||
        '';

      if (
        !generatedText
      ) {

        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      return res
        .status(200)
        .json({

          ok: true,

          text:
            generatedText,

          response:
            generatedText,

          model:
            modelName,

          timestamp:
            new Date().toISOString(),
        });

    } catch (
      error: unknown
    ) {

      console.error(
        'Generate text error:',
        error
      );

      if (
        isRateLimitError(
          error
        )
      ) {

        return res
          .status(429)
          .json({

            ok: false,

            error:
              'Todas las API keys de Gemini están temporalmente limitadas.',

            retryable:
              true,

            retryAfter:
              getRetryAfterMs(
                error
              ),
          });
      }

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res
        .status(500)
        .json({

          ok: false,

          error:
            'Error generando el texto con Gemini.',

          details:
            message,
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
  upload.single(
    'file'
  ),
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

      if (
        !buffer
      ) {

        const dataBase64 =
          typeof body.dataBase64 ===
          'string'
            ? body.dataBase64
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

            return res
              .status(400)
              .json({

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

      if (
        !buffer
      ) {

        return res
          .status(400)
          .json({

            error:
              'Falta el archivo. Usá multipart/form-data con campo "file" o JSON con "dataBase64".',
          });
      }

      if (
        buffer.length >
        MAX_UPLOAD_SIZE
      ) {

        return res
          .status(413)
          .json({

            error:
              'El archivo supera el límite máximo de 20 MB.',
          });
      }

      /* =====================================================
         EXTENSION
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
        requestedExt
      ) {

        finalExtension =
          requestedExt;

      } else if (
        originalName.includes(
          '.'
        )
      ) {

        finalExtension =
          originalName
            .split('.')
            .pop()
            ?.toLowerCase() ||
          '';
      }

      /* =====================================================
         DETECTAR PDF REAL
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

      if (
        isPdf
      ) {

        finalExtension =
          'pdf';

        mimeType =
          'application/pdf';
      }

      if (
        !finalExtension
      ) {

        finalExtension =
          'jpg';
      }

      if (
        finalExtension ===
          'jpeg' ||
        finalExtension ===
          'jpe'
      ) {

        finalExtension =
          'jpg';
      }

      /* =====================================================
         EXTENSIONES PERMITIDAS
         ===================================================== */

      const allowedExtensions =
        new Set([
          'pdf',
          'png',
          'jpg',
          'jpeg',
          'webp',
          'gif',
          'svg',
        ]);

      if (
        !allowedExtensions.has(
          finalExtension
        )
      ) {

        return res
          .status(400)
          .json({

            error:
              `Tipo de archivo no permitido: .${finalExtension}`,
          });
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
        {
          originalName,
          mimeType,
          finalExtension,
          size:
            buffer.length,
          path:
            requestedPath,
          pdfDetected:
            isPdf,
        }
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
        req.get(
          'host'
        );

      const protocol =
        req.headers[
          'x-forwarded-proto'
        ]
          ?.toString()
          .split(',')[0]
          .trim() ||
        req.protocol;

      const absoluteUrl =
        host
          ? `${protocol}://${host}${receiptUrl}`
          : receiptUrl;

      /* =====================================================
         RESPONSE
         ===================================================== */

      return res
        .status(200)
        .json({

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
        error instanceof Error
          ? error.message
          : String(error);

      return res
        .status(500)
        .json({

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

        return res
          .status(503)
          .json({

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
        ).length ===
        0
      ) {

        return res
          .status(400)
          .json({

            error:
              'La solicitud no contiene un body JSON válido.',

            expected:
              '{"title":"...","text":"..."}',
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

          return res
            .status(400)
            .json({

              error:
                '"segments" debe ser un array.',
            });
        }

        if (
          segments.length ===
          0
        ) {

          return res
            .status(400)
            .json({

              error:
                '"segments" no puede estar vacío.',
            });
        }

        const targetLanguage =
          language === 'en'
            ? 'inglés'
            : 'español';

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

${JSON.stringify(
  segments
)}

Devuelve exclusivamente JSON válido con esta estructura:

{
  "translations": [
    "texto traducido 1",
    "texto traducido 2"
  ]
}
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

                      schema:
                        {
                          type:
                            'object',

                          properties:
                            {
                              translations:
                                {
                                  type:
                                    'array',

                                  items:
                                    {
                                      type:
                                        'string',
                                    },
                                },
                            },

                          required:
                            [
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
          .json(
            result
          );
      }

      /* =====================================================
         PROJECT
         ===================================================== */

      if (
        typeof title !==
          'string' ||
        title.trim()
          .length ===
          0
      ) {

        return res
          .status(400)
          .json({

            error:
              'Falta el campo "title".',
          });
      }

      if (
        typeof text !==
          'string' ||
        text.trim()
          .length ===
          0
      ) {

        return res
          .status(400)
          .json({

            error:
              'Falta el campo "text".',
          });
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
  "lead": "string",
  "discipline": "string",
  "sections": [
    {
      "title": "string",
      "summary": "string"
    }
  ],
  "imageAlts": [
    "string"
  ]
}
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

                    schema:
                      {
                        type:
                          'object',

                        properties:
                          {
                            lead:
                              {
                                type:
                                  'string',
                              },

                            discipline:
                              {
                                type:
                                  'string',
                              },

                            sections:
                              {
                                type:
                                  'array',

                                items:
                                  {
                                    type:
                                      'object',

                                    properties:
                                      {
                                        title:
                                          {
                                            type:
                                              'string',
                                          },

                                        summary:
                                          {
                                            type:
                                              'string',
                                          },
                                      },

                                    required:
                                      [
                                        'title',
                                        'summary',
                                      ],
                                  },
                              },

                            imageAlts:
                              {
                                type:
                                  'array',

                                items:
                                  {
                                    type:
                                      'string',
                                  },
                              },
                          },

                        required:
                          [
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
        .json(
          result
        );

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

        return res
          .status(429)
          .json({

            error:
              'Todas las API keys de Gemini están temporalmente limitadas.',

            retryable:
              true,

            retryAfter:
              getRetryAfterMs(
                error
              ),
          });
      }

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res
        .status(500)
        .json({

          error:
            'Error procesando la solicitud con Gemini.',

          details:
            message,
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

        return res
          .status(503)
          .json({

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
          .length ===
          0
      ) {

        return res
          .status(400)
          .json({

            error:
              'fileUrl es requerido.',
          });
      }

      if (
        typeof prompt !==
          'string' ||
        prompt.trim()
          .length ===
          0
      ) {

        return res
          .status(400)
          .json({

            error:
              'prompt es requerido.',
          });
      }

      /* ===================================================
         DOWNLOAD
         =================================================== */

      let {
        buffer,
        mimeType,
      } =
        await downloadReceipt(
          fileUrl
        );

      console.log(
        `Receipt: archivo descargado (${buffer.length} bytes, ${mimeType})`
      );

      /* ===================================================
         PDF
         =================================================== */

      if (
        mimeType ===
        'application/pdf'
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
          pdfError
        ) {

          console.error(
            'Receipt PDF conversion error:',
            pdfError
          );

          return res
            .status(422)
            .json({

              error:
                'No se pudo convertir el PDF a imagen.',

              details:
                pdfError instanceof
                Error
                  ? pdfError.message
                  : String(
                      pdfError
                    ),
            });
        }
      }

      /* ===================================================
         GEMINI
         =================================================== */

      const base64 =
        buffer.toString(
          'base64'
        );

      const geminiPrompt = `
${prompt}

IMPORTANTE:
- Analiza cuidadosamente el comprobante.
- Extrae únicamente información visible o claramente inferible.
- No inventes datos.
- Si un campo no puede determinarse, utiliza null.
- Devuelve ÚNICAMENTE JSON válido.
`;

      const response =
        await runGemini(
          (ai) =>
            ai.models.generateContent(
              {
                model:
                  modelName,

                contents:
                  [
                    {
                      role:
                        'user',

                      parts:
                        [
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

                config:
                  {
                    responseMimeType:
                      'application/json',

                    temperature:
                      0.1,
                  },
              }
            )
        );

      const text =
        response.text
          ?.trim() ||
        '';

      if (
        !text
      ) {

        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      /* ===================================================
         JSON
         =================================================== */

      const parsed =
        parseGeminiJson(
          text
        );

      return res
        .status(200)
        .json({

          ok: true,

          detected:
            parsed,

          raw:
            text,
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

        return res
          .status(429)
          .json({

            error:
              'Todas las API keys de Gemini están temporalmente limitadas.',

            retryable:
              true,

            retryAfter:
              getRetryAfterMs(
                error
              ),
          });
      }

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res
        .status(500)
        .json({

          error:
            'Error analizando el comprobante.',

          details:
            message,
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
      error instanceof
      multer.MulterError
    ) {

      if (
        error.code ===
        'LIMIT_FILE_SIZE'
      ) {

        return res
          .status(413)
          .json({

            error:
              'El archivo supera el límite máximo de 20 MB.',
          });
      }

      return res
        .status(400)
        .json({

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

    return res
      .status(404)
      .json({

        error:
          'Endpoint no encontrado.',
      });
  }
);

/* =========================================================
   ERROR GENERAL
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

    const message =
      error instanceof Error
        ? error.message
        : String(error);

    return res
      .status(500)
      .json({

        error:
          'Error interno del servidor.',

        details:
          message,
      });
  }
);

/* =========================================================
   START
   ========================================================= */

const server =
  app.listen(
    PORT,
    '0.0.0.0',
    () => {

      console.log(
        '========================================'
      );

      console.log(
        'Torchill API iniciada correctamente'
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
        '----------------------------------------'
      );

      console.log(
        'GET  /health'
      );

      console.log(
        'GET  /api/gemini/project-copy'
      );

      console.log(
        'POST /api/gemini/project-copy'
      );

      console.log(
        'POST /generate-text'
      );

      console.log(
        'POST /upload'
      );

      console.log(
        'GET  /receipt/:pathB64'
      );

      console.log(
        'POST /analyze-receipt'
      );

      console.log(
        '========================================'
      );
    }
  );

/* =========================================================
   SHUTDOWN
   ========================================================= */

function shutdown(
  signal: string
) {

  console.log(
    `${signal}: cerrando servidor...`
  );

  server.close(
    () => {

      console.log(
        'Servidor cerrado correctamente.'
      );

      process.exit(
        0
      );
    }
  );

  setTimeout(
    () => {

      console.error(
        'Forzando cierre del servidor.'
      );

      process.exit(
        1
      );

    },
    10_000
  ).unref();
}

process.on(
  'SIGTERM',
  () =>
    shutdown(
      'SIGTERM'
    )
);

process.on(
  'SIGINT',
  () =>
    shutdown(
      'SIGINT'
    )
);
