import express, {
  Request,
  Response,
  NextFunction,
} from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import multer from 'multer';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.js';
import { createCanvas } from '@napi-rs/canvas';
import crypto from 'node:crypto';

const app = express();

/* =========================================================
   CONFIGURACIÓN GENERAL
   ========================================================= */

const PORT =
  Number(process.env.PORT) || 3000;

const modelName =
  process.env.GEMINI_MODEL ||
  'gemini-3-flash-preview';

const API_TOKEN =
  process.env.API_TOKEN;

/*
 * Tamaño máximo permitido para archivos.
 */
const MAX_UPLOAD_SIZE =
  20 * 1024 * 1024;

const MAX_RECEIPT_FILE_SIZE =
  MAX_UPLOAD_SIZE;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_UPLOAD_SIZE,
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

app.options(
  '/{*splat}',
  cors()
);

/* =========================================================
   BODY PARSER
   ========================================================= */

app.use(
  express.json({
    limit: '1mb',
  })
);

app.use(
  express.text({
    type: ['text/plain'],
    limit: '1mb',
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
].filter(
  (
    key
  ): key is string =>
    typeof key === 'string' &&
    key.trim().length > 0
);

if (apiKeys.length === 0) {
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

interface KeyState {
  ai: GoogleGenAI;
  blockedUntil: number;
}

const keyStates: KeyState[] =
  apiKeys.map((key) => ({
    ai: new GoogleGenAI({
      apiKey: key,
    }),
    blockedUntil: 0,
  }));

let currentKeyIndex = 0;

/* =========================================================
   OBTENER KEY DISPONIBLE
   ========================================================= */

function getAvailableKey():
  KeyState | null {
  if (keyStates.length === 0) {
    return null;
  }

  const now = Date.now();

  for (
    let i = 0;
    i < keyStates.length;
    i++
  ) {
    const index =
      (currentKeyIndex + i) %
      keyStates.length;

    const keyState =
      keyStates[index];

    if (
      keyState.blockedUntil <= now
    ) {
      currentKeyIndex =
        (index + 1) %
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
    Date.now() + retryAfterMs;
}

/* =========================================================
   OBTENER RETRY-AFTER
   ========================================================= */

function getRetryAfterMs(
  error: unknown
): number {
  const message =
    error instanceof Error
      ? error.message
      : String(error);

  const match =
    message.match(
      /retry in\s+([\d.]+)s/i
    );

  if (match) {
    const seconds =
      Number.parseFloat(
        match[1]
      );

    if (
      Number.isFinite(seconds) &&
      seconds > 0
    ) {
      return Math.ceil(
        seconds * 1000
      );
    }
  }

  return 30_000;
}

/* =========================================================
   DETECTAR RATE LIMIT
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
    normalized.includes('429') ||
    normalized.includes(
      'too_many_requests'
    ) ||
    normalized.includes(
      'ratelimiterror'
    ) ||
    normalized.includes(
      'rate limit'
    ) ||
    normalized.includes(
      'quota exceeded'
    ) ||
    normalized.includes(
      'resource exhausted'
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
  if (keyStates.length === 0) {
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

    const keyNumber =
      keyStates.indexOf(
        keyState
      ) + 1;

    try {
      console.log(
        `Gemini: usando API key ${keyNumber}`
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
        !isRateLimitError(error)
      ) {
        throw error;
      }

      const retryAfterMs =
        getRetryAfterMs(
          error
        );

      const retryAfterSeconds =
        Math.ceil(
          retryAfterMs / 1000
        );

      console.warn(
        `Gemini: API key ${keyNumber} alcanzó el límite. ` +
        `Bloqueando durante ${retryAfterSeconds}s.`
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
    typeof body === 'string'
  ) {
    if (
      body.trim().length === 0
    ) {
      return {};
    }

    try {
      body =
        JSON.parse(body);
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

  return body as Record<
    string,
    any
  >;
}

/* =========================================================
   PARSEAR JSON DE GEMINI
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
  } catch {
    const objectMatch =
      cleaned.match(
        /\{[\s\S]*\}/
      );

    if (objectMatch) {
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

    if (arrayMatch) {
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
}

/* =========================================================
   AUTH
   ========================================================= */

function auth(
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (!API_TOKEN) {
    return next();
  }

  const authorization =
    req.headers.authorization ||
    '';

  const sent =
    req.headers[
      'x-api-key'
    ] ||
    authorization.replace(
      /^Bearer\s+/i,
      ''
    );

  if (
    sent !== API_TOKEN
  ) {
    return res.status(401).json({
      error: 'no autorizado',
    });
  }

  return next();
}

/* =========================================================
   GITHUB STORAGE
   ========================================================= */

function ghConfig() {
  const owner =
    process.env.GITHUB_OWNER;

  const repo =
    process.env.GITHUB_REPO;

  const branch =
    process.env.GITHUB_BRANCH ||
    'main';

  const token =
    process.env.GITHUB_TOKEN;

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
   VALIDAR PATH DE GITHUB
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
      );

  if (
    !normalized ||
    normalized.includes('..') ||
    normalized.startsWith(
      '.git/'
    )
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
    normalized.length % 4 !==
    0
  ) {
    normalized += '=';
  }

  return validateGithubPath(
    Buffer.from(
      normalized,
      'base64'
    ).toString(
      'utf8'
    )
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
   OBTENER ARCHIVO DE GITHUB
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

  const encodedPath =
    encodePathForGithub(
      path
    );

  const url =
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/${encodedPath}` +
    `?ref=${encodeURIComponent(branch)}`;

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

  if (!response.ok) {
    if (
      response.status ===
      404
    ) {
      return null;
    }

    const text =
      await response.text();

    throw new Error(
      `GitHub GET error ${response.status}: ${text}`
    );
  }

  return response.json();
}

/* =========================================================
   SUBIR ARCHIVO A GITHUB
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
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/${encodePathForGithub(safePath)}`;

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

  if (!response.ok) {
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
   MIME TYPE
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

  if (
    lower.endsWith('.bmp')
  ) {
    return 'image/bmp';
  }

  return 'application/octet-stream';
}

/* =========================================================
   DETECTAR MIME TYPE DESDE URL / HEADER
   ========================================================= */

function getMimeType(
  fileUrl: string,
  contentType: string
): string {
  const normalizedContentType =
    contentType
      .split(';')[0]
      .trim()
      .toLowerCase();

  if (
    normalizedContentType &&
    normalizedContentType !==
      'application/octet-stream'
  ) {
    return normalizedContentType;
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
   DESCARGAR ARCHIVO DESDE GITHUB
   ========================================================= */

async function downloadGithubFile(
  path: string
): Promise<{
  buffer: Buffer;
  mimeType: string;
}> {
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
    `https://api.github.com/repos/${encodeURIComponent(owner)}` +
    `/${encodeURIComponent(repo)}/contents/${encodedPath}` +
    `?ref=${encodeURIComponent(branch)}`;

  /*
   * Pedimos el archivo como contenido RAW.
   *
   * Esto evita depender del campo "content" de la API
   * de GitHub, que puede no estar disponible para archivos
   * grandes.
   */

  const response =
    await fetch(
      url,
      {
        headers: {
          Authorization:
            `Bearer ${token}`,
          Accept:
            'application/vnd.github.raw',
          'X-GitHub-Api-Version':
            '2022-11-28',
        },
      }
    );

  if (!response.ok) {
    if (
      response.status ===
      404
    ) {
      throw new Error(
        'Archivo no encontrado en GitHub.'
      );
    }

    const text =
      await response.text();

    throw new Error(
      `GitHub download error ${response.status}: ${text}`
    );
  }

  const arrayBuffer =
    await response.arrayBuffer();

  const buffer =
    Buffer.from(
      arrayBuffer
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
      isEvalSupported: false,
      useSystemFonts: false,
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
   DESCARGAR RECEIPT DESDE URL
   ========================================================= */

async function downloadReceipt(
  fileUrl: string
): Promise<{
  buffer: Buffer;
  mimeType: string;
}> {
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

  if (!fileRes.ok) {
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

  if (!fileRes.body) {
    throw new Error(
      'La respuesta no contiene datos.'
    );
  }

  const reader =
    fileRes.body.getReader();

  const chunks: Uint8Array[] =
    [];

  let total = 0;

  try {
    while (true) {
      const {
        done,
        value,
      } = await reader.read();

      if (done) {
        break;
      }

      if (value) {
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
   GET RECEIPT DESDE GITHUB
   ========================================================= */

app.get(
  '/receipt/:pathB64',
  async (
    req: Request,
    res: Response
  ) => {
    try {
      const path =
        decodePathB64(
          req.params.pathB64
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

      return res.status(404).json({
        error:
          'No se pudo obtener el archivo.',
        details:
          message,
      });
    }
  }
);

/* =========================================================
   HEALTH CHECK
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
   HEALTH CHECK PROJECT COPY
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
        getRequestBody(req);

      /*
       * También acepta JSON:
       *
       * {
       *   dataBase64: "...",
       *   fileName: "ticket.pdf",
       *   mimeType: "application/pdf"
       * }
       */

      if (!buffer) {
        const dataBase64 =
          typeof body.dataBase64 ===
          'string'
            ? body.dataBase64
            : '';

        if (dataBase64) {
          const clean =
            dataBase64.replace(
              /^data:[^;]+;base64,/i,
              ''
            );

          buffer =
            Buffer.from(
              clean,
              'base64'
            );

          originalName =
            typeof body.fileName ===
            'string'
              ? body.fileName
              : 'receipt';

          mimeType =
            typeof body.mimeType ===
            'string'
              ? body.mimeType
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

      const safeName =
        String(
          originalName ||
            'receipt'
        ).replace(
          /[^a-zA-Z0-9._-]/g,
          '_'
        );

      const extension =
        safeName.includes('.')
          ? safeName
              .split('.')
              .pop()
              ?.toLowerCase()
          : '';

      const finalExtension =
        extension ||
        (
          mimeType ===
          'application/pdf'
            ? 'pdf'
            : mimeType ===
                'image/png'
              ? 'png'
              : mimeType ===
                  'image/webp'
                ? 'webp'
                : mimeType ===
                    'image/gif'
                  ? 'gif'
                  : 'jpg'
        );

      let requestedPath =
        typeof body.path ===
        'string'
          ? body.path
          : '';

      if (!requestedPath) {
        requestedPath =
          `receipts/${Date.now()}-${crypto.randomUUID()}.${finalExtension}`;
      }

      requestedPath =
        validateGithubPath(
          requestedPath
        );

      const result =
        await uploadToGithub(
          requestedPath,
          buffer
        );

      const receiptUrl =
        `/receipt/${result.pathB64}`;

      return res.status(200).json({
        ok: true,
        ...result,
        mimeType:
          mimeType ||
          mimeForPath(
            result.path
          ),
        size:
          buffer.length,
        url:
          receiptUrl,
        receiptUrl,
        absoluteUrl:
          `${req.protocol}://${req.get('host')}${receiptUrl}`,
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
        apiKeys.length === 0
      ) {
        return res.status(503).json({
          error:
            'El servicio Gemini no está configurado.',
        });
      }

      const body =
        getRequestBody(req);

      const {
        action,
        text,
        language,
        title,
        segments,
      } = body;

      if (
        Object.keys(body).length ===
        0
      ) {
        return res.status(400).json({
          error:
            'La solicitud no contiene un body JSON válido.',
          expected:
            'Para proyectos: {"title":"...","text":"..."}',
        });
      }

      /* =====================================================
         TRADUCCIÓN
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
          segments.length ===
          0
        ) {
          return res.status(400).json({
            error:
              '"segments" no puede estar vacío.',
          });
        }

        const targetLanguage =
          language === 'en'
            ? 'inglés'
            : 'español';

        const prompt = `
Actúa como traductor profesional especializado
en diseño gráfico y comunicación visual.

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

Devuelve exclusivamente un JSON válido con esta estructura:

{
  "translations": ["texto traducido 1", "texto traducido 2"]
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
                  response_format: {
                    type: 'text',
                    mime_type:
                      'application/json',
                    schema: {
                      type: 'object',
                      properties: {
                        translations:
                          {
                            type: 'array',
                            items: {
                              type: 'string',
                            },
                          },
                      },
                      required: [
                        'translations',
                      ],
                    },
                  },
                }
              )
          );

        const output =
          interaction.output_text?.trim();

        if (!output) {
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
         PROYECTO / PORTFOLIO
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
- Mantener el contenido apropiado
  para un portfolio de diseño.

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
  "imageAlts": ["string"]
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
                response_format: {
                  type: 'text',
                  mime_type:
                    'application/json',
                  schema: {
                    type: 'object',
                    properties: {
                      lead: {
                        type: 'string',
                      },
                      discipline: {
                        type: 'string',
                      },
                      sections: {
                        type: 'array',
                        items: {
                          type: 'object',
                          properties: {
                            title: {
                              type: 'string',
                            },
                            summary: {
                              type: 'string',
                            },
                          },
                          required: [
                            'title',
                            'summary',
                          ],
                        },
                      },
                      imageAlts: {
                        type: 'array',
                        items: {
                          type: 'string',
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
              }
            )
        );

      const output =
        interaction.output_text
          ?.trim();

      if (!output) {
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
          retryable: true,
        });
      }

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res.status(500).json({
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
        return res.status(503).json({
          error:
            'El servicio Gemini no está configurado.',
        });
      }

      const body =
        getRequestBody(req);

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

      /* ===================================================
         1. DESCARGAR ARCHIVO
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
         2. PDF -> PNG
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

          buffer = png;
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

          return res.status(422).json({
            error:
              'No se pudo convertir el PDF a imagen.',
            details:
              pdfError instanceof Error
                ? pdfError.message
                : String(
                    pdfError
                  ),
          });
        }
      }

      /* ===================================================
         3. VALIDAR TIPO DE ARCHIVO
         =================================================== */

      const allowedMimeTypes = [
        'image/jpeg',
        'image/png',
        'image/webp',
        'image/gif',
        'image/bmp',
      ];

      if (
        !allowedMimeTypes.includes(
          mimeType
        )
      ) {
        return res.status(415).json({
          error:
            'El archivo debe ser un PDF o una imagen compatible.',
          mimeType,
        });
      }

      /* ===================================================
         4. BASE64
         =================================================== */

      const base64 =
        buffer.toString(
          'base64'
        );

      /* ===================================================
         5. ANALIZAR CON GEMINI
         =================================================== */

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
                contents: [
                  {
                    role: 'user',
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
                  temperature: 0.1,
                },
              }
            )
        );

      const text =
        response.text?.trim() ||
        '';

      if (!text) {
        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      /* ===================================================
         6. PARSEAR JSON
         =================================================== */

      const parsed =
        parseGeminiJson(
          text
        );

      return res.status(200).json({
        ok: true,
        detected:
          parsed,
        raw: text,
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
          retryable: true,
        });
      }

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res.status(500).json({
        error:
          'Error analizando el comprobante.',
        details:
          message,
      });
    }
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
   ERROR HANDLER
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

    return res.status(500).json({
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

app.listen(
  PORT,
  () => {
    console.log(
      `Servidor corriendo en puerto ${PORT}`
    );

    console.log(
      `Modelo Gemini: ${modelName}`
    );

    console.log(
      `Gemini API keys disponibles: ${apiKeys.length}`
    );

    console.log(
      `Analyze receipt: /analyze-receipt`
    );

    console.log(
      `Upload: /upload`
    );

    console.log(
      `Receipt: /receipt/:pathB64`
    );
  }
);
