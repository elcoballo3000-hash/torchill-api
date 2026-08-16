import express, {
  Request,
  Response,
  NextFunction,
} from 'express';

import cors from 'cors';
import multer from 'multer';
import crypto from 'node:crypto';

import {
  GoogleGenAI,
} from '@google/genai';

import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.js';
import * as napiCanvas from '@napi-rs/canvas';

/* =========================================================
   TIPOS
   ========================================================= */

interface KeyState {
  ai: GoogleGenAI;
  blockedUntil: number;
  keyNumber: number;
}

interface GithubConfig {
  owner: string;
  repo: string;
  branch: string;
  token: string;
}

interface DownloadedFile {
  buffer: Buffer;
  mimeType: string;
}

interface TarotCache {
  files: string[];
  timestamp: number;
}

/* =========================================================
   POLYFILLS PARA PDF.JS
   ========================================================= */

const canvasAny =
  napiCanvas as any;

if (
  !(globalThis as any).DOMMatrix &&
  canvasAny.DOMMatrix
) {
  (globalThis as any).DOMMatrix =
    canvasAny.DOMMatrix;
}

if (
  !(globalThis as any).ImageData &&
  canvasAny.ImageData
) {
  (globalThis as any).ImageData =
    canvasAny.ImageData;
}

if (
  !(globalThis as any).Path2D &&
  canvasAny.Path2D
) {
  (globalThis as any).Path2D =
    canvasAny.Path2D;
}

/* =========================================================
   EXPRESS
   ========================================================= */

const app = express();

app.set(
  'trust proxy',
  1
);

/* =========================================================
   CONFIG
   ========================================================= */

const PORT =
  Number(
    process.env.PORT
  ) || 3000;

const MODEL_NAME =
  process.env.GEMINI_MODEL?.trim() ||
  'gemini-3-flash-preview';

const API_TOKEN =
  process.env.API_TOKEN?.trim() ||
  '';

const MAX_UPLOAD_SIZE =
  20 * 1024 * 1024;

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

/* =========================================================
   BODY PARSERS
   ========================================================= */

app.use(
  express.json({
    limit: '60mb',
  })
);

app.use(
  express.text({
    type: 'text/plain',
    limit: '25mb',
  })
);

/* =========================================================
   MULTER
   ========================================================= */

const upload =
  multer({
    storage:
      multer.memoryStorage(),

    limits: {
      fileSize:
        MAX_UPLOAD_SIZE,
    },
  });

/* =========================================================
   HELPERS
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
    Array.isArray(
      value
    )
  ) {
    return value[0] || '';
  }

  return '';
}

function getRequestBody(
  req: Request
): Record<string, any> {
  let body: unknown =
    req.body;

  if (
    typeof body ===
    'string'
  ) {
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
    Array.isArray(
      body
    )
  ) {
    return {};
  }

  return body as Record<
    string,
    any
  >;
}

/* =========================================================
   AUTH
   ========================================================= */

function auth(
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (
    !API_TOKEN
  ) {
    return next();
  }

  const headerKey =
    req.headers[
      'x-api-key'
    ];

  const authorization =
    req.headers
      .authorization ||
    '';

  const sent =
    typeof headerKey ===
    'string'
      ? headerKey
      : authorization.replace(
          /^Bearer\s+/i,
          ''
        );

  if (
    sent !==
    API_TOKEN
  ) {
    return res
      .status(401)
      .json({
        ok: false,
        error:
          'no autorizado',
      });
  }

  return next();
}

/* =========================================================
   GEMINI KEYS
   ========================================================= */

const rawApiKeys = [
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
];

const apiKeys =
  [
    ...new Set(
      rawApiKeys
        .map(
          (key) =>
            typeof key ===
            'string'
              ? key.trim()
              : ''
        )
        .filter(
          (
            key
          ): key is string =>
            key.length > 0
        )
    ),
  ];

const keyStates:
  KeyState[] =
  apiKeys.map(
    (
      key,
      index
    ) => ({
      ai:
        new GoogleGenAI({
          apiKey:
            key,
        }),

      blockedUntil:
        0,

      keyNumber:
        index + 1,
    })
  );

let currentKeyIndex =
  0;

if (
  keyStates.length ===
  0
) {
  console.warn(
    'ADVERTENCIA: no hay API keys de Gemini configuradas.'
  );
} else {
  console.log(
    `Gemini: ${keyStates.length} API keys configuradas.`
  );
}

/* =========================================================
   GEMINI RATE LIMIT
   ========================================================= */

function isRateLimitError(
  error: unknown
): boolean {
  const message =
    error instanceof Error
      ? error.message
      : String(
          error
        );

  const normalized =
    message.toLowerCase();

  return (
    normalized.includes(
      '429'
    ) ||
    normalized.includes(
      'rate limit'
    ) ||
    normalized.includes(
      'too_many_requests'
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

function getRetryAfterMs(
  error: unknown
): number {
  const message =
    error instanceof Error
      ? error.message
      : String(
          error
        );

  const patterns = [
    /retry in\s+([\d.]+)s/i,
    /retryDelay[^0-9]*([\d.]+)s/i,
    /retry-after[^0-9]*([\d.]+)/i,
  ];

  for (
    const pattern of
    patterns
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
          seconds *
            1000
        );
      }
    }
  }

  return 30_000;
}

function getAvailableKey():
  KeyState | null {
  if (
    keyStates.length ===
    0
  ) {
    return null;
  }

  const now =
    Date.now();

  for (
    let i = 0;
    i <
    keyStates.length;
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

async function runGemini<T>(
  operation: (
    ai: GoogleGenAI
  ) => Promise<T>
): Promise<T> {
  if (
    keyStates.length ===
    0
  ) {
    throw new Error(
      'No hay ninguna API key de Gemini configurada.'
    );
  }

  const attempted =
    new Set<number>();

  for (
    let i = 0;
    i <
    keyStates.length;
    i++
  ) {
    const keyState =
      getAvailableKey();

    if (
      !keyState
    ) {
      break;
    }

    if (
      attempted.has(
        keyState.keyNumber
      )
    ) {
      break;
    }

    attempted.add(
      keyState.keyNumber
    );

    try {
      console.log(
        `Gemini: usando API key ${keyState.keyNumber}`
      );

      return await operation(
        keyState.ai
      );
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

      const retry =
        getRetryAfterMs(
          error
        );

      keyState.blockedUntil =
        Date.now() +
        retry;

      console.warn(
        `Gemini key ${keyState.keyNumber} limitada durante ${Math.ceil(
          retry / 1000
        )} segundos.`
      );
    }
  }

  throw new Error(
    'Todas las API keys de Gemini están temporalmente limitadas.'
  );
}

/* =========================================================
   GITHUB
   ========================================================= */

function ghConfig():
  GithubConfig {
  const owner =
    process.env
      .GITHUB_OWNER
      ?.trim();

  const repo =
    process.env
      .GITHUB_REPO
      ?.trim();

  const branch =
    process.env
      .GITHUB_BRANCH
      ?.trim() ||
    'main';

  const token =
    process.env
      .GITHUB_TOKEN
      ?.trim();

  if (
    !owner ||
    !repo ||
    !token
  ) {
    throw new Error(
      'Faltan GITHUB_OWNER, GITHUB_REPO o GITHUB_TOKEN.'
    );
  }

  return {
    owner,
    repo,
    branch,
    token,
  };
}

function githubHeaders(
  token: string
): Record<
  string,
  string
> {
  return {
    Authorization:
      `Bearer ${token}`,

    Accept:
      'application/vnd.github+json',

    'X-GitHub-Api-Version':
      '2022-11-28',

    'User-Agent':
      'torchill-api',

    'Content-Type':
      'application/json',
  };
}

/* =========================================================
   GITHUB PATH
   ========================================================= */

function validateGithubPath(
  path: string
): string {
  const result =
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
    !result ||
    result.includes(
      '..'
    ) ||
    result.includes(
      '\0'
    )
  ) {
    throw new Error(
      'Ruta de GitHub inválida.'
    );
  }

  return result;
}

function encodeGithubPath(
  path: string
): string {
  return path
    .split('/')
    .map(
      (segment) =>
        encodeURIComponent(
          segment
        )
    )
    .join('/');
}

/* =========================================================
   BASE64 URL PATH
   ========================================================= */

function encodePathB64(
  path: string
): string {
  return Buffer.from(
    path,
    'utf8'
  ).toString(
    'base64url'
  );
}

function decodePathB64(
  path: string
): string {
  return validateGithubPath(
    Buffer.from(
      path,
      'base64url'
    ).toString(
      'utf8'
    )
  );
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
    lower.endsWith(
      '.pdf'
    )
  ) {
    return 'application/pdf';
  }

  if (
    lower.endsWith(
      '.png'
    )
  ) {
    return 'image/png';
  }

  if (
    lower.endsWith(
      '.webp'
    )
  ) {
    return 'image/webp';
  }

  if (
    lower.endsWith(
      '.gif'
    )
  ) {
    return 'image/gif';
  }

  if (
    lower.endsWith(
      '.svg'
    )
  ) {
    return 'image/svg+xml';
  }

  if (
    lower.endsWith(
      '.jpg'
    ) ||
    lower.endsWith(
      '.jpeg'
    )
  ) {
    return 'image/jpeg';
  }

  return 'application/octet-stream';
}

/* =========================================================
   SUBIR ARCHIVO A GITHUB
   ========================================================= */

async function uploadToGithub(
  path: string,
  buffer: Buffer
) {
  if (
    buffer.length >
    MAX_UPLOAD_SIZE
  ) {
    throw new Error(
      'El archivo supera los 20 MB.'
    );
  }

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

  const url =
    `https://api.github.com/repos/${encodeURIComponent(
      owner
    )}/${encodeURIComponent(
      repo
    )}/contents/${encodeGithubPath(
      safePath
    )}`;

  const existingResponse =
    await fetch(
      `${url}?ref=${encodeURIComponent(
        branch
      )}`,
      {
        headers:
          githubHeaders(
            token
          ),
      }
    );

  let existingSha:
    string | undefined;

  if (
    existingResponse.ok
  ) {
    const existing =
      await existingResponse.json();

    if (
      existing &&
      typeof existing.sha ===
        'string'
    ) {
      existingSha =
        existing.sha;
    }
  }

  const body:
    Record<
      string,
      any
    > = {
    message:
      `Torchill upload: ${safePath}`,

    content:
      buffer.toString(
        'base64'
      ),

    branch,
  };

  if (
    existingSha
  ) {
    body.sha =
      existingSha;
  }

  const response =
    await fetch(
      url,
      {
        method:
          'PUT',

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
      result
        ?.content
        ?.sha ||
      null,

    pathB64:
      encodePathB64(
        safePath
      ),
  };
}

/* =========================================================
   DESCARGAR ARCHIVO PRIVADO DE GITHUB
   ========================================================= */

async function downloadGithubFile(
  path: string
): Promise<
  DownloadedFile
> {
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

  const url =
    `https://api.github.com/repos/${encodeURIComponent(
      owner
    )}/${encodeURIComponent(
      repo
    )}/contents/${encodeGithubPath(
      safePath
    )}?ref=${encodeURIComponent(
      branch
    )}`;

  const response =
    await fetch(
      url,
      {
        headers: {
          ...githubHeaders(
            token
          ),

          Accept:
            'application/vnd.github.raw',
        },
      }
    );

  if (
    !response.ok
  ) {
    throw new Error(
      `No se pudo descargar ${safePath} (${response.status}).`
    );
  }

  const buffer =
    Buffer.from(
      await response.arrayBuffer()
    );

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
): Promise<
  Buffer | null
> {
  let pdf:
    any = null;

  try {
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

    pdf =
      await loadingTask.promise;

    if (
      pdf.numPages <
      1
    ) {
      return null;
    }

    const page =
      await pdf.getPage(
        1
      );

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

    const canvas =
      canvasAny.createCanvas(
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

    const canvasFactory =
      {
        create(
          width: number,
          height: number
        ) {
          const canvas =
            canvasAny.createCanvas(
              width,
              height
            );

          return {
            canvas,
            context:
              canvas.getContext(
                '2d'
              ),
          };
        },

        reset(
          canvasAndContext:
            any,
          width: number,
          height: number
        ) {
          canvasAndContext
            .canvas.width =
            width;

          canvasAndContext
            .canvas.height =
            height;
        },

        destroy(
          canvasAndContext:
            any
        ) {
          canvasAndContext
            .canvas.width =
            0;

          canvasAndContext
            .canvas.height =
            0;

          canvasAndContext
            .canvas =
            null;

          canvasAndContext
            .context =
            null;
        },
      };

    await page.render({
      canvasContext:
        context,

      viewport,

      canvasFactory,
    }).promise;

    return canvas.toBuffer(
      'image/png'
    );
  } catch (
    error
  ) {
    console.error(
      'PDF -> PNG error:',
      error
    );

    return null;
  } finally {
    try {
      if (
        pdf
      ) {
        await pdf.destroy();
      }
    } catch {}
  }
}

/* =========================================================
   TAROT DIRECTORY CACHE
   ========================================================= */

let tarotCache:
  TarotCache | null =
  null;

const TAROT_CACHE_MS =
  10 * 60 * 1000;

async function listTarotFiles():
  Promise<string[]> {
  if (
    tarotCache &&
    Date.now() -
      tarotCache.timestamp <
      TAROT_CACHE_MS
  ) {
    return tarotCache.files;
  }

  const {
    owner,
    repo,
    branch,
    token,
  } =
    ghConfig();

  const url =
    `https://api.github.com/repos/${encodeURIComponent(
      owner
    )}/${encodeURIComponent(
      repo
    )}/contents/tarot?ref=${encodeURIComponent(
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
    !response.ok
  ) {
    const text =
      await response.text();

    throw new Error(
      `No se pudo listar /tarot (${response.status}): ${text}`
    );
  }

  const data =
    await response.json();

  if (
    !Array.isArray(
      data
    )
  ) {
    throw new Error(
      '/tarot no es un directorio válido.'
    );
  }

  const files =
    data
      .filter(
        (
          item: any
        ) =>
          item?.type ===
            'file' &&
          typeof item.name ===
            'string' &&
          /\.(png|jpe?g|webp)$/i.test(
            item.name
          )
      )
      .map(
        (
          item: any
        ) =>
          item.name as string
      );

  tarotCache = {
    files,
    timestamp:
      Date.now(),
  };

  return files;
}

function findTarotFile(
  files: string[],
  number: number
):
  string | null {
  /*
   * Busca nombres como:
   *
   * tarot juli-3.png
   * juli-03.jpg
   * juli_3.webp
   */

  const regex =
    new RegExp(
      `juli[\\s_-]*0*${number}(?=\\D|$)`,
      'i'
    );

  return (
    files.find(
      (
        filename
      ) =>
        regex.test(
          filename
        )
    ) ||
    null
  );
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
    return res.json({
      ok: true,

      service:
        'torchill-api',

      model:
        MODEL_NAME,

      geminiKeys:
        keyStates.length,

      github:
        Boolean(
          process.env
            .GITHUB_OWNER &&
          process.env
            .GITHUB_REPO &&
          process.env
            .GITHUB_TOKEN
        ),

      timestamp:
        new Date()
          .toISOString(),
    });
  }
);

/* =========================================================
   ROOT
   ========================================================= */

app.get(
  '/',
  (
    _req: Request,
    res: Response
  ) => {
    return res.json({
      ok: true,

      service:
        'Torchill API',

      endpoints: [
        'GET /health',
        'POST /upload',
        'GET /receipt/:pathB64',
        'GET /tarot/:n',
        'POST /generate-text',
        'POST /analyze-receipt',
        'GET /api/gemini/project-copy',
        'POST /api/gemini/project-copy',
      ],
    });
  }
);

/* =========================================================
   UPLOAD
   ========================================================= */

app.post(
  '/upload',
  upload.single(
    'file'
  ),
  async (
    req: Request,
    res: Response
  ) => {
    try {
      let buffer:
        Buffer;

      let extension =
        'jpg';

      /*
       * Modo multipart/form-data
       */
      if (
        req.file
      ) {
        buffer =
          req.file.buffer;

        const filename =
          req.file
            .originalname ||
          '';

        const match =
          filename.match(
            /\.([A-Za-z0-9]+)$/
          );

        if (
          match
        ) {
          extension =
            match[1]
              .toLowerCase()
              .slice(
                0,
                5
              );
        }
      } else {
        /*
         * Modo JSON:
         *
         * {
         *   "base64":"...",
         *   "ext":"jpg"
         * }
         */

        const body =
          getRequestBody(
            req
          );

        let base64 =
          typeof body.base64 ===
          'string'
            ? body.base64
            : '';

        if (
          !base64
        ) {
          return res
            .status(400)
            .json({
              ok: false,

              error:
                'Falta el archivo o el campo base64.',
            });
        }

        /*
         * Soporta data URL.
         */
        base64 =
          base64.replace(
            /^data:[^;]+;base64,/i,
            ''
          );

        buffer =
          Buffer.from(
            base64,
            'base64'
          );

        extension =
          typeof body.ext ===
          'string'
            ? body.ext
                .replace(
                  /[^a-zA-Z0-9]/g,
                  ''
                )
                .toLowerCase()
                .slice(
                  0,
                  5
                ) ||
              'jpg'
            : 'jpg';
      }

      if (
        buffer.length ===
        0
      ) {
        return res
          .status(400)
          .json({
            ok: false,

            error:
              'Archivo vacío.',
          });
      }

      if (
        buffer.length >
        MAX_UPLOAD_SIZE
      ) {
        return res
          .status(413)
          .json({
            ok: false,

            error:
              'El archivo supera los 20 MB.',
          });
      }

      const now =
        new Date();

      const ym =
        `${now.getUTCFullYear()}${String(
          now.getUTCMonth() +
            1
        ).padStart(
          2,
          '0'
        )}`;

      const id =
        crypto.randomUUID();

      const path =
        `receipts/${ym}/${id}.${extension}`;

      const result =
        await uploadToGithub(
          path,
          buffer
        );

      const receiptUrl =
        `/receipt/${result.pathB64}`;

      const absoluteUrl =
        `${req.protocol}://${req.get(
          'host'
        )}${receiptUrl}`;

      return res.json({
        ok: true,

        ...result,

        url:
          receiptUrl,

        receiptUrl,

        fileUrl:
          absoluteUrl,

        absoluteUrl,

        mimeType:
          mimeForPath(
            path
          ),

        size:
          buffer.length,
      });
    } catch (
      error: unknown
    ) {
      console.error(
        'Upload error:',
        error
      );

      return res
        .status(500)
        .json({
          ok: false,

          error:
            error instanceof
            Error
              ? error.message
              : String(
                  error
                ),
        });
    }
  }
);

/* =========================================================
   RECEIPT PRIVADO
   ========================================================= */

app.get(
  '/receipt/:pathB64',
  async (
    req: Request,
    res: Response
  ) => {
    try {
      const encoded =
        getParamString(
          req.params
            .pathB64
        );

      const path =
        decodePathB64(
          encoded
        );

      /*
       * Seguridad:
       * /receipt solo debe servir receipts/.
       */

      if (
        !path.startsWith(
          'receipts/'
        )
      ) {
        return res
          .status(403)
          .send(
            'ruta no permitida'
          );
      }

      const file =
        await downloadGithubFile(
          path
        );

      res.setHeader(
        'Content-Type',
        file.mimeType
      );

      res.setHeader(
        'Content-Length',
        String(
          file.buffer
            .length
        )
      );

      res.setHeader(
        'Cache-Control',
        'private, max-age=3600'
      );

      return res.send(
        file.buffer
      );
    } catch (
      error
    ) {
      console.error(
        'Receipt error:',
        error
      );

      return res
        .status(404)
        .send(
          'not found'
        );
    }
  }
);

/* =========================================================
   TAROT
   ========================================================= */

app.get(
  '/tarot/:n',
  async (
    req: Request,
    res: Response
  ) => {
    try {
      const raw =
        getParamString(
          req.params.n
        ).trim();

      if (
        !/^\d+$/.test(
          raw
        )
      ) {
        return res
          .status(400)
          .json({
            ok: false,

            error:
              'Número de carta inválido.',
          });
      }

      const number =
        Number.parseInt(
          raw,
          10
        );

      if (
        number < 3 ||
        number > 80
      ) {
        return res
          .status(404)
          .json({
            ok: false,

            error:
              `La carta ${number} está fuera del rango 3–80.`,
          });
      }

      const files =
        await listTarotFiles();

      const filename =
        findTarotFile(
          files,
          number
        );

      if (
        !filename
      ) {
        return res
          .status(404)
          .json({
            ok: false,

            error:
              `No se encontró la carta ${number}.`,

            expected:
              `Archivo en /tarot cuyo nombre contenga juli-${number}`,
          });
      }

      const path =
        `tarot/${filename}`;

      const file =
        await downloadGithubFile(
          path
        );

      res.setHeader(
        'Content-Type',
        file.mimeType
      );

      res.setHeader(
        'Content-Length',
        String(
          file.buffer
            .length
        )
      );

      res.setHeader(
        'Access-Control-Allow-Origin',
        '*'
      );

      res.setHeader(
        'Cache-Control',
        'public, max-age=86400'
      );

      res.setHeader(
        'X-Tarot-Card',
        String(
          number
        )
      );

      return res.send(
        file.buffer
      );
    } catch (
      error
    ) {
      console.error(
        'Tarot error:',
        error
      );

      return res
        .status(500)
        .json({
          ok: false,

          error:
            error instanceof
            Error
              ? error.message
              : String(
                  error
                ),
        });
    }
  }
);

/* =========================================================
   GENERATE TEXT
   ========================================================= */

app.post(
  '/generate-text',
  auth,
  async (
    req: Request,
    res: Response
  ) => {
    try {
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

      const systemPrompt =
        typeof body.systemPrompt ===
        'string'
          ? body.systemPrompt.trim()
          : '';

      const context =
        typeof body.context ===
        'string'
          ? body.context.trim()
          : '';

      const language =
        typeof body.language ===
        'string'
          ? body.language.trim()
          : '';

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
              'Falta prompt o text.',
          });
      }

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

      const response =
        await runGemini(
          (
            ai
          ) =>
            ai.models
              .generateContent({
                model:
                  MODEL_NAME,

                contents: [
                  {
                    role:
                      'user',

                    parts: [
                      {
                        text:
                          finalPrompt,
                      },
                    ],
                  },
                ],

                config: {
                  temperature,

                  maxOutputTokens,
                },
              })
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

      return res.json({
        ok: true,

        text:
          generatedText,

        response:
          generatedText,

        model:
          MODEL_NAME,
      });
    } catch (
      error
    ) {
      console.error(
        'Generate text error:',
        error
      );

      return res
        .status(
          isRateLimitError(
            error
          )
            ? 429
            : 500
        )
        .json({
          ok: false,

          error:
            error instanceof
            Error
              ? error.message
              : String(
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
      const body =
        getRequestBody(
          req
        );

      const fileUrl =
        typeof body.fileUrl ===
        'string'
          ? body.fileUrl.trim()
          : '';

      const prompt =
        typeof body.prompt ===
        'string'
          ? body.prompt.trim()
          : '';

      if (
        !fileUrl ||
        !prompt
      ) {
        return res
          .status(400)
          .json({
            ok: false,

            error:
              'fileUrl y prompt son requeridos.',
          });
      }

      let buffer:
        Buffer;

      let mimeType:
        string;

      /*
       * Si es una URL generada por este servidor,
       * descargamos directamente desde GitHub.
       */
      try {
        const parsed =
          new URL(
            fileUrl,
            `${req.protocol}://${req.get(
              'host'
            )}`
          );

        const receiptMatch =
          parsed.pathname.match(
            /^\/receipt\/([^/]+)$/
          );

        if (
          receiptMatch
        ) {
          const githubPath =
            decodePathB64(
              receiptMatch[1]
            );

          const file =
            await downloadGithubFile(
              githubPath
            );

          buffer =
            file.buffer;

          mimeType =
            file.mimeType;
        } else {
          const response =
            await fetch(
              fileUrl
            );

          if (
            !response.ok
          ) {
            throw new Error(
              `No se pudo descargar el archivo (${response.status}).`
            );
          }

          buffer =
            Buffer.from(
              await response.arrayBuffer()
            );

          mimeType =
            (
              response.headers.get(
                'content-type'
              ) ||
              mimeForPath(
                parsed.pathname
              )
            )
              .split(';')[0]
              .trim();
        }
      } catch (
        error
      ) {
        throw new Error(
          `No se pudo descargar el comprobante: ${
            error instanceof
            Error
              ? error.message
              : String(
                  error
                )
          }`
        );
      }

      if (
        buffer.length >
        MAX_UPLOAD_SIZE
      ) {
        return res
          .status(413)
          .json({
            ok: false,

            error:
              'El comprobante supera los 20 MB.',
          });
      }

      const isPdf =
        mimeType ===
          'application/pdf' ||
        fileUrl
          .toLowerCase()
          .includes(
            '.pdf'
          );

      if (
        isPdf
      ) {
        const png =
          await pdfFirstPageToPng(
            buffer
          );

        if (
          png
        ) {
          buffer =
            png;

          mimeType =
            'image/png';
        }
      }

      const base64 =
        buffer.toString(
          'base64'
        );

      const finalPrompt =
        `${prompt}

Devolvé ÚNICAMENTE JSON válido.
No uses markdown.
No agregues explicaciones fuera del JSON.`;

      const response =
        await runGemini(
          (
            ai
          ) =>
            ai.models
              .generateContent({
                model:
                  MODEL_NAME,

                contents: [
                  {
                    role:
                      'user',

                    parts: [
                      {
                        text:
                          finalPrompt,
                      },

                      {
                        inlineData: {
                          mimeType,

                          data:
                            base64,
                        },
                      },
                    ],
                  },
                ],

                config: {
                  temperature:
                    0.1,

                  responseMimeType:
                    'application/json',
                },
              })
        );

      const raw =
        response.text
          ?.trim() ||
        '';

      let detected:
        any = null;

      try {
        detected =
          JSON.parse(
            raw
          );
      } catch {
        const match =
          raw.match(
            /\{[\s\S]*\}/
          );

        if (
          match
        ) {
          try {
            detected =
              JSON.parse(
                match[0]
              );
          } catch {}
        }
      }

      return res.json({
        ok: true,

        detected,

        raw,

        mimeType,

        model:
          MODEL_NAME,
      });
    } catch (
      error
    ) {
      console.error(
        'Analyze receipt error:',
        error
      );

      return res
        .status(
          isRateLimitError(
            error
          )
            ? 429
            : 500
        )
        .json({
          ok: false,

          error:
            error instanceof
            Error
              ? error.message
              : String(
                  error
                ),
        });
    }
  }
);

/* =========================================================
   PROJECT COPY INFO
   ========================================================= */

app.get(
  '/api/gemini/project-copy',
  (
    _req: Request,
    res: Response
  ) => {
    return res.json({
      ok: true,

      endpoint:
        '/api/gemini/project-copy',

      model:
        MODEL_NAME,

      actions: [
        'generate',
        'translate',
      ],
    });
  }
);

/* =========================================================
   PROJECT COPY
   ========================================================= */

app.post(
  '/api/gemini/project-copy',
  auth,
  async (
    req: Request,
    res: Response
  ) => {
    try {
      const body =
        getRequestBody(
          req
        );

      const action =
        typeof body.action ===
        'string'
          ? body.action
          : 'generate';

      /*
       * TRADUCCIÓN POR SEGMENTOS
       */
      if (
        action ===
        'translate'
      ) {
        const segments =
          body.segments;

        if (
          !Array.isArray(
            segments
          ) ||
          segments.length ===
            0
        ) {
          return res
            .status(400)
            .json({
              ok: false,

              error:
                'segments debe ser un array no vacío.',
            });
        }

        const language =
          body.language ===
          'en'
            ? 'inglés'
            : body.language ===
                'es'
              ? 'español'
              : String(
                  body.language ||
                  'español'
                );

        const prompt =
          `Actúa como traductor profesional especializado en diseño gráfico y comunicación visual.

Traduce los siguientes textos al ${language}.

Reglas:
- Mantén el significado.
- No inventes información.
- No elimines información.
- Conserva exactamente el orden.
- Mantén un tono profesional.
- No agregues explicaciones.

Textos:

${JSON.stringify(
  segments
)}

Devuelve ÚNICAMENTE JSON válido con esta estructura:

{
  "translations": [
    "texto traducido 1",
    "texto traducido 2"
  ]
}`;

        const response =
          await runGemini(
            (
              ai
            ) =>
              ai.models
                .generateContent({
                  model:
                    MODEL_NAME,

                  contents: [
                    {
                      role:
                        'user',

                      parts: [
                        {
                          text:
                            prompt,
                        },
                      ],
                    },
                  ],

                  config: {
                    temperature:
                      0.1,

                    responseMimeType:
                      'application/json',
                  },
                })
          );

        const raw =
          response.text
            ?.trim() ||
          '';

        let parsed:
          any;

        try {
          parsed =
            JSON.parse(
              raw
            );
        } catch {
          throw new Error(
            'Gemini devolvió JSON inválido.'
          );
        }

        return res.json({
          ok: true,

          ...parsed,
        });
      }

      /*
       * GENERACIÓN NORMAL
       */

      const text =
        typeof body.text ===
        'string'
          ? body.text
          : '';

      const title =
        typeof body.title ===
        'string'
          ? body.title
          : '';

      const prompt =
        typeof body.prompt ===
        'string'
          ? body.prompt
          : '';

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
              'Falta prompt o text.',
          });
      }

      const finalPrompt =
        title
          ? `Título del proyecto: ${title}

${userInput}`
          : userInput;

      const response =
        await runGemini(
          (
            ai
          ) =>
            ai.models
              .generateContent({
                model:
                  MODEL_NAME,

                contents: [
                  {
                    role:
                      'user',

                    parts: [
                      {
                        text:
                          finalPrompt,
                      },
                    ],
                  },
                ],

                config: {
                  temperature:
                    0.7,

                  maxOutputTokens:
                    4096,
                },
              })
        );

      const generated =
        response.text
          ?.trim() ||
        '';

      return res.json({
        ok: true,

        text:
          generated,

        response:
          generated,

        model:
          MODEL_NAME,
      });
    } catch (
      error
    ) {
      console.error(
        'Project copy error:',
        error
      );

      return res
        .status(
          isRateLimitError(
            error
          )
            ? 429
            : 500
        )
        .json({
          ok: false,

          error:
            error instanceof
            Error
              ? error.message
              : String(
                  error
                ),
        });
    }
  }
);

/* =========================================================
   MULTER ERRORS
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
            ok: false,

            error:
              'El archivo supera los 20 MB.',
          });
      }

      return res
        .status(400)
        .json({
          ok: false,

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
        ok: false,

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

    return res
      .status(500)
      .json({
        ok: false,

        error:
          'Error interno del servidor.',

        details:
          error instanceof
          Error
            ? error.message
            : String(
                error
              ),
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
        `Modelo Gemini: ${MODEL_NAME}`
      );

      console.log(
        `Gemini API keys: ${keyStates.length}`
      );

      console.log(
        `GitHub configurado: ${Boolean(
          process.env
            .GITHUB_OWNER &&
          process.env
            .GITHUB_REPO &&
          process.env
            .GITHUB_TOKEN
        )}`
      );

      console.log(
        '----------------------------------------'
      );

      console.log(
        'GET  /'
      );

      console.log(
        'GET  /health'
      );

      console.log(
        'POST /upload'
      );

      console.log(
        'GET  /receipt/:pathB64'
      );

      console.log(
        'GET  /tarot/:n'
      );

      console.log(
        'POST /generate-text'
      );

      console.log(
        'POST /analyze-receipt'
      );

      console.log(
        'GET  /api/gemini/project-copy'
      );

      console.log(
        'POST /api/gemini/project-copy'
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
        'Servidor cerrado.'
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
