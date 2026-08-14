import express, { Request, Response } from 'express';
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

const PORT = Number(process.env.PORT) || 3000;

const modelName =
  process.env.GEMINI_MODEL || 'gemini-3-flash-preview';

const API_TOKEN = process.env.API_TOKEN;

/*
 * Tamaño máximo del archivo que /analyze-receipt
 * puede descargar desde fileUrl.
 */
const MAX_RECEIPT_FILE_SIZE = 20 * 1024 * 1024; // 20 MB

/* =========================================================
   CORS
   ========================================================= */

app.use(
  cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'X-API-Key',
    ],
  })
);

app.options('/{*splat}', cors());

/* =========================================================
   BODY PARSER
   ========================================================= */

app.use(
  express.json({
    limit: '1mb',
  })
);

/*
 * También aceptamos text/plain por si Base44
 * envía accidentalmente JSON como texto.
 */
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
  (key): key is string =>
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

const keyStates: KeyState[] = apiKeys.map((key) => ({
  ai: new GoogleGenAI({
    apiKey: key,
  }),
  blockedUntil: 0,
}));

let currentKeyIndex = 0;

/* =========================================================
   OBTENER KEY DISPONIBLE
   ========================================================= */

function getAvailableKey(): KeyState | null {
  if (keyStates.length === 0) {
    return null;
  }

  const now = Date.now();

  for (let i = 0; i < keyStates.length; i++) {
    const index =
      (currentKeyIndex + i) %
      keyStates.length;

    const keyState = keyStates[index];

    if (keyState.blockedUntil <= now) {
      currentKeyIndex =
        (index + 1) % keyStates.length;

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
      Number.parseFloat(match[1]);

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
    normalized.includes('too_many_requests') ||
    normalized.includes('ratelimiterror') ||
    normalized.includes('rate limit') ||
    normalized.includes('quota exceeded') ||
    normalized.includes('resource exhausted')
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

    /*
     * Evita utilizar la misma key dos veces
     * durante una única solicitud.
     */
    if (attemptedKeys.has(keyState)) {
      break;
    }

    attemptedKeys.add(keyState);

    const keyNumber =
      keyStates.indexOf(keyState) + 1;

    try {
      console.log(
        `Gemini: usando API key ${keyNumber}`
      );

      const result =
        await operation(
          keyState.ai
        );

      return result;
    } catch (error: unknown) {
      if (!isRateLimitError(error)) {
        throw error;
      }

      const retryAfterMs =
        getRetryAfterMs(error);

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
  let body: unknown = req.body;

  if (typeof body === 'string') {
    if (body.trim().length === 0) {
      return {};
    }

    try {
      body = JSON.parse(body);
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

  return body as Record<string, any>;
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
      .replace(/^```json\s*/i, '')
      .replace(/^```\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    /*
     * Fallback por si Gemini devuelve texto
     * alrededor del JSON.
     */
    const objectMatch =
      cleaned.match(
        /\{[\s\S]*\}/
      );

    if (objectMatch) {
      return JSON.parse(
        objectMatch[0]
      );
    }

    const arrayMatch =
      cleaned.match(
        /[\s\S]*/
      );

    if (arrayMatch) {
      return JSON.parse(
        arrayMatch[0]
      );
    }

    throw new Error(
      'Gemini devolvió un JSON inválido.'
    );
  }
}

/* =========================================================
   AUTH PARA RECEIPTS
   ========================================================= */

function auth(
  req: Request,
  res: Response,
  next: () => void
) {
  /*
   * Si API_TOKEN no está configurado,
   * el endpoint sigue funcionando sin auth,
   * igual que el index.js original.
   */
  if (!API_TOKEN) {
    return next();
  }

  const sent =
    req.headers['x-api-key'] ||
    (req.headers.authorization || '')
      .replace(
        /^Bearer\s+/i,
        ''
      );

  if (sent !== API_TOKEN) {
    return res.status(401).json({
      error: 'no autorizado',
    });
  }

  return next();
}

/* =========================================================
   MIME TYPE
   ========================================================= */

function getMimeType(
  fileUrl: string,
  contentType: string
): string {
  const detected =
    contentType
      .split(';')[0]
      .trim()
      .toLowerCase();

  if (
    detected &&
    detected !== 'application/octet-stream'
  ) {
    return detected;
  }

  const lowerUrl =
    fileUrl.toLowerCase();

  if (
    lowerUrl.includes('.pdf')
  ) {
    return 'application/pdf';
  }

  if (
    lowerUrl.includes('.png')
  ) {
    return 'image/png';
  }

  if (
    lowerUrl.includes('.webp')
  ) {
    return 'image/webp';
  }

  if (
    lowerUrl.includes('.gif')
  ) {
    return 'image/gif';
  }

  return 'image/jpeg';
}

/* =========================================================
   DESCARGAR RECEIPT
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
      new URL(fileUrl);
  } catch {
    throw new Error(
      'fileUrl no es una URL válida.'
    );
  }

  if (
    parsedUrl.protocol !== 'http:' &&
    parsedUrl.protocol !== 'https:'
  ) {
    throw new Error(
      'fileUrl debe utilizar HTTP o HTTPS.'
    );
  }

  const fileRes =
    await fetch(fileUrl);

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

  const chunks: Uint8Array[] = [];
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
        total += value.length;

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

        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }

  const buffer =
    Buffer.concat(
      chunks.map(
        (chunk) =>
          Buffer.from(chunk)
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
   HEALTH CHECK GENERAL
   ========================================================= */

app.get(
  '/health',
  (_req: Request, res: Response) => {
    return res.status(200).json({
      ok: true,
      service: 'torchill-api',
      model: modelName,
      gemini: apiKeys.length > 0,
      geminiConfigured:
        apiKeys.length > 0,
      geminiKeys:
        apiKeys.length,
    });
  }
);

/* =========================================================
   HEALTH CHECK PROJECT COPY
   ========================================================= */

app.get(
  '/api/gemini/project-copy',
  (_req: Request, res: Response) => {
    return res.status(200).json({
      status: 'ok',
      service: 'torchill-api',
      model: modelName,
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
   GEMINI PROJECT COPY
   ========================================================= */

app.post(
  '/api/gemini/project-copy',
  async (
    req: Request,
    res: Response
  ) => {
    try {
      if (apiKeys.length === 0) {
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
        Object.keys(body).length === 0
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
        action === 'translate'
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

${JSON.stringify(segments)}

Devuelve exclusivamente un JSON válido con esta estructura:

{
  "translations": ["texto traducido 1", "texto traducido 2"]
}
`;

        const interaction =
          await runGemini(
            (ai) =>
              ai.interactions.create({
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
                      translations: {
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
              })
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
      }

      /* =====================================================
         PROYECTO / PORTFOLIO
         ===================================================== */

      if (
        typeof title !== 'string' ||
        title.trim().length === 0
      ) {
        return res.status(400).json({
          error:
            'Falta el campo "title".',
        });
      }

      if (
        typeof text !== 'string' ||
        text.trim().length === 0
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
            ai.interactions.create({
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
            })
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
        isRateLimitError(error)
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
      if (apiKeys.length === 0) {
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
        typeof fileUrl !== 'string' ||
        fileUrl.trim().length === 0
      ) {
        return res.status(400).json({
          error:
            'fileUrl es requerido.',
        });
      }

      if (
        typeof prompt !== 'string' ||
        prompt.trim().length === 0
      ) {
        return res.status(400).json({
          error:
            'prompt es requerido.',
        });
      }

      /* ===================================================
         1. DESCARGAR ARCHIVO
         =================================================== */

      const {
        buffer,
        mimeType,
      } =
        await downloadReceipt(
          fileUrl
        );

      const base64 =
        buffer.toString(
          'base64'
        );

      console.log(
        `Receipt: archivo descargado (${buffer.length} bytes, ${mimeType})`
      );

      /* ===================================================
         2. ANALIZAR CON GEMINI
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
            ai.models.generateContent({
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
                      inlineData: {
                        mimeType,
                        data: base64,
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
            })
        );

      const text =
        response.text?.trim() || '';

      if (!text) {
        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      /* ===================================================
         3. PARSEAR JSON
         =================================================== */

      const parsed =
        parseGeminiJson(
          text
        );

      return res.status(200).json({
        ok: true,
        detected: parsed,
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
        isRateLimitError(error)
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
  }
);