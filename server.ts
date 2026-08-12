import express, { Request, Response } from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';

const app = express();

/* =========================================================
   CONFIGURACIÓN GENERAL
   ========================================================= */

const PORT = Number(process.env.PORT) || 3000;

const modelName =
  process.env.GEMINI_MODEL || 'gemini-3-flash-preview';

/* =========================================================
   CORS
   ========================================================= */

app.use(
  cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
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
 * También aceptamos text/plain por si algún cliente,
 * como Base44, envía accidentalmente JSON como texto.
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

function getAvailableKey(): KeyState | null {

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

    if (
      attemptedKeys.has(keyState)
    ) {
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

      if (
        !isRateLimitError(error)
      ) {

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

  let body: unknown =
    req.body;

  /*
   * Si Express recibió un string,
   * intentamos convertirlo a JSON.
   */

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

  /*
   * Si no existe body,
   * devolvemos objeto vacío en lugar
   * de provocar un TypeError.
   */

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
   HEALTH CHECK
   ========================================================= */

app.get(
  '/api/gemini/project-copy',
  (_req: Request, res: Response) => {

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
   GEMINI ENDPOINT
   ========================================================= */

app.post(
  '/api/gemini/project-copy',
  async (
    req: Request,
    res: Response
  ) => {

    try {

      /* ===================================================
         VERIFICAR GEMINI
         =================================================== */

      if (
        apiKeys.length === 0
      ) {

        return res.status(503).json({

          error:
            'El servicio Gemini no está configurado.',

        });
      }

      /* ===================================================
         NORMALIZAR BODY
         =================================================== */

      const body =
        getRequestBody(req);

      const {
        action,
        text,
        language,
        title,
        segments,
      } = body;

      /*
       * Si Base44 manda una petición sin body,
       * ahora devolvemos 400 en lugar de provocar
       * un TypeError y terminar en 500/502.
       */

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

      /* ===================================================
         TRADUCCIÓN
         =================================================== */

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

                    type:
                      'object',

                    properties: {

                      translations: {

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
              })
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

        let result;

        try {

          result =
            JSON.parse(output);

        } catch {

          throw new Error(
            'Gemini devolvió un JSON inválido.'
          );
        }

        return res
          .status(200)
          .json(result);
      }

      /* ===================================================
         PROYECTO / PORTFOLIO
         =================================================== */

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

Devuelve exclusivamente JSON.
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
            })
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

      let result;

      try {

        result =
          JSON.parse(output);

      } catch {

        throw new Error(
          'Gemini devolvió un JSON inválido.'
        );
      }

      return res
        .status(200)
        .json(result);

    } catch (
      error: unknown
    ) {

      console.error(
        'Gemini error:',
        error
      );

      /* ================================================
         RATE LIMIT
         ================================================ */

      if (
        isRateLimitError(error)
      ) {

        return res.status(429).json({

          error:
            'Todas las API keys de Gemini están temporalmente limitadas.',

          retryable:
            true,

        });
      }

      /* ================================================
         ERROR GENERAL
         ================================================ */

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      return res.status(500).json({

        error:
          'Error procesando la solicitud con Gemini.',

        /*
         * Esto ayuda durante desarrollo.
         * No contiene las API keys.
         */

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
  }
);
