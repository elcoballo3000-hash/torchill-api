import express, { Request, Response } from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';

const app = express();

app.use(cors());
app.use(express.json({ limit: '1mb' }));

const PORT = Number(process.env.PORT) || 3000;

const modelName =
  process.env.GEMINI_MODEL || 'gemini-3-flash-preview';

/* =========================================================
   GEMINI API KEYS
   ========================================================= */

const apiKeys = [
  process.env.GEMINI_API_KEY_1,
  process.env.GEMINI_API_KEY_2,
  process.env.GEMINI_API_KEY_3,
  process.env.GEMINI_API_KEY_4,
].filter((key): key is string => Boolean(key));

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

/**
 * Obtiene la siguiente API key disponible.
 *
 * Primero intenta respetar el orden de rotación.
 * Si una key está temporalmente bloqueada por 429,
 * la salta.
 */
function getAvailableKey(): KeyState | null {
  if (keyStates.length === 0) {
    return null;
  }

  const now = Date.now();

  for (let i = 0; i < keyStates.length; i++) {
    const index =
      (currentKeyIndex + i) % keyStates.length;

    const keyState = keyStates[index];

    if (keyState.blockedUntil <= now) {
      currentKeyIndex =
        (index + 1) % keyStates.length;

      return keyState;
    }
  }

  return null;
}

/**
 * Marca una key como temporalmente bloqueada.
 */
function blockKey(
  keyState: KeyState,
  retryAfterMs: number
) {
  keyState.blockedUntil =
    Date.now() + retryAfterMs;
}

/**
 * Extrae el tiempo de espera indicado por Gemini.
 *
 * Busca mensajes como:
 *
 * "Please retry in 24.585487833s."
 *
 * Si no encuentra el tiempo,
 * utiliza 30 segundos.
 */
function getRetryAfterMs(error: unknown): number {
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
      return Math.ceil(seconds * 1000);
    }
  }

  return 30_000;
}

/**
 * Determina si el error corresponde
 * a un límite de cuota / rate limit.
 */
function isRateLimitError(
  error: unknown
): boolean {
  const message =
    error instanceof Error
      ? error.message
      : String(error);

  return (
    message.includes('429') ||
    message.includes('too_many_requests') ||
    message.includes('RateLimitError') ||
    message.includes('quota exceeded') ||
    message.includes('Quota exceeded')
  );
}

/* =========================================================
   EJECUTAR GEMINI CON FALLBACK AUTOMÁTICO
   ========================================================= */

async function runGemini<T>(
  operation: (ai: GoogleGenAI) => Promise<T>
): Promise<T> {

  if (keyStates.length === 0) {
    throw new Error(
      'No hay ninguna GEMINI_API_KEY configurada en el servidor.'
    );
  }

  const attemptedKeys = new Set<KeyState>();

  for (
    let attempt = 0;
    attempt < keyStates.length;
    attempt++
  ) {

    const keyState =
      getAvailableKey();

    if (!keyState) {
      throw new Error(
        'Todas las API keys de Gemini están temporalmente limitadas. Intenta nuevamente en unos segundos.'
      );
    }

    attemptedKeys.add(keyState);

    try {

      console.log(
        `Gemini: usando API key ${keyStates.indexOf(keyState) + 1}`
      );

      const result =
        await operation(keyState.ai);

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
        `Gemini: API key ${
          keyStates.indexOf(keyState) + 1
        } alcanzó el límite. ` +
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
   HEALTH CHECK
   ========================================================= */

app.get(
  '/api/gemini/project-copy',
  (_req: Request, res: Response) => {

    res.status(200).json({
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
   GEMINI
   ========================================================= */

app.post(
  '/api/gemini/project-copy',
  async (req: Request, res: Response) => {

    try {

      if (apiKeys.length === 0) {
        return res.status(500).json({
          error:
            'No hay GEMINI_API_KEY configurada.',
        });
      }

      const {
        action,
        text,
        language,
        title,
        segments,
      } = req.body;

      /* =====================================================
         TRADUCCIÓN
         ===================================================== */

      if (action === 'translate') {

        if (!Array.isArray(segments)) {
          return res.status(400).json({
            error:
              '"segments" debe ser un array.',
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
          await runGemini((ai) =>
            ai.interactions.create({

              model: modelName,

              input: prompt,

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
          interaction.output_text?.trim();

        if (!output) {
          throw new Error(
            'Gemini devolvió una respuesta vacía.'
          );
        }

        const result =
          JSON.parse(output);

        return res.status(200).json(result);
      }

      /* =====================================================
         PROYECTO / PORTFOLIO
         ===================================================== */

      if (
        !title ||
        typeof title !== 'string'
      ) {

        return res.status(400).json({
          error:
            'Falta el campo "title".',
        });

      }

      if (
        !text ||
        typeof text !== 'string'
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
        await runGemini((ai) =>
          ai.interactions.create({

            model: modelName,

            input: prompt,

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
        interaction.output_text?.trim();

      if (!output) {
        throw new Error(
          'Gemini devolvió una respuesta vacía.'
        );
      }

      const result =
        JSON.parse(output);

      return res.status(200).json(result);

    } catch (error: unknown) {

      console.error(
        'Gemini error detallado:',
        error
      );

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      if (isRateLimitError(error)) {

        return res.status(429).json({

          error:
            'Todas las API keys de Gemini están temporalmente limitadas.',

          details:
            'Intenta nuevamente en unos segundos.',

        });

      }

      return res.status(500).json({

        error:
          'Error procesando la solicitud con Gemini.',

        details: message,

      });

    }

  }
);

/* =========================================================
   404
   ========================================================= */

app.use(
  (_req: Request, res: Response) => {

    res.status(404).json({
      error:
        'Endpoint no encontrado.',
    });

  }
);

/* =========================================================
   START
   ========================================================= */

app.listen(PORT, () => {

  console.log(
    `Servidor corriendo en puerto ${PORT}`
  );

  console.log(
    `Modelo Gemini: ${modelName}`
  );

  console.log(
    `Gemini API keys disponibles: ${apiKeys.length}`
  );

});
