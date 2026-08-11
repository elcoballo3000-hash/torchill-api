import express, { Request, Response } from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';

const app = express();

app.use(cors());
app.use(express.json({ limit: '1mb' }));

const PORT = Number(process.env.PORT) || 3000;

const apiKey = process.env.GEMINI_API_KEY;

const modelName =
  process.env.GEMINI_MODEL || 'gemini-3-flash-preview';

if (!apiKey) {
  console.error('ERROR: GEMINI_API_KEY no está configurada.');
}

const ai = apiKey
  ? new GoogleGenAI({
      apiKey: apiKey,
    })
  : null;


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
      geminiConfigured: Boolean(apiKey),
      message: 'Torchill API funcionando correctamente.',
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

      if (!apiKey || !ai) {
        return res.status(500).json({
          error: 'GEMINI_API_KEY no configurada.',
        });
      }

      const { action, text, language, title, segments } = req.body;


      /* =====================================================
         TRADUCCIÓN
         ===================================================== */

      if (action === 'translate') {

        if (!Array.isArray(segments)) {
          return res.status(400).json({
            error: '"segments" debe ser un array.',
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
          await ai.interactions.create({

            model: modelName,

            input: prompt,

            response_format: {
              type: 'text',
              mime_type: 'application/json',

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
          });


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

      if (!title || typeof title !== 'string') {

        return res.status(400).json({
          error: 'Falta el campo "title".',
        });

      }


      if (!text || typeof text !== 'string') {

        return res.status(400).json({
          error: 'Falta el campo "text".',
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
        await ai.interactions.create({

          model: modelName,

          input: prompt,

          response_format: {

            type: 'text',

            mime_type: 'application/json',

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
        });


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
      error: 'Endpoint no encontrado.',
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

});