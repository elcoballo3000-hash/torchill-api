import express, { Request, Response } from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';

const app = express();

app.use(cors());
app.use(express.json({ limit: '1mb' }));

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  console.error('ERROR: GEMINI_API_KEY no está configurada.');
}

const modelName = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

const ai = apiKey
  ? new GoogleGenAI({
      apiKey,
    })
  : null;

/**
 * Health check
 */
app.get('/api/gemini/project-copy', (_req: Request, res: Response) => {
  res.status(200).json({
    status: 'ok',
    service: 'torchill-api',
    message:
      'API de Torchill funcionando correctamente. Envía un POST para procesar con IA.',
    model: modelName,
    geminiConfigured: Boolean(apiKey),
  });
});

/**
 * Procesamiento principal con Gemini
 */
app.post('/api/gemini/project-copy', async (req: Request, res: Response) => {
  try {
    if (!apiKey || !ai) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY no está configurada en el servidor.',
      });
    }

    if (!req.body || Object.keys(req.body).length === 0) {
      return res.status(400).json({
        error: 'El cuerpo de la petición debe ser un JSON válido.',
      });
    }

    const { action, text, language, title, segments } = req.body;

    /**
     * =========================================================
     * TRADUCCIÓN
     * =========================================================
     */
    if (action === 'translate') {
      if (!Array.isArray(segments)) {
        return res.status(400).json({
          error: 'Para traducir, "segments" debe ser un array de textos.',
        });
      }

      const targetLanguage =
        language === 'en' ? 'inglés' : 'español';

      const prompt = `
Traduce los siguientes textos al idioma ${targetLanguage}.

Contexto:
Son textos pertenecientes a un portafolio profesional de diseño gráfico.

Reglas:
- Mantén el significado original.
- Mantén un tono profesional y natural.
- No agregues información que no esté presente.
- No elimines información.
- Conserva el orden de los textos.
- Devuelve únicamente el resultado solicitado.

Textos:
${JSON.stringify(segments)}
`;

      const result = await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: 'object',
            properties: {
              translations: {
                type: 'array',
                items: {
                  type: 'string',
                },
              },
            },
            required: ['translations'],
          },
        },
      });

      const rawText = result.text?.trim() || '';

      if (!rawText) {
        throw new Error('Gemini devolvió una respuesta vacía.');
      }

      const parsed = JSON.parse(rawText);

      return res.status(200).json(parsed);
    }

    /**
     * =========================================================
     * MEJORA DE CASO DE ESTUDIO
     * =========================================================
     */

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
Actúa como un director de arte editorial especializado en
portafolios profesionales de diseño gráfico.

Tu tarea es mejorar y estructurar el texto de un caso de estudio.

Proyecto:
"${title}"

Texto actual:
${text}

Objetivos:
- Mejorar claridad y redacción.
- Mantener la intención original.
- Utilizar un tono profesional.
- Evitar frases genéricas o excesivamente publicitarias.
- Mantener un lenguaje apropiado para un portfolio de diseño.
- No inventar información.
- No agregar datos que no estén presentes.

Devuelve únicamente un objeto JSON con esta estructura exacta:
{
  "lead": "Un párrafo de introducción potente y pulido",
  "discipline": "Disciplina depurada",
  "sections": [
    {
      "title": "Título de sección",
      "summary": "Resumen mejorado"
    }
  ],
  "imageAlts": [
    "Texto alternativo descriptivo de imagen"
  ]
}
`;

    const result = await ai.models.generateContent({
      model: modelName,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
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
                required: ['title', 'summary'],
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

    const rawText = result.text?.trim() || '';

    if (!rawText) {
      throw new Error('Gemini devolvió una respuesta vacía.');
    }

    const parsed = JSON.parse(rawText);

    return res.status(200).json(parsed);
  } catch (error: unknown) {
    console.error('Gemini error detallado:', error);

    const message =
      error instanceof Error
        ? error.message
        : String(error);

    return res.status(500).json({
      error: 'Error procesando la solicitud con IA.',
      details: message,
    });
  }
});

/**
 * Ruta 404
 */
app.use((_req: Request, res: Response) => {
  res.status(404).json({
    error: 'Endpoint no encontrado.',
  });
});

/**
 * Inicio del servidor
 */
const PORT = Number(process.env.PORT) || 3000;

app.listen(PORT, () => {
  console.log(`Servidor corriendo en puerto ${PORT}`);
  console.log(`Modelo Gemini: ${modelName}`);
});