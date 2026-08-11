import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
app.use(cors());
app.use(express.json());

const apiKey = process.env.GEMINI_API_KEY || '';
const modelName = process.env.GEMINI_MODEL || 'gemini-1.5-flash';

const genAI = new GoogleGenerativeAI(apiKey);

// Forzamos el uso de la versión v1 de la API para evitar conflictos de modelos y rutas
const model = genAI.getGenerativeModel({
  model: modelName
}, { apiVersion: 'v1' });

// Ruta de comprobación GET
app.get('/api/gemini/project-copy', (req: any, res: any) => {
  res.json({ status: 'API de Torchill funcionando correctamente. Envía un POST para procesar con IA.' });
});

// Ruta principal POST
app.post('/api/gemini/project-copy', async (req: any, res: any) => {
  try {
    const { action, text, language, title, segments } = req.body;

    if (action === 'translate') {
      const prompt = `Traduce los siguientes textos al idioma ${language === 'en' ? 'inglés' : 'español'}, manteniendo el tono profesional y de diseño. Devuelve ÚNICAMENTE un objeto JSON válido con el array "translations" de strings traducidos.\nTextos: ${JSON.stringify(segments)}`;
      const result = await model.generateContent(prompt);
      const rawText = result.response.text() || '{}';
      const cleanJson = rawText.replace(/```json/g, '').replace(/```/g, '').trim();
      return res.json(JSON.parse(cleanJson));
    }

    const prompt = `Actúa como un director de arte editorial. Mejora los siguientes textos del caso de estudio "${title}" para un portafolio de diseño de alto nivel.
    Texto actual: ${text}
    Devuelve ÚNICAMENTE un objeto JSON válido con la siguiente estructura exacta:
    {
      "lead": "Un párrafo de introducción potente y pulido",
      "discipline": "Disciplina depurada",
      "sections": [{"title": "Título de sección", "summary": "Resumen mejorado"}],
      "imageAlts": ["alt de imagen 1"]
    }`;

    const result = await model.generateContent(prompt);
    const rawText = result.response.text() || '{}';
    const cleanJson = rawText.replace(/```json/g, '').replace(/```/g, '').trim();
    res.json(JSON.parse(cleanJson));
  } catch (error: any) {
    console.error("Gemini error detallado:", error?.message || error);
    res.status(500).json({ error: 'Error procesando la solicitud con IA.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Servidor corriendo en puerto ${PORT}`);
});
