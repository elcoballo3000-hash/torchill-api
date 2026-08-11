import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// Forzamos el uso de la versión v1 de la API si la v1beta da 404
const model = genAI.getGenerativeModel({
  model: process.env.GEMINI_MODEL || "gemini-1.5-flash"
}, { apiVersion: 'v1' });

const app = express();
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');
const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

// Ruta de comprobación para que no dé 404 si se abre desde el navegador
app.get('/api/gemini/project-copy', (req: any, res: any) => {
  res.json({ status: 'API de Torchill funcionando correctamente. Envía un POST para procesar con IA.' });
});

app.post('/api/gemini/project-copy', async (req: any, res: any) => {
  try {
    const { action, text, language, title, lead, discipline, sections, imageAlts, segments } = req.body;

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
    console.error(error);
    res.status(500).json({ error: 'Error procesando la solicitud con IA.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Servidor corriendo en puerto ${PORT}`);
});
