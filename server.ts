import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post('/api/gemini/project-copy', async (req: any, res: any) => {
  try {
    const { action, text, language, title, lead, discipline, sections, imageAlts, segments } = req.body;

    if (action === 'translate') {
      const prompt = `Traduce los siguientes textos al idioma ${language === 'en' ? 'inglés' : 'español'}, manteniendo el tono profesional y de diseño. Devuelve ÚNICAMENTE un objeto JSON con el array "translations" de strings traducidos.\nTextos: ${JSON.stringify(segments)}`;
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
      });
      const rawText = response.text || '{}';
      const cleanJson = rawText.replace(/```json/g, '').replace(/```/g, '').trim();
      return res.json(JSON.parse(cleanJson));
    }

    const prompt = `Actúa como un director de arte editorial. Mejora los siguientes textos del caso de estudio "${title}" para un portafolio de diseño de alto nivel.
    Texto actual: ${text}
    Devuelve ÚNICAMENTE un objeto JSON válido con la siguiente estructura:
    {
      "lead": "Un párrafo de introducción potente y pulido",
      "discipline": "Disciplina depurada",
      "sections": [{"title": "Título de sección", "summary": "Resumen mejorado"}],
      "imageAlts": ["alt de imagen 1"]
    }`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    const rawText = response.text || '{}';
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
