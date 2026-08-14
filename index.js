
const express = require("express");

const app = express();
app.use(express.json({ limit: "60mb" }));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const API_TOKEN = process.env.API_TOKEN; // opcional

if (!GEMINI_API_KEY) {
  console.warn("GEMINI_API_KEY no esta configurada en Render -> Environment.");
}

function auth(req, res, next) {
  if (!API_TOKEN) return next();
  const sent =
    req.headers["x-api-key"] ||
    (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  if (sent !== API_TOKEN) return res.status(401).json({ error: "no autorizado" });
  return next();
}

app.get("/health", (_req, res) => res.json({ ok: true, gemini: !!GEMINI_API_KEY }));

app.post("/analyze-receipt", auth, async (req, res) => {
  try {
    const { fileUrl, prompt } = req.body || {};
    if (!fileUrl || !prompt) {
      return res.status(400).json({ error: "fileUrl y prompt son requeridos" });
    }
    if (!GEMINI_API_KEY) {
      return res.status(500).json({ error: "GEMINI_API_KEY no configurada en Render" });
    }

    // 1) Descargar el archivo (imagen o PDF).
    const fileRes = await fetch(fileUrl);
    if (!fileRes.ok) {
      return res
        .status(400)
        .json({ error: "No se pudo descargar el archivo (" + fileRes.status + ")" });
    }
    const arrayBuffer = await fileRes.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const detectedMime = (fileRes.headers.get("content-type") || "")
      .split(";")[0]
      .trim();
    const mimeByExt = fileUrl.toLowerCase().endsWith(".pdf")
      ? "application/pdf"
      : "image/jpeg";
    const mimeType =
      detectedMime && !detectedMime.includes("octet-stream")
        ? detectedMime
        : mimeByExt;
    const base64 = buffer.toString("base64");

    // 2) Llamar a Gemini 2.0 Flash (soporta imagenes y PDFs via inline_data).
    const geminiRes = await fetch(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text:
                    prompt +
                    "\n\nDevolvÃ© ÃšNICAMENTE un JSON vÃ¡lido, sin markdown ni texto adicional.",
                },
                { inline_data: { mime_type: mimeType, data: base64 } },
              ],
            },
          ],
          generation_config: {
            response_mime_type: "application/json",
            temperature: 0.1,
          },
        }),
      }
    );

    if (!geminiRes.ok) {
      const errText = await geminiRes.text();
      return res.status(502).json({ error: "Gemini error: " + errText });
    }

    const geminiJson = await geminiRes.json();
    const text =
      (geminiJson &&
        geminiJson.candidates &&
        geminiJson.candidates[0] &&
        geminiJson.candidates[0].content &&
        geminiJson.candidates[0].content.parts &&
        geminiJson.candidates[0].content.parts[0] &&
        geminiJson.candidates[0].content.parts[0].text) ||
      "";
    let parsed = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      const m = text.match(/\{[\s\S]*\}/);
      if (m) {
        try {
          parsed = JSON.parse(m[0]);
        } catch {}
      }
    }

    return res.json({ ok: true, detected: parsed, raw: text });
  } catch (e) {
    return res.status(500).json({ error: e?.message || "unknown error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Torchill API escuchando en puerto " + PORT));