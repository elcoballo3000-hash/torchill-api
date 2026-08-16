// @ts-nocheck

import express from "express";
import cors from "cors";
import crypto from "node:crypto";

import {
  createCanvas,
  loadImage,
} from "@napi-rs/canvas";

import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";

/* =========================================================
   TORCHILL API
   ========================================================= */

const app = express();

app.set("trust proxy", 1);

app.use(
  cors({
    origin: "*",
    methods: [
      "GET",
      "POST",
      "PUT",
      "OPTIONS",
    ],
    allowedHeaders: [
      "Content-Type",
      "Authorization",
      "X-API-Key",
    ],
  })
);

app.use(
  express.json({
    limit: "60mb",
  })
);

/* =========================================================
   CONFIG
   ========================================================= */

const PORT =
  Number(process.env.PORT) ||
  3000;

const API_TOKEN =
  process.env.API_TOKEN ||
  "";

const GEMINI_API_KEY =
  process.env.GEMINI_API_KEY ||
  "";

const GEMINI_MODEL =
  process.env.GEMINI_MODEL ||
  "gemini-2.0-flash";

/* =========================================================
   AUTH
   ========================================================= */

function auth(
  req,
  res,
  next
) {
  if (!API_TOKEN) {
    return next();
  }

  const headerKey =
    req.headers[
      "x-api-key"
    ];

  const authorization =
    req.headers.authorization ||
    "";

  const sent =
    typeof headerKey ===
    "string"
      ? headerKey
      : authorization.replace(
          /^Bearer\s+/i,
          ""
        );

  if (
    sent !==
    API_TOKEN
  ) {
    return res
      .status(401)
      .json({
        error:
          "no autorizado",
      });
  }

  return next();
}

/* =========================================================
   HELPERS
   ========================================================= */

function getParamString(
  value
) {
  if (
    typeof value ===
    "string"
  ) {
    return value;
  }

  if (
    Array.isArray(value)
  ) {
    return value[0] || "";
  }

  return "";
}

/* =========================================================
   GITHUB CONFIG
   ========================================================= */

function ghConfig() {
  const owner =
    process.env.GITHUB_OWNER ||
    "";

  const repo =
    process.env.GITHUB_REPO ||
    "";

  const branch =
    process.env.GITHUB_BRANCH ||
    "main";

  const token =
    process.env.GITHUB_TOKEN ||
    "";

  if (
    !owner ||
    !repo ||
    !token
  ) {
    throw new Error(
      "GitHub no configurado. Faltan GITHUB_OWNER, GITHUB_REPO o GITHUB_TOKEN."
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
  token
) {
  return {
    Authorization:
      `Bearer ${token}`,

    Accept:
      "application/vnd.github+json",

    "X-GitHub-Api-Version":
      "2022-11-28",

    "User-Agent":
      "torchill-api",
  };
}

/* =========================================================
   GITHUB PATH
   ========================================================= */

function encodeGithubPath(
  path
) {
  return path
    .split("/")
    .map(
      (part) =>
        encodeURIComponent(
          part
        )
    )
    .join("/");
}

/* =========================================================
   MIME
   ========================================================= */

function mimeForPath(
  path
) {
  const lower =
    String(
      path || ""
    ).toLowerCase();

  if (
    lower.endsWith(
      ".pdf"
    )
  ) {
    return "application/pdf";
  }

  if (
    lower.endsWith(
      ".png"
    )
  ) {
    return "image/png";
  }

  if (
    lower.endsWith(
      ".webp"
    )
  ) {
    return "image/webp";
  }

  if (
    lower.endsWith(
      ".jpg"
    ) ||
    lower.endsWith(
      ".jpeg"
    )
  ) {
    return "image/jpeg";
  }

  return "application/octet-stream";
}

/* =========================================================
   HEALTH
   ========================================================= */

app.get(
  "/health",
  (
    _req,
    res
  ) => {
    return res.json({
      ok: true,

      service:
        "torchill-api",

      gemini:
        Boolean(
          GEMINI_API_KEY
        ),

      github:
        Boolean(
          process.env
            .GITHUB_OWNER &&
          process.env
            .GITHUB_REPO &&
          process.env
            .GITHUB_TOKEN
        ),

      model:
        GEMINI_MODEL,

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
  "/",
  (
    _req,
    res
  ) => {
    return res.json({
      ok: true,

      service:
        "Torchill API",

      endpoints: [
        "GET /health",
        "POST /upload",
        "GET /receipt/:pathB64",
        "GET /tarot/:n",
        "GET /tarot/:w/:n",
        "GET /tarot-manifest",
        "POST /analyze-receipt",
        "POST /generate-text",
      ],
    });
  }
);

/* =========================================================
   UPLOAD RECEIPT
   ========================================================= */

app.post(
  "/upload",
  async (
    req,
    res
  ) => {
    try {
      const {
        base64,
        ext,
      } =
        req.body || {};

      if (!base64) {
        return res
          .status(400)
          .json({
            error:
              "base64 requerido",
          });
      }

      const {
        owner,
        repo,
        branch,
        token,
      } =
        ghConfig();

      const safeExt =
        String(
          ext || "jpg"
        )
          .replace(
            /[^a-zA-Z0-9]/g,
            ""
          )
          .slice(
            0,
            5
          )
          .toLowerCase() ||
        "jpg";

      const now =
        new Date();

      const ym =
        `${now.getUTCFullYear()}${String(
          now.getUTCMonth() +
            1
        ).padStart(
          2,
          "0"
        )}`;

      const id =
        crypto.randomUUID();

      const path =
        `receipts/${ym}/${id}.${safeExt}`;

      const cleanBase64 =
        String(base64)
          .replace(
            /^data:[^;]+;base64,/i,
            ""
          );

      const url =
        `https://api.github.com/repos/${encodeURIComponent(
          owner
        )}/${encodeURIComponent(
          repo
        )}/contents/${encodeGithubPath(
          path
        )}`;

      const response =
        await fetch(
          url,
          {
            method:
              "PUT",

            headers: {
              ...githubHeaders(
                token
              ),

              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify({
                message:
                  `receipt: ${id}.${safeExt}`,

                content:
                  cleanBase64,

                branch,
              }),
          }
        );

      if (
        !response.ok
      ) {
        const text =
          await response.text();

        return res
          .status(502)
          .json({
            error:
              "GitHub error: " +
              text,
          });
      }

      const pathB64 =
        Buffer.from(
          path
        ).toString(
          "base64url"
        );

      const relativeUrl =
        `/receipt/${pathB64}`;

      const absoluteUrl =
        `${req.protocol}://${req.get(
          "host"
        )}${relativeUrl}`;

      return res.json({
        ok: true,

        path,

        pathB64,

        url:
          relativeUrl,

        receiptUrl:
          relativeUrl,

        fileUrl:
          absoluteUrl,

        absoluteUrl,
      });
    } catch (
      error
    ) {
      console.error(
        "Upload error:",
        error
      );

      return res
        .status(500)
        .json({
          error:
            error?.message ||
            "unknown error",
        });
    }
  }
);

/* =========================================================
   RECEIPT
   ========================================================= */

app.get(
  "/receipt/:pathB64",
  async (
    req,
    res
  ) => {
    try {
      const {
        owner,
        repo,
        branch,
        token,
      } =
        ghConfig();

      const encoded =
        getParamString(
          req.params
            .pathB64
        );

      const path =
        Buffer.from(
          encoded,
          "base64url"
        ).toString(
          "utf8"
        );

      if (
        !path.startsWith(
          "receipts/"
        )
      ) {
        return res
          .status(403)
          .send(
            "ruta no permitida"
          );
      }

      const url =
        `https://api.github.com/repos/${encodeURIComponent(
          owner
        )}/${encodeURIComponent(
          repo
        )}/contents/${encodeGithubPath(
          path
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
                "application/vnd.github.raw",
            },
          }
        );

      if (
        !response.ok
      ) {
        return res
          .status(
            response.status
          )
          .send(
            "not found"
          );
      }

      const buffer =
        Buffer.from(
          await response.arrayBuffer()
        );

      res.setHeader(
        "Content-Type",
        mimeForPath(
          path
        )
      );

      res.setHeader(
        "Cache-Control",
        "private, max-age=3600"
      );

      res.setHeader(
        "Content-Length",
        String(
          buffer.length
        )
      );

      return res.send(
        buffer
      );
    } catch (
      error
    ) {
      console.error(
        "Receipt error:",
        error
      );

      return res
        .status(500)
        .send(
          "error"
        );
    }
  }
);

/* =========================================================
   TAROT DIRECTORY CACHE
   ========================================================= */

let tarotDirectoryCache =
  null;

const TAROT_DIRECTORY_TTL =
  10 *
  60 *
  1000;

async function listTarotFiles(
  forceRefresh =
    false
) {
  if (
    !forceRefresh &&
    tarotDirectoryCache &&
    Date.now() -
      tarotDirectoryCache
        .timestamp <
      TAROT_DIRECTORY_TTL
  ) {
    return tarotDirectoryCache
      .files;
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
      "/tarot no es un directorio válido."
    );
  }

  const files =
    data
      .filter(
        (
          item
        ) =>
          item &&
          item.type ===
            "file" &&
          typeof item.name ===
            "string" &&
          /\.(jpg|jpeg|png|webp)$/i.test(
            item.name
          )
      )
      .map(
        (
          item
        ) => ({
          name:
            item.name,

          sha:
            String(
              item.sha ||
              ""
            ),

          size:
            Number(
              item.size ||
              0
            ),
        })
      );

  tarotDirectoryCache = {
    files,

    timestamp:
      Date.now(),
  };

  console.log(
    `Tarot: ${files.length} imágenes encontradas.`
  );

  return files;
}

/* =========================================================
   FIND TAROT FILE
   ========================================================= */

function findTarotFile(
  number,
  files
) {
  const regex =
    new RegExp(
      `juli[\\s_-]*0*${number}\\b`,
      "i"
    );

  return (
    files.find(
      (
        file
      ) =>
        regex.test(
          file.name
        )
    ) ||
    null
  );
}

/* =========================================================
   ORIGINAL TAROT CACHE
   ========================================================= */

const tarotOriginalCache =
  new Map();

const TAROT_ORIGINAL_TTL =
  5 *
  60 *
  1000;

async function fetchTarotOriginal(
  number,
  filename
) {
  const cached =
    tarotOriginalCache.get(
      number
    );

  if (
    cached &&
    Date.now() -
      cached.timestamp <
      TAROT_ORIGINAL_TTL
  ) {
    return cached;
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
    )}/contents/tarot/${encodeURIComponent(
      filename
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
            "application/vnd.github.raw",
        },
      }
    );

  if (
    !response.ok
  ) {
    const text =
      await response.text();

    throw new Error(
      `No se pudo descargar ${filename}: ${text}`
    );
  }

  const buffer =
    Buffer.from(
      await response.arrayBuffer()
    );

  let width =
    0;

  let height =
    0;

  try {
    const image =
      await loadImage(
        buffer
      );

    width =
      image.width;

    height =
      image.height;
  } catch (
    error
  ) {
    console.error(
      "No se pudieron leer dimensiones:",
      filename,
      error
    );
  }

  const entry = {
    buffer,

    timestamp:
      Date.now(),

    mimeType:
      mimeForPath(
        filename
      ),

    width,

    height,
  };

  tarotOriginalCache.set(
    number,
    entry
  );

  return entry;
}

/* =========================================================
   TAROT ORIGINAL

   /tarot/25
   ========================================================= */

app.get(
  "/tarot/:n",
  async (
    req,
    res
  ) => {
    try {
      const raw =
        getParamString(
          req.params.n
        );

      if (
        !/^\d+$/.test(
          raw
        )
      ) {
        return res
          .status(400)
          .send(
            "n inválido"
          );
      }

      const number =
        Number.parseInt(
          raw,
          10
        );

      const files =
        await listTarotFiles();

      const match =
        findTarotFile(
          number,
          files
        );

      if (
        !match
      ) {
        return res
          .status(404)
          .send(
            `carta no encontrada: ${number}`
          );
      }

      const original =
        await fetchTarotOriginal(
          number,
          match.name
        );

      res.setHeader(
        "Content-Type",
        original.mimeType
      );

      res.setHeader(
        "Cache-Control",
        "public, max-age=86400"
      );

      res.setHeader(
        "Access-Control-Allow-Origin",
        "*"
      );

      res.setHeader(
        "Content-Length",
        String(
          original.buffer
            .length
        )
      );

      return res.send(
        original.buffer
      );
    } catch (
      error
    ) {
      console.error(
        "Tarot original error:",
        error
      );

      return res
        .status(500)
        .send(
          "error"
        );
    }
  }
);

/* =========================================================
   TAROT RESPONSIVE

   /tarot/:w/:n

   EJ:
   /tarot/200/25
   /tarot/400/25
   /tarot/800/25
   ========================================================= */

app.get(
  "/tarot/:w/:n",
  async (
    req,
    res
  ) => {
    try {
      const requestedWidth =
        Number.parseInt(
          getParamString(
            req.params.w
          ),
          10
        );

      const number =
        Number.parseInt(
          getParamString(
            req.params.n
          ),
          10
        );

      if (
        !Number.isFinite(
          requestedWidth
        ) ||
        requestedWidth <=
          0 ||
        !Number.isFinite(
          number
        ) ||
        number <=
          0
      ) {
        return res
          .status(400)
          .send(
            "params inválidos"
          );
      }

      const width =
        Math.min(
          requestedWidth,
          4096
        );

      const files =
        await listTarotFiles();

      const match =
        findTarotFile(
          number,
          files
        );

      if (
        !match
      ) {
        return res
          .status(404)
          .send(
            `carta no encontrada: ${number}`
          );
      }

      const original =
        await fetchTarotOriginal(
          number,
          match.name
        );

      res.setHeader(
        "Access-Control-Allow-Origin",
        "*"
      );

      res.setHeader(
        "Vary",
        "Accept"
      );

      /*
       * IMPORTANTE:
       *
       * Como el manifest usa SHA para detectar cambios,
       * se puede usar cache largo para las variantes.
       */

      res.setHeader(
        "Cache-Control",
        "public, max-age=31536000, immutable"
      );

      /*
       * SIN UPSCALE
       */

      if (
        !original.width ||
        width >=
          original.width
      ) {
        res.setHeader(
          "Content-Type",
          original.mimeType
        );

        res.setHeader(
          "Content-Length",
          String(
            original.buffer
              .length
          )
        );

        res.setHeader(
          "X-Tarot-Original-Width",
          String(
            original.width ||
            0
          )
        );

        res.setHeader(
          "X-Tarot-Output-Width",
          String(
            original.width ||
            0
          )
        );

        return res.send(
          original.buffer
        );
      }

      /*
       * RESIZE
       */

      const image =
        await loadImage(
          original.buffer
        );

      const outputWidth =
        width;

      const outputHeight =
        Math.max(
          1,
          Math.round(
            image.height *
              (
                outputWidth /
                image.width
              )
          )
        );

      const canvas =
        createCanvas(
          outputWidth,
          outputHeight
        );

      const context =
        canvas.getContext(
          "2d"
        );

      context.imageSmoothingEnabled =
        true;

      try {
        context.imageSmoothingQuality =
          "high";
      } catch {}

      context.drawImage(
        image,
        0,
        0,
        outputWidth,
        outputHeight
      );

      const accept =
        String(
          req.headers.accept ||
          ""
        );

      /*
       * NO DECLARAMOS:
       *
       * let outputBuffer: Buffer
       *
       * Esto evita la incompatibilidad
       * Buffer<ArrayBuffer> /
       * Buffer<ArrayBufferLike>.
       */

      if (
        accept.includes(
          "image/webp"
        )
      ) {
        const encoded =
          await canvas.encode(
            "webp",
            92
          );

        res.setHeader(
          "Content-Type",
          "image/webp"
        );

        res.setHeader(
          "Content-Length",
          String(
            encoded.byteLength
          )
        );

        res.setHeader(
          "X-Tarot-Original-Width",
          String(
            original.width
          )
        );

        res.setHeader(
          "X-Tarot-Output-Width",
          String(
            outputWidth
          )
        );

        return res.send(
          encoded
        );
      }

      const encoded =
        await canvas.encode(
          "jpeg",
          92
        );

      res.setHeader(
        "Content-Type",
        "image/jpeg"
      );

      res.setHeader(
        "Content-Length",
        String(
          encoded.byteLength
        )
      );

      res.setHeader(
        "X-Tarot-Original-Width",
        String(
          original.width
        )
      );

      res.setHeader(
        "X-Tarot-Output-Width",
        String(
          outputWidth
        )
      );

      return res.send(
        encoded
      );
    } catch (
      error
    ) {
      console.error(
        "Tarot resize error:",
        error
      );

      return res
        .status(500)
        .send(
          "error"
        );
    }
  }
);

/* =========================================================
   TAROT MANIFEST
   ========================================================= */

app.get(
  "/tarot-manifest",
  async (
    _req,
    res
  ) => {
    try {
      const {
        owner,
        repo,
        branch,
        token,
      } =
        ghConfig();

      /*
       * MANIFEST SIEMPRE FRESCO
       */

      const files =
        await listTarotFiles(
          true
        );

      let commitSha =
        null;

      try {
        const response =
          await fetch(
            `https://api.github.com/repos/${encodeURIComponent(
              owner
            )}/${encodeURIComponent(
              repo
            )}/branches/${encodeURIComponent(
              branch
            )}`,
            {
              headers:
                githubHeaders(
                  token
                ),
            }
          );

        if (
          response.ok
        ) {
          const data =
            await response.json();

          commitSha =
            data &&
            data.commit &&
            data.commit.sha
              ? data.commit.sha
              : null;
        }
      } catch (
        error
      ) {
        console.error(
          "Commit SHA error:",
          error
        );
      }

      const cards =
        files
          .map(
            (
              file
            ) => {
              const match =
                file.name.match(
                  /juli[\s_-]*0*(\d+)\b/i
                );

              if (
                !match
              ) {
                return null;
              }

              return {
                juli:
                  Number.parseInt(
                    match[1],
                    10
                  ),

                sha:
                  file.sha,

                size:
                  file.size,

                name:
                  file.name,
              };
            }
          )
          .filter(
            Boolean
          )
          .sort(
            (
              a,
              b
            ) =>
              a.juli -
              b.juli
          );

      res.setHeader(
        "Cache-Control",
        "no-store"
      );

      res.setHeader(
        "Access-Control-Allow-Origin",
        "*"
      );

      return res.json({
        ok: true,

        commitSha,

        branch,

        count:
          cards.length,

        cards,
      });
    } catch (
      error
    ) {
      console.error(
        "Tarot manifest error:",
        error
      );

      return res
        .status(500)
        .json({
          ok: false,

          error:
            error?.message ||
            "error",
        });
    }
  }
);

/* =========================================================
   PDF FIRST PAGE -> PNG
   ========================================================= */

async function pdfFirstPageToPng(
  buffer
) {
  class NodeCanvasFactory {
    create(
      width,
      height
    ) {
      const canvas =
        createCanvas(
          Math.ceil(
            width
          ),
          Math.ceil(
            height
          )
        );

      const context =
        canvas.getContext(
          "2d"
        );

      return {
        canvas,
        context,
      };
    }

    reset(
      item,
      width,
      height
    ) {
      item.canvas.width =
        width;

      item.canvas.height =
        height;
    }

    destroy(
      item
    ) {
      item.canvas.width =
        0;

      item.canvas.height =
        0;

      item.context =
        null;

      item.canvas =
        null;
    }
  }

  let pdf =
    null;

  try {
    const task =
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
      await task.promise;

    const page =
      await pdf.getPage(
        1
      );

    const base =
      page.getViewport({
        scale:
          1,
      });

    const maxSide =
      1600;

    const scale =
      Math.min(
        2,

        maxSide /
          Math.max(
            base.width,
            base.height
          )
      );

    const viewport =
      page.getViewport({
        scale,
      });

    const factory =
      new NodeCanvasFactory();

    const item =
      factory.create(
        viewport.width,
        viewport.height
      );

    await page.render({
      canvasContext:
        item.context,

      viewport,

      canvasFactory:
        factory,
    }).promise;

    const encoded =
      await item.canvas.encode(
        "png"
      );

    factory.destroy(
      item
    );

    return encoded;
  } catch (
    error
  ) {
    console.error(
      "PDF render error:",
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
   ANALYZE RECEIPT
   ========================================================= */

app.post(
  "/analyze-receipt",
  auth,
  async (
    req,
    res
  ) => {
    try {
      const {
        fileUrl,
        prompt,
      } =
        req.body || {};

      if (
        !fileUrl ||
        !prompt
      ) {
        return res
          .status(400)
          .json({
            error:
              "fileUrl y prompt son requeridos",
          });
      }

      if (
        !GEMINI_API_KEY
      ) {
        return res
          .status(500)
          .json({
            error:
              "GEMINI_API_KEY no configurada",
          });
      }

      const fileResponse =
        await fetch(
          fileUrl
        );

      if (
        !fileResponse.ok
      ) {
        return res
          .status(400)
          .json({
            error:
              `No se pudo descargar el archivo (${fileResponse.status})`,
          });
      }

      const originalBuffer =
        Buffer.from(
          await fileResponse.arrayBuffer()
        );

      const detectedMime =
        (
          fileResponse.headers.get(
            "content-type"
          ) ||
          ""
        )
          .split(";")[0]
          .trim()
          .toLowerCase();

      const isPdf =
        detectedMime ===
          "application/pdf" ||
        String(
          fileUrl
        )
          .toLowerCase()
          .endsWith(
            ".pdf"
          );

      let mimeType =
        detectedMime &&
        !detectedMime.includes(
          "octet-stream"
        )
          ? detectedMime
          : isPdf
            ? "application/pdf"
            : "image/jpeg";

      /*
       * No tipamos dataBuffer como Buffer
       * deliberadamente.
       */

      let dataBuffer =
        originalBuffer;

      if (
        isPdf
      ) {
        const png =
          await pdfFirstPageToPng(
            originalBuffer
          );

        if (
          png
        ) {
          mimeType =
            "image/png";

          dataBuffer =
            png;
        }
      }

      const base64 =
        Buffer.from(
          dataBuffer
        ).toString(
          "base64"
        );

      const geminiUrl =
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
          GEMINI_MODEL
        )}:generateContent?key=${encodeURIComponent(
          GEMINI_API_KEY
        )}`;

      const geminiResponse =
        await fetch(
          geminiUrl,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify({
                contents: [
                  {
                    parts: [
                      {
                        text:
                          prompt +
                          "\n\nDevolvé ÚNICAMENTE un JSON válido, sin markdown ni texto adicional.",
                      },

                      {
                        inline_data: {
                          mime_type:
                            mimeType,

                          data:
                            base64,
                        },
                      },
                    ],
                  },
                ],

                generation_config: {
                  response_mime_type:
                    "application/json",

                  temperature:
                    0.1,
                },
              }),
          }
        );

      if (
        !geminiResponse.ok
      ) {
        const text =
          await geminiResponse.text();

        return res
          .status(502)
          .json({
            error:
              "Gemini error: " +
              text,
          });
      }

      const geminiJson =
        await geminiResponse.json();

      const text =
        geminiJson &&
        geminiJson.candidates &&
        geminiJson.candidates[0] &&
        geminiJson
          .candidates[0]
          .content &&
        geminiJson
          .candidates[0]
          .content.parts &&
        geminiJson
          .candidates[0]
          .content.parts[0] &&
        geminiJson
          .candidates[0]
          .content.parts[0]
          .text
          ? geminiJson
              .candidates[0]
              .content.parts[0]
              .text
          : "";

      let parsed =
        null;

      try {
        parsed =
          JSON.parse(
            text
          );
      } catch {
        const match =
          text.match(
            /\{[\s\S]*\}/
          );

        if (
          match
        ) {
          try {
            parsed =
              JSON.parse(
                match[0]
              );
          } catch {}
        }
      }

      return res.json({
        ok: true,

        detected:
          parsed,

        raw:
          text,
      });
    } catch (
      error
    ) {
      console.error(
        "Analyze receipt error:",
        error
      );

      return res
        .status(500)
        .json({
          error:
            error?.message ||
            "unknown error",
        });
    }
  }
);

/* =========================================================
   GENERATE TEXT
   ========================================================= */

app.post(
  "/generate-text",
  auth,
  async (
    req,
    res
  ) => {
    try {
      const {
        prompt,
      } =
        req.body || {};

      if (
        !prompt
      ) {
        return res
          .status(400)
          .json({
            error:
              "prompt requerido",
          });
      }

      if (
        !GEMINI_API_KEY
      ) {
        return res
          .status(500)
          .json({
            error:
              "GEMINI_API_KEY no configurada",
          });
      }

      const geminiUrl =
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
          GEMINI_MODEL
        )}:generateContent?key=${encodeURIComponent(
          GEMINI_API_KEY
        )}`;

      const geminiResponse =
        await fetch(
          geminiUrl,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify({
                contents: [
                  {
                    parts: [
                      {
                        text:
                          prompt,
                      },
                    ],
                  },
                ],

                generation_config: {
                  temperature:
                    0.85,

                  max_output_tokens:
                    2400,

                  top_p:
                    0.95,
                },
              }),
          }
        );

      if (
        !geminiResponse.ok
      ) {
        const text =
          await geminiResponse.text();

        return res
          .status(502)
          .json({
            error:
              "Gemini error: " +
              text,
          });
      }

      const geminiJson =
        await geminiResponse.json();

      const candidate =
        geminiJson &&
        geminiJson
          .candidates &&
        geminiJson
          .candidates[0];

      const text =
        candidate &&
        candidate.content &&
        candidate.content.parts &&
        candidate
          .content
          .parts[0] &&
        candidate
          .content
          .parts[0]
          .text
          ? candidate
              .content
              .parts[0]
              .text
          : "";

      if (
        !text &&
        candidate &&
        candidate.finishReason &&
        candidate.finishReason !==
          "STOP"
      ) {
        return res
          .status(502)
          .json({
            error:
              "Gemini finishReason: " +
              candidate
                .finishReason,
          });
      }

      return res.json({
        ok: true,

        text,
      });
    } catch (
      error
    ) {
      console.error(
        "Generate text error:",
        error
      );

      return res
        .status(500)
        .json({
          error:
            error?.message ||
            "unknown error",
        });
    }
  }
);

/* =========================================================
   404
   ========================================================= */

app.use(
  (
    _req,
    res
  ) => {
    return res
      .status(404)
      .json({
        ok: false,

        error:
          "Endpoint no encontrado",
      });
  }
);

/* =========================================================
   START SERVER
   ========================================================= */

app.listen(
  PORT,
  "0.0.0.0",
  () => {
    console.log(
      "======================================"
    );

    console.log(
      "Torchill API iniciada"
    );

    console.log(
      `Puerto: ${PORT}`
    );

    console.log(
      `Gemini: ${Boolean(
        GEMINI_API_KEY
      )}`
    );

    console.log(
      `Modelo: ${GEMINI_MODEL}`
    );

    console.log(
      "--------------------------------------"
    );

    console.log(
      "GET  /health"
    );

    console.log(
      "POST /upload"
    );

    console.log(
      "GET  /receipt/:pathB64"
    );

    console.log(
      "GET  /tarot/:n"
    );

    console.log(
      "GET  /tarot/:w/:n"
    );

    console.log(
      "GET  /tarot-manifest"
    );

    console.log(
      "POST /analyze-receipt"
    );

    console.log(
      "POST /generate-text"
    );

    console.log(
      "======================================"
    );
  }
);
