import express, {
  Request,
  Response,
  NextFunction,
} from "express";

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
   ENVIRONMENT
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
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (!API_TOKEN) {
    return next();
  }

  const apiKeyHeader =
    req.headers["x-api-key"];

  const authorization =
    req.headers.authorization ||
    "";

  const sent =
    typeof apiKeyHeader ===
    "string"
      ? apiKeyHeader
      : authorization.replace(
          /^Bearer\s+/i,
          ""
        );

  if (sent !== API_TOKEN) {
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
  value:
    | string
    | string[]
    | undefined
): string {
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

interface GithubConfig {
  owner: string;
  repo: string;
  branch: string;
  token: string;
}

function ghConfig():
  GithubConfig {
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
  token: string
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
   MIME
   ========================================================= */

function mimeForPath(
  path: string
): string {
  const lower =
    (path || "")
      .toLowerCase();

  if (
    lower.endsWith(".pdf")
  ) {
    return "application/pdf";
  }

  if (
    lower.endsWith(".png")
  ) {
    return "image/png";
  }

  if (
    lower.endsWith(".webp")
  ) {
    return "image/webp";
  }

  if (
    lower.endsWith(".jpg") ||
    lower.endsWith(".jpeg")
  ) {
    return "image/jpeg";
  }

  return "application/octet-stream";
}

/* =========================================================
   ENCODE GITHUB PATH
   ========================================================= */

function encodeGithubPath(
  path: string
): string {
  return path
    .split("/")
    .map((part) =>
      encodeURIComponent(part)
    )
    .join("/");
}

/* =========================================================
   HEALTH
   ========================================================= */

app.get(
  "/health",
  (
    _req: Request,
    res: Response
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
    _req: Request,
    res: Response
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
   UPLOAD RECEIPT A GITHUB
   ========================================================= */

app.post(
  "/upload",
  async (
    req: Request,
    res: Response
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

      let cleanBase64 =
        String(base64);

      cleanBase64 =
        cleanBase64.replace(
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

      const ghRes =
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

      if (!ghRes.ok) {
        const errText =
          await ghRes.text();

        return res
          .status(502)
          .json({
            error:
              "GitHub error: " +
              errText,
          });
      }

      const pathB64 =
        Buffer.from(
          path
        ).toString(
          "base64url"
        );

      const rel =
        `/receipt/${pathB64}`;

      const absoluteUrl =
        `${req.protocol}://${req.get(
          "host"
        )}${rel}`;

      return res.json({
        ok: true,

        path,

        pathB64,

        fileUrl:
          absoluteUrl,

        url:
          rel,

        receiptUrl:
          rel,

        absoluteUrl,
      });
    } catch (
      error: unknown
    ) {
      return res
        .status(500)
        .json({
          error:
            error instanceof
            Error
              ? error.message
              : "unknown error",
        });
    }
  }
);

/* =========================================================
   RECEIPT PRIVADO
   ========================================================= */

app.get(
  "/receipt/:pathB64",
  async (
    req: Request,
    res: Response
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

      const rawUrl =
        `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${encodeGithubPath(
          path
        )}`;

      const response =
        await fetch(
          rawUrl,
          {
            headers: {
              Authorization:
                `Bearer ${token}`,

              Accept:
                "application/vnd.github.raw",
            },
          }
        );

      if (!response.ok) {
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
        mimeForPath(path)
      );

      res.setHeader(
        "Cache-Control",
        "private, max-age=3600"
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
   TAROT TYPES
   ========================================================= */

interface TarotGithubFile {
  name: string;
  sha: string;
  size: number;
}

interface TarotDirectoryCache {
  files:
    TarotGithubFile[];

  timestamp:
    number;
}

interface TarotOriginal {
  buffer:
    Buffer;

  timestamp:
    number;

  mimeType:
    string;

  width:
    number;

  height:
    number;
}

/* =========================================================
   TAROT CACHE
   ========================================================= */

let tarotDirectoryCache:
  TarotDirectoryCache | null =
  null;

const TAROT_DIRECTORY_TTL =
  10 *
  60 *
  1000;

const tarotOriginalCache =
  new Map<
    number,
    TarotOriginal
  >();

const TAROT_ORIGINAL_TTL =
  5 *
  60 *
  1000;

/* =========================================================
   LIST TAROT DIRECTORY
   ========================================================= */

async function listTarotFiles(
  forceRefresh =
    false
): Promise<
  TarotGithubFile[]
> {
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

  if (!response.ok) {
    const text =
      await response.text();

    throw new Error(
      `No se pudo listar /tarot (${response.status}): ${text}`
    );
  }

  const data =
    await response.json();

  if (
    !Array.isArray(data)
  ) {
    throw new Error(
      "/tarot no es un directorio válido."
    );
  }

  const files:
    TarotGithubFile[] =
    data
      .filter(
        (item: any) =>
          item?.type ===
            "file" &&
          typeof item.name ===
            "string" &&
          /\.(jpg|jpeg|png|webp)$/i.test(
            item.name
          )
      )
      .map(
        (item: any) => ({
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

  return files;
}

/* =========================================================
   FIND TAROT FILE
   ========================================================= */

function findTarotFile(
  number: number,
  files:
    TarotGithubFile[]
):
  TarotGithubFile | null {
  const regex =
    new RegExp(
      `juli[\\s_-]*0*${number}\\b`,
      "i"
    );

  return (
    files.find(
      (file) =>
        regex.test(
          file.name
        )
    ) ||
    null
  );
}

/* =========================================================
   DOWNLOAD ORIGINAL TAROT
   ========================================================= */

async function fetchTarotOriginal(
  juli: number,
  filename: string
): Promise<
  TarotOriginal
> {
  const cached =
    tarotOriginalCache.get(
      juli
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

  const rawUrl =
    `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/tarot/${encodeURIComponent(
      filename
    )}`;

  const response =
    await fetch(
      rawUrl,
      {
        headers: {
          Authorization:
            `Bearer ${token}`,

          Accept:
            "application/vnd.github.raw",
        },
      }
    );

  if (!response.ok) {
    throw new Error(
      `No se pudo descargar ${filename}.`
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

  const original:
    TarotOriginal = {
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
    juli,
    original
  );

  return original;
}

/* =========================================================
   TAROT ORIGINAL
   ========================================================= */

app.get(
  "/tarot/:n",
  async (
    req: Request,
    res: Response
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

      if (!match) {
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
        "Tarot error:",
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

   Ejemplos:

   /tarot/160/25
   /tarot/320/25
   /tarot/640/25

   Sin upscale.
   WebP cuando sea posible.
   JPEG fallback.
   ========================================================= */

app.get(
  "/tarot/:w/:n",
  async (
    req: Request,
    res: Response
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
        number <= 0
      ) {
        return res
          .status(400)
          .send(
            "params inválidos"
          );
      }

      /*
       * Protección.
       */
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

      if (!match) {
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

      /*
       * Cache 1 año.
       */
      res.setHeader(
        "Cache-Control",
        "public, max-age=31536000, immutable"
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
       * SIN UPSCALE.
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
            original.width
          )
        );

        res.setHeader(
          "X-Tarot-Output-Width",
          String(
            original.width
          )
        );

        return res.send(
          original.buffer
        );
      }

      /*
       * Resize.
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
        (
          context as any
        ).imageSmoothingQuality =
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

      let outputBuffer:
        Buffer;

      let outputMime:
        string;

      /*
       * WEBP preferido.
       */
      if (
        accept.includes(
          "image/webp"
        )
      ) {
        outputBuffer =
  Buffer.from(
    await canvas.encode(
      "webp",
      92
    )
  );

        outputMime =
          "image/webp";
      } else {
        outputBuffer =
  Buffer.from(
    await canvas.encode(
      "jpeg",
      92
    )
  );

        outputMime =
          "image/jpeg";
      }

      res.setHeader(
        "Content-Type",
        outputMime
      );

      res.setHeader(
        "Content-Length",
        String(
          outputBuffer.length
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
        outputBuffer
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
    _req: Request,
    res: Response
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
       * Forzar lectura fresca.
       */
      const files =
        await listTarotFiles(
          true
        );

      let commitSha:
        string | null =
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
            data
              ?.commit
              ?.sha ||
            null;
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
            (file) => {
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
            (
              card
            ): card is {
              juli: number;
              sha: string;
              size: number;
              name: string;
            } =>
              card !==
              null
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
        "Manifest error:",
        error
      );

      return res
        .status(500)
        .json({
          ok: false,

          error:
            error instanceof
            Error
              ? error.message
              : "error",
        });
    }
  }
);

/* =========================================================
   PDF FIRST PAGE -> PNG
   ========================================================= */

async function pdfFirstPageToPng(
  buffer: Buffer
): Promise<
  Buffer | null
> {
  class NodeCanvasFactory {
    create(
      width: number,
      height: number
    ) {
      const canvas =
        createCanvas(
          width,
          height
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
      item: any,
      width: number,
      height: number
    ) {
      item.canvas.width =
        width;

      item.canvas.height =
        height;
    }

    destroy(
      item: any
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

  let pdf:
    any = null;

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
        scale: 1,
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

    const result =
      factory.create(
        viewport.width,
        viewport.height
      );

    await page.render({
      canvasContext:
        result.context,

      viewport,

      canvasFactory:
        factory as any,
    }).promise;

    const png =
      await result.canvas.encode(
        "png"
      );

    factory.destroy(
      result
    );

    return png;
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
      if (pdf) {
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
    req: Request,
    res: Response
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

      const fileRes =
        await fetch(
          fileUrl
        );

      if (!fileRes.ok) {
        return res
          .status(400)
          .json({
            error:
              `No se pudo descargar el archivo (${fileRes.status})`,
          });
      }

      const buffer =
        Buffer.from(
          await fileRes.arrayBuffer()
        );

      const detectedMime =
        (
          fileRes.headers.get(
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

      let dataBuffer =
        buffer;

      if (isPdf) {
        const png =
          await pdfFirstPageToPng(
            buffer
          );

        if (png) {
          mimeType =
            "image/png";

          dataBuffer =
            png;
        }
      }

      const base64 =
        dataBuffer.toString(
          "base64"
        );

      const geminiRes =
        await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
            GEMINI_MODEL
          )}:generateContent?key=${encodeURIComponent(
            GEMINI_API_KEY
          )}`,
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

      if (!geminiRes.ok) {
        const text =
          await geminiRes.text();

        return res
          .status(502)
          .json({
            error:
              "Gemini error: " +
              text,
          });
      }

      const geminiJson =
        await geminiRes.json();

      const text =
        geminiJson
          ?.candidates?.[0]
          ?.content
          ?.parts?.[0]
          ?.text ||
        "";

      let parsed:
        any =
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

        if (match) {
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
      error: unknown
    ) {
      return res
        .status(500)
        .json({
          error:
            error instanceof
            Error
              ? error.message
              : "unknown error",
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
    req: Request,
    res: Response
  ) => {
    try {
      const {
        prompt,
      } =
        req.body || {};

      if (!prompt) {
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

      const geminiRes =
        await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
            GEMINI_MODEL
          )}:generateContent?key=${encodeURIComponent(
            GEMINI_API_KEY
          )}`,
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

      if (!geminiRes.ok) {
        const text =
          await geminiRes.text();

        return res
          .status(502)
          .json({
            error:
              "Gemini error: " +
              text,
          });
      }

      const geminiJson =
        await geminiRes.json();

      const candidate =
        geminiJson
          ?.candidates?.[0];

      const text =
        candidate
          ?.content
          ?.parts?.[0]
          ?.text ||
        "";

      if (
        !text &&
        candidate
          ?.finishReason &&
        candidate
          .finishReason !==
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
      error: unknown
    ) {
      return res
        .status(500)
        .json({
          error:
            error instanceof
            Error
              ? error.message
              : "unknown error",
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
   SERVER
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
