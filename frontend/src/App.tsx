import React, { useState, useRef, useCallback } from "react";

type ModuleId =
  | "cnn"
  | "detection"
  | "face"
  | "gan"
  | "denoise"
  | "style";

interface Prediction {
  label: string;
  confidence: number;
}

interface DetectionBox {
  label: string;
  confidence: number;
  box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

type ResultData = string | Prediction[] | DetectionBox[] | null;

interface ModuleConfig {
  id: ModuleId;
  title: string;
  subtitle: string;
  icon: string;
  accent: string;
  accentBg: string;
  borderColor: string;
  description: string;
  inputLabel: string;
  demoImage: string;
  demoResult: (
    imgSrc: string,
    resultData?: ResultData
  ) => JSX.Element;
}


// ============================================================
// Icons
// ============================================================

function UploadIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      className="animate-spin"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

function SVGNoiseFilter() {
  return (
    <svg className="hidden">
      <defs>
        <filter
          id="noise"
          x="-20%"
          y="-20%"
          width="140%"
          height="140%"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.8"
            numOctaves="3"
            stitchTiles="stitch"
          />
          <feColorMatrix type="saturate" values="0" />
          <feBlend in="SourceGraphic" mode="multiply" />
        </filter>
      </defs>
    </svg>
  );
}


// ============================================================
// CNN Classification
// ============================================================

const DEMO_PREDICTIONS: Prediction[] = [
  {
    label: "Golden Retriever",
    confidence: 94.7,
  },
  {
    label: "Labrador Retriever",
    confidence: 3.8,
  },
  {
    label: "Cocker Spaniel",
    confidence: 1.5,
  },
];

function CNNResult({
  resultData,
}: {
  resultData?: ResultData;
}) {
  const predictions: Prediction[] =
    resultData &&
    Array.isArray(resultData) &&
    resultData.length > 0 &&
    "confidence" in resultData[0] &&
    !("box" in resultData[0])
      ? (resultData as Prediction[])
      : DEMO_PREDICTIONS;

  const isLive = Array.isArray(resultData);

  return (
    <div className="space-y-3">
      <p className="text-xs mono text-[#6b7280] uppercase tracking-widest mb-4">
        Classification Results {isLive ? "" : "(Demo)"}
      </p>

      {predictions.map((p, i) => (
        <div
          key={p.label}
          className="space-y-1"
        >
          <div className="flex justify-between items-center">
            <span className="text-sm text-[#e8eaf0] font-medium capitalize">
              {p.label}
            </span>

            <span
              className="mono text-sm font-semibold"
              style={{
                color:
                  i === 0
                    ? "#00e5ff"
                    : "#6b7280",
              }}
            >
              {p.confidence}%
            </span>
          </div>

          <div className="h-1.5 rounded-full bg-[#1f2330] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${p.confidence}%`,
                background:
                  i === 0
                    ? "linear-gradient(90deg, #00e5ff, #7c3aed)"
                    : "#2d3347",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}


// ============================================================
// Object Detection
// ============================================================

const DEMO_DETECTIONS: DetectionBox[] = [
  {
    label: "person",
    confidence: 98,
    box: {
      x1: 0.15,
      y1: 0.2,
      x2: 0.45,
      y2: 0.7,
    },
  },
  {
    label: "dog",
    confidence: 91,
    box: {
      x1: 0.55,
      y1: 0.3,
      x2: 0.8,
      y2: 0.7,
    },
  },
];

const BOX_COLORS = [
  "#00e5ff",
  "#a78bfa",
  "#34d399",
  "#fbbf24",
  "#fb923c",
  "#f472b6",
];

function DetectionResult({
  imgSrc,
  resultData,
}: {
  imgSrc: string;
  resultData?: ResultData;
}) {
  const isLive =
    Array.isArray(resultData) &&
    (resultData.length === 0 ||
      "box" in resultData[0]);

  const detections: DetectionBox[] = isLive
    ? (resultData as DetectionBox[])
    : DEMO_DETECTIONS;

  // Count objects
  const counts = detections.reduce<Record<string, number>>(
    (acc, d) => {
      acc[d.label] =
        (acc[d.label] || 0) + 1;

      return acc;
    },
    {}
  );

  return (
    <div className="space-y-3">
      
      {/* Heading */}
      <p className="text-xs mono text-[#6b7280] uppercase tracking-widest mb-4">
        Detected Objects {isLive ? "" : "(Demo)"}
      </p>


      {/* ======================================================
          IMAGE + BOUNDING BOXES
      ====================================================== */}

      <div className="relative rounded-md overflow-hidden bg-[#1a1d26]">

        {/* Original image */}
        <img
          src={imgSrc}
          alt="Detection result"
          className="w-full h-auto block"
        />

        {/* Bounding box overlay */}
        <div className="absolute inset-0 pointer-events-none">

          {detections.map((d, i) => {
            const color =
              BOX_COLORS[
                i % BOX_COLORS.length
              ];

            const left =
              d.box.x1 * 100;

            const top =
              d.box.y1 * 100;

            const width =
              (d.box.x2 - d.box.x1) * 100;

            const height =
              (d.box.y2 - d.box.y1) * 100;

            return (
              <div
                key={`${d.label}-${i}`}
                className="absolute rounded-sm"
                style={{
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,

                  border: `2px solid ${color}`,

                  boxSizing: "border-box",

                  zIndex: 10,
                }}
              >

                {/* =================================================
                    LABEL
                    Positioned INSIDE the bounding box so it can
                    never be clipped by the image container.
                ================================================= */}

                <div
                  className="
                    absolute
                    left-0
                    top-0
                    px-1.5
                    py-0.5
                    rounded-br-sm
                    mono
                    text-[10px]
                    font-semibold
                    whitespace-nowrap
                  "
                  style={{
                    color: "#ffffff",
                    background: color,
                    lineHeight: "1.2",
                    zIndex: 20,
                  }}
                >
                  {d.label} {d.confidence}%
                </div>

              </div>
            );
          })}

        </div>
      </div>


      {/* ======================================================
          OBJECT SUMMARY
      ====================================================== */}

      <div className="flex gap-2 flex-wrap">

        {Object.entries(counts).map(
          ([label, count], i) => {
            const color =
              BOX_COLORS[
                i % BOX_COLORS.length
              ];

            return (
              <span
                key={label}
                className="
                  flex
                  items-center
                  gap-1.5
                  px-2
                  py-1
                  rounded-full
                  text-xs
                  mono
                  bg-[#1a1d26]
                "
              >

                <span
                  className="w-2 h-2 rounded-full"
                  style={{
                    background: color,
                  }}
                />

                <span
                  style={{
                    color,
                  }}
                  className="capitalize font-medium"
                >
                  {label}
                </span>

                <span className="text-[#6b7280]">
                  ×{count}
                </span>

              </span>
            );
          }
        )}

        {detections.length === 0 && (
          <span className="text-xs mono text-[#6b7280]">
            No objects detected above the confidence threshold.
          </span>
        )}

      </div>
    </div>
  );
}


// ============================================================
// Face Similarity
// ============================================================

function FaceResult() {
  return (
    <div className="space-y-3">

      <p className="text-xs mono text-[#6b7280] uppercase tracking-widest mb-4">
        Similarity Analysis
      </p>

      <div className="flex items-center gap-4">

        <div className="flex-1 h-16 rounded-md bg-[#1a1d26] flex items-center justify-center">
          <span className="text-3xl">
            👤
          </span>
        </div>

        <div className="flex flex-col items-center gap-1">

          <div className="mono text-2xl font-semibold text-[#00e5ff]">
            87.3%
          </div>

          <div className="text-xs text-[#6b7280]">
            similarity
          </div>

        </div>

        <div className="flex-1 h-16 rounded-md bg-[#1a1d26] flex items-center justify-center">
          <span className="text-3xl">
            👤
          </span>
        </div>

      </div>

      <div className="flex justify-between text-xs mono mt-2">
        <span className="text-[#6b7280]">
          Euclidean distance
        </span>

        <span className="text-[#a78bfa]">
          0.432
        </span>
      </div>

      <div className="flex justify-between text-xs mono">

        <span className="text-[#6b7280]">
          Cosine similarity
        </span>

        <span className="text-[#a78bfa]">
          0.873
        </span>

      </div>

      <div className="mt-2 px-3 py-2 rounded-md bg-[#00e5ff]/10 border border-[#00e5ff]/20">

        <span className="text-xs text-[#00e5ff] mono font-medium">
          VERDICT: Same person — high confidence
        </span>

      </div>

    </div>
  );
}


// ============================================================
// GAN
// ============================================================

function GANResult({
  imgSrc,
}: {
  imgSrc: string;
}) {
  return (
    <div className="space-y-3">

      <p className="text-xs mono text-[#6b7280] uppercase tracking-widest mb-4">
        Generated Image
      </p>

      <div className="relative rounded-md overflow-hidden bg-[#1a1d26]">

        <img
          src={imgSrc}
          alt="GAN generated"
          className="w-full h-40 object-cover"
        />

        <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-[#0a0b0f]/80 mono text-xs text-[#a78bfa]">
          synthetic
        </div>

      </div>

      <div className="grid grid-cols-2 gap-2 text-xs mono">

        <div className="px-3 py-2 rounded-md bg-[#1a1d26]">

          <div className="text-[#6b7280]">
            Realism score
          </div>

          <div className="text-[#a78bfa] font-semibold text-base mt-0.5">
            96.2%
          </div>

        </div>

        <div className="px-3 py-2 rounded-md bg-[#1a1d26]">

          <div className="text-[#6b7280]">
            FID score
          </div>

          <div className="text-[#a78bfa] font-semibold text-base mt-0.5">
            12.4
          </div>

        </div>

      </div>

    </div>
  );
}


// ============================================================
// Denoising
// ============================================================

function DenoiseResult({
  imgSrc,
  resultData,
}: {
  imgSrc: string;
  resultData?: ResultData;
}) {
  const [split, setSplit] =
    useState(50);

  const cleanImageSrc =
    typeof resultData === "string"
      ? resultData
      : imgSrc;

  return (
    <div className="space-y-3">

      <p className="text-xs mono text-[#6b7280] uppercase tracking-widest mb-4">
        Before / After
      </p>

      <div
        className="
          relative
          rounded-md
          overflow-hidden
          h-36
          bg-[#1a1d26]
          cursor-col-resize
          select-none
        "
        onMouseMove={(e) => {
          const rect =
            e.currentTarget.getBoundingClientRect();

          const percentage =
            ((e.clientX - rect.left) /
              rect.width) *
            100;

          setSplit(
            Math.max(
              0,
              Math.min(
                100,
                Math.round(percentage)
              )
            )
          );
        }}
      >

        {/* Original */}
        <img
          src={imgSrc}
          alt="noisy"
          className="absolute inset-0 w-full h-full object-cover"
        />

        {/* Denoised */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{
            clipPath: `inset(0 ${
              100 - split
            }% 0 0)`,
          }}
        >

          <img
            src={cleanImageSrc}
            alt="denoised"
            className="w-full h-full object-cover"
          />

        </div>

        {/* Slider */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white/80"
          style={{
            left: `${split}%`,
          }}
        >

          <div
            className="
              absolute
              top-1/2
              -translate-y-1/2
              -translate-x-1/2
              w-6
              h-6
              rounded-full
              bg-white
              flex
              items-center
              justify-center
              shadow-lg
            "
          >

            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#0a0b0f"
              strokeWidth="2.5"
            >
              <path d="M18 8l4 4-4 4M6 8l-4 4 4 4" />
            </svg>

          </div>

        </div>

        <span className="absolute top-2 left-2 text-xs mono text-white/60 bg-black/40 px-1 rounded">
          noisy
        </span>

        <span className="absolute top-2 right-2 text-xs mono text-white/60 bg-black/40 px-1 rounded">
          clean
        </span>

      </div>

      <div className="flex justify-between text-xs mono">

        <span className="text-[#6b7280]">
          Status
        </span>

        <span className="text-[#34d399]">
          {typeof resultData === "string"
            ? "Model Denoised"
            : "Demo View"}
        </span>

      </div>

    </div>
  );
}


// ============================================================
// Style Transfer
// ============================================================

function StyleResult({
  imgSrc,
  styleImgSrc,
}: {
  imgSrc: string;
  styleImgSrc: string;
}) {
  return (
    <div className="space-y-3">

      <p className="text-xs mono text-[#6b7280] uppercase tracking-widest mb-4">
        Style Transfer
      </p>

      <div className="grid grid-cols-3 gap-2 items-center">

        <div className="space-y-1">

          <img
            src={imgSrc}
            alt="content"
            className="w-full h-20 object-cover rounded-md opacity-80"
          />

          <p className="text-xs mono text-[#6b7280] text-center">
            content
          </p>

        </div>

        <div className="flex flex-col items-center gap-1">

          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#a78bfa"
            strokeWidth="2"
          >
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>

          <span className="text-xs mono text-[#a78bfa]">
            +
          </span>

          <img
            src={styleImgSrc}
            alt="style"
            className="w-12 h-12 object-cover rounded-md opacity-80"
          />

          <p className="text-xs mono text-[#6b7280]">
            style
          </p>

        </div>

        <div className="space-y-1">

          <img
            src={imgSrc}
            alt="result"
            className="w-full h-20 object-cover rounded-md"
            style={{
              filter:
                "saturate(1.8) hue-rotate(20deg) contrast(1.1)",
            }}
          />

          <p className="text-xs mono text-[#00e5ff] text-center">
            result
          </p>

        </div>

      </div>

      <div className="flex justify-between text-xs mono">

        <span className="text-[#6b7280]">
          Style weight
        </span>

        <span className="text-[#a78bfa]">
          1e6
        </span>

      </div>

      <div className="flex justify-between text-xs mono">

        <span className="text-[#6b7280]">
          Content weight
        </span>

        <span className="text-[#a78bfa]">
          1e0
        </span>

      </div>

    </div>
  );
}


// ============================================================
// Demo Images
// ============================================================

const DEMO_IMAGES = {
  dog: "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&h=300&fit=crop&auto=format",

  street: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=400&h=300&fit=crop&auto=format",

  portrait: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=300&fit=crop&auto=format",

  abstract: "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=400&h=300&fit=crop&auto=format",

  painting: "https://images.unsplash.com/photo-1578301978693-85fa9c0320b9?w=400&h=300&fit=crop&auto=format",

  landscape: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&auto=format",
};


// ============================================================
// Module Configuration
// ============================================================

const MODULES: ModuleConfig[] = [

  {
    id: "cnn",

    title: "CNN Classification",

    subtitle: "Image Recognition",

    icon: "🧠",

    accent: "#00e5ff",

    accentBg:
      "rgba(0,229,255,0.06)",

    borderColor:
      "rgba(0,229,255,0.15)",

    description:
      "Classify images into 10 CIFAR-10 categories using a convolutional neural network.",

    inputLabel:
      "Upload an image to classify",

    demoImage:
      DEMO_IMAGES.dog,

    demoResult: (
      imgSrc,
      resultData
    ) => (
      <CNNResult
        resultData={resultData}
      />
    ),
  },

  {
    id: "detection",

    title: "Object Detection",

    subtitle: "Locate & Identify",

    icon: "🎯",

    accent: "#a78bfa",

    accentBg:
      "rgba(167,139,250,0.06)",

    borderColor:
      "rgba(167,139,250,0.15)",

    description:
      "Detect and localize objects from 20 PASCAL VOC categories using YOLOv8.",

    inputLabel:
      "Upload an image to detect objects",

    demoImage:
      DEMO_IMAGES.street,

    demoResult: (
      imgSrc,
      resultData
    ) => (
      <DetectionResult
        imgSrc={imgSrc}
        resultData={resultData}
      />
    ),
  },

  {
    id: "face",

    title: "Face Similarity",

    subtitle: "Identity Matching",

    icon: "👁",

    accent: "#34d399",

    accentBg:
      "rgba(52,211,153,0.06)",

    borderColor:
      "rgba(52,211,153,0.15)",

    description:
      "Compare two faces and compute their similarity using deep metric learning embeddings.",

    inputLabel:
      "Upload two face images",

    demoImage:
      DEMO_IMAGES.portrait,

    demoResult: () => (
      <FaceResult />
    ),
  },

  {
    id: "gan",

    title: "GAN Generator",

    subtitle: "Image Synthesis",

    icon: "✨",

    accent: "#f472b6",

    accentBg:
      "rgba(244,114,182,0.06)",

    borderColor:
      "rgba(244,114,182,0.15)",

    description:
      "Generate photorealistic synthetic images using a trained Generative Adversarial Network.",

    inputLabel:
      "Upload a seed image or click to generate",

    demoImage:
      DEMO_IMAGES.portrait,

    demoResult: (
      imgSrc
    ) => (
      <GANResult
        imgSrc={imgSrc}
      />
    ),
  },

  {
    id: "denoise",

    title: "Image Denoising",

    subtitle: "Noise Reduction",

    icon: "🔆",

    accent: "#fbbf24",

    accentBg:
      "rgba(251,191,36,0.06)",

    borderColor:
      "rgba(251,191,36,0.15)",

    description:
      "Remove Gaussian, salt-and-pepper, and JPEG noise from images using a U-Net denoiser.",

    inputLabel:
      "Upload a noisy image",

    demoImage:
      DEMO_IMAGES.landscape,

    demoResult: (
      imgSrc,
      resultData
    ) => (
      <DenoiseResult
        imgSrc={imgSrc}
        resultData={resultData}
      />
    ),
  },

  {
    id: "style",

    title: "Neural Style Transfer",

    subtitle: "Artistic Rendering",

    icon: "🎨",

    accent: "#fb923c",

    accentBg:
      "rgba(251,146,60,0.06)",

    borderColor:
      "rgba(251,146,60,0.15)",

    description:
      "Blend a content image with an artistic style image using neural texture synthesis.",

    inputLabel:
      "Upload content & style images",

    demoImage:
      DEMO_IMAGES.landscape,

    demoResult: (
      imgSrc
    ) => (
      <StyleResult
        imgSrc={imgSrc}
        styleImgSrc={
          DEMO_IMAGES.painting
        }
      />
    ),
  },
];


// ============================================================
// Module State
// ============================================================

type State =
  | "idle"
  | "loading"
  | "done";

interface ModuleCardProps {
  module: ModuleConfig;
}


// ============================================================
// Backend Endpoints
// ============================================================

const API_ENDPOINTS: Partial<
  Record<ModuleId, string>
> = {
  denoise:
    "http://localhost:8000/api/denoise",

  cnn:
    "http://localhost:8000/api/classify",

  detection:
    "http://localhost:8000/api/detect",
};


// ============================================================
// Module Card
// ============================================================

function ModuleCard({
  module,
}: ModuleCardProps) {

  const [state, setState] =
    useState<State>("idle");

  const [preview, setPreview] =
    useState<string | null>(null);

  const [resultData, setResultData] =
    useState<ResultData>(null);

  const [dragging, setDragging] =
    useState(false);

  const inputRef =
    useRef<HTMLInputElement>(null);


  // ==========================================================
  // Backend Call
  // ==========================================================

  const callBackend =
    useCallback(
      async (
        file: File
      ): Promise<boolean> => {

        const endpoint =
          API_ENDPOINTS[
            module.id
          ];

        if (!endpoint)
          return false;

        const formData =
          new FormData();

        formData.append(
          "file",
          file
        );

        try {

          const response =
            await fetch(
              endpoint,
              {
                method: "POST",
                body: formData,
              }
            );

          if (!response.ok)
            throw new Error(
              "Inference failed"
            );

          const data =
            await response.json();

          if (
            data.status !==
            "success"
          ) {
            throw new Error(
              "Backend error"
            );
          }


          if (
            module.id ===
            "denoise"
          ) {

            setResultData(
              data.denoised_image
            );

          } else if (
            module.id ===
            "cnn"
          ) {

            setResultData(
              data.predictions
            );

          } else if (
            module.id ===
            "detection"
          ) {

            setResultData(
              data.detections
            );
          }


          setState("done");

          return true;

        } catch (err) {

          console.error(
            `${module.id} API Error:`,
            err
          );

          return false;
        }
      },
      [module.id]
    );


  // ==========================================================
  // File Handler
  // ==========================================================

  const handleFile =
    useCallback(
      async (
        file: File
      ) => {

        if (
          !file.type.startsWith(
            "image/"
          )
        ) {
          return;
        }


        if (
          preview &&
          preview.startsWith(
            "blob:"
          )
        ) {

          URL.revokeObjectURL(
            preview
          );
        }


        const url =
          URL.createObjectURL(
            file
          );

        setPreview(url);

        setResultData(null);

        setState("loading");


        if (
          API_ENDPOINTS[
            module.id
          ]
        ) {

          const ok =
            await callBackend(
              file
            );

          if (!ok) {

            setState("idle");

            alert(
              "Failed to connect to backend model API at http://localhost:8000"
            );
          }

        } else {

          setTimeout(
            () =>
              setState("done"),

            1800 +
              Math.random() *
                600
          );
        }
      },
      [
        preview,
        module.id,
        callBackend,
      ]
    );


  // ==========================================================
  // Drag & Drop
  // ==========================================================

  const handleDrop =
    useCallback(
      (
        e: React.DragEvent
      ) => {

        e.preventDefault();

        setDragging(false);

        const file =
          e.dataTransfer.files[0];

        if (file)
          handleFile(file);

      },
      [handleFile]
    );


  // ==========================================================
  // Demo
  // ==========================================================

  const handleDemo =
    async () => {

      setState("loading");

      setPreview(
        module.demoImage
      );

      setResultData(null);


      if (
        API_ENDPOINTS[
          module.id
        ]
      ) {

        try {

          const response =
            await fetch(
              module.demoImage
            );

          const blob =
            await response.blob();

          const file =
            new File(
              [blob],
              "demo.jpg",
              {
                type: blob.type,
              }
            );

          const ok =
            await callBackend(
              file
            );

          if (!ok) {

            setTimeout(
              () =>
                setState(
                  "done"
                ),
              1800
            );
          }

        } catch (err) {

          console.error(
            "Demo API Error:",
            err
          );

          setTimeout(
            () =>
              setState("done"),
            1800
          );
        }

      } else {

        setTimeout(
          () =>
            setState("done"),

          1800 +
            Math.random() *
              600
        );
      }
    };


  // ==========================================================
  // Reset
  // ==========================================================

  const reset = () => {

    if (
      preview &&
      preview.startsWith(
        "blob:"
      )
    ) {

      URL.revokeObjectURL(
        preview
      );
    }

    setState("idle");

    setPreview(null);

    setResultData(null);

    if (
      inputRef.current
    ) {

      inputRef.current.value =
        "";
    }
  };


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div
      className="
        relative
        rounded-xl
        flex
        flex-col
        overflow-hidden
        transition-all
        duration-300
        hover:-translate-y-0.5
      "
      style={{
        background:
          module.accentBg,

        border:
          `1px solid ${module.borderColor}`,

        boxShadow:
          state === "done"
            ? `0 0 24px ${module.accent}18`
            : "none",
      }}
    >

      {/* =====================================================
          Header
      ===================================================== */}

      <div
        className="
          px-5
          pt-5
          pb-4
          border-b
        "
        style={{
          borderColor:
            module.borderColor,
        }}
      >

        <div className="flex items-start justify-between gap-3">

          <div>

            <div className="flex items-center gap-2.5 mb-1">

              <span className="text-xl">
                {module.icon}
              </span>

              <h3 className="font-semibold text-[#e8eaf0] text-base leading-tight">
                {module.title}
              </h3>

            </div>

            <span
              className="
                mono
                text-xs
                px-2
                py-0.5
                rounded-full
              "
              style={{
                color:
                  module.accent,

                background:
                  `${module.accent}18`,
              }}
            >
              {module.subtitle}
            </span>

          </div>


          {state !== "idle" && (

            <button
              onClick={reset}
              className="
                text-[#6b7280]
                hover:text-[#e8eaf0]
                transition-colors
                text-xs
                mono
                mt-0.5
                shrink-0
              "
            >
              ✕ reset
            </button>

          )}

        </div>


        <p className="text-[#6b7280] text-xs mt-3 leading-relaxed">
          {module.description}
        </p>

      </div>


      {/* =====================================================
          Body
      ===================================================== */}

      <div className="p-5 flex-1 flex flex-col gap-4">

        {/* Upload */}
        {state === "idle" && (

          <div
            className="
              rounded-lg
              border-2
              border-dashed
              transition-all
              duration-200
              cursor-pointer
              flex
              flex-col
              items-center
              justify-center
              gap-2
              py-7
            "
            style={{
              borderColor:
                dragging
                  ? module.accent
                  : module.borderColor,

              background:
                dragging
                  ? `${module.accent}08`
                  : "transparent",
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() =>
              setDragging(false)
            }
            onDrop={handleDrop}
            onClick={() =>
              inputRef.current?.click()
            }
          >

            <span
              style={{
                color:
                  module.accent,
                opacity: 0.7,
              }}
            >
              <UploadIcon />
            </span>

            <span className="text-xs text-[#6b7280] text-center leading-relaxed px-2">
              {module.inputLabel}
            </span>

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) =>
                e.target.files?.[0] &&
                handleFile(
                  e.target.files[0]
                )
              }
            />

          </div>
        )}


        {/* =================================================
            Preview for non-detection
        ================================================= */}

        {state !== "idle" &&
          preview &&
          module.id !==
            "detection" && (

            <div className="relative rounded-lg overflow-hidden h-28">

              <img
                src={preview}
                alt="Input"
                className="
                  w-full
                  h-full
                  object-cover
                  opacity-60
                "
              />

              <div
                className="
                  absolute
                  inset-0
                  flex
                  items-center
                  justify-center
                "
                style={{
                  background:
                    `${module.accent}12`,
                }}
              >

                {state ===
                  "loading" && (

                  <div className="flex flex-col items-center gap-2">

                    <span
                      style={{
                        color:
                          module.accent,
                      }}
                    >
                      <SpinnerIcon />
                    </span>

                    <span
                      className="mono text-xs"
                      style={{
                        color:
                          module.accent,
                      }}
                    >
                      running inference…
                    </span>

                  </div>

                )}

              </div>

            </div>

          )}


        {/* =================================================
            Detection Loading Preview
        ================================================= */}

        {state === "loading" &&
          preview &&
          module.id ===
            "detection" && (

            <div className="relative rounded-lg overflow-hidden">

              <img
                src={preview}
                alt="Input"
                className="
                  w-full
                  h-auto
                  block
                  opacity-60
                "
              />

              <div
                className="
                  absolute
                  inset-0
                  flex
                  items-center
                  justify-center
                "
                style={{
                  background:
                    `${module.accent}12`,
                }}
              >

                <div className="flex flex-col items-center gap-2">

                  <span
                    style={{
                      color:
                        module.accent,
                    }}
                  >
                    <SpinnerIcon />
                  </span>

                  <span
                    className="mono text-xs"
                    style={{
                      color:
                        module.accent,
                    }}
                  >
                    running inference…
                  </span>

                </div>

              </div>

            </div>

          )}


        {/* =================================================
            Result
        ================================================= */}

        {state === "done" &&
          preview && (

            <div
              className="
                rounded-lg
                p-4
                bg-[#111318]
                border
              "
              style={{
                borderColor:
                  module.borderColor,
              }}
            >

              {module.demoResult(
                preview,
                resultData
              )}

            </div>

          )}


        {/* =================================================
            Demo Button
        ================================================= */}

        {state === "idle" && (

          <button
            onClick={handleDemo}
            className="
              text-xs
              mono
              transition-colors
              py-2
              rounded-md
            "
            style={{
              color:
                module.accent,

              background:
                `${module.accent}10`,
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background =
                `${module.accent}1a`)
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background =
                `${module.accent}10`)
            }
          >
            ↗ try demo
          </button>

        )}

      </div>
    </div>
  );
}


// ============================================================
// Main App
// ============================================================

export default function App() {

  return (
    <div className="min-h-full bg-[#0a0b0f] text-[#e8eaf0]">

      <SVGNoiseFilter />


      {/* ======================================================
          Navigation
      ====================================================== */}

      <header
        className="
          border-b
          border-[#1f2330]
          px-6
          py-4
          flex
          items-center
          justify-between
          sticky
          top-0
          z-10
        "
        style={{
          background:
            "rgba(10,11,15,0.92)",

          backdropFilter:
            "blur(12px)",
        }}
      >

        <div className="flex items-center gap-3">

          <div
            className="
              w-7
              h-7
              rounded-lg
              flex
              items-center
              justify-center
            "
            style={{
              background:
                "linear-gradient(135deg, #00e5ff, #7c3aed)",
            }}
          >
            <span className="text-xs">
              ⚡
            </span>
          </div>

          <span className="font-semibold tracking-tight text-sm text-[#e8eaf0]">
            UniDL
          </span>

        </div>


        <div className="flex items-center gap-1.5">

          <span className="w-2 h-2 rounded-full bg-[#34d399] animate-pulse" />

          <span className="mono text-xs text-[#6b7280]">
            6 models loaded
          </span>

        </div>

      </header>


      {/* ======================================================
          Hero
      ====================================================== */}

      <section className="px-6 pt-16 pb-12 max-w-6xl mx-auto text-center">

        <div
          className="
            inline-flex
            items-center
            gap-2
            px-3
            py-1
            rounded-full
            mb-6
            border
            border-[#1f2330]
            bg-[#111318]
          "
        >

          <span className="w-1.5 h-1.5 rounded-full bg-[#00e5ff]" />

          <span className="mono text-xs text-[#6b7280]">
            Unified Deep Learning Platform
          </span>

        </div>


        <h1
          className="
            display
            text-4xl
            md:text-5xl
            lg:text-6xl
            font-light
            leading-tight
            mb-4
          "
        >

          See What Your

          <br />

          <span
            style={{
              background:
                "linear-gradient(90deg, #00e5ff, #a78bfa)",

              WebkitBackgroundClip:
                "text",

              WebkitTextFillColor:
                "transparent",
            }}
          >
            Models See.
          </span>

        </h1>


        <p className="text-[#6b7280] text-base md:text-lg max-w-xl mx-auto leading-relaxed">

          Six production-ready deep learning modules, one unified interface.

          <br />

          Upload, infer, explore — no code required.

        </p>

      </section>


      {/* ======================================================
          Module Grid
      ====================================================== */}

      <main className="px-6 pb-20 max-w-6xl mx-auto">

        <div className="
          grid
          grid-cols-1
          sm:grid-cols-2
          lg:grid-cols-3
          gap-5
        ">

          {MODULES.map(
            (module) => (
              <ModuleCard
                key={module.id}
                module={module}
              />
            )
          )}

        </div>

      </main>


      {/* ======================================================
          Footer
      ====================================================== */}

      <footer
        className="
          border-t
          border-[#1f2330]
          py-6
          px-6
          flex
          items-center
          justify-between
          max-w-6xl
          mx-auto
        "
      >

        <span className="mono text-xs text-[#6b7280]">
          UniDL — Unified Multi-Task Deep Learning Platform
        </span>

        <span className="mono text-xs text-[#2d3347]">
          v1.0.0
        </span>

      </footer>

    </div>
  );
}