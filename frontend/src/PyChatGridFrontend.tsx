import * as React from "react";
import { useSectionSocket } from "./useSection";
import { DatasetPicker } from "./DatasetPicker";
import MDMpic from "./assets/MDM.png";
import MoMASKpic from "./assets/MoMASK.png";
import MotionAgentpic from "./assets/MotionAgent.png";
import MotionCLIPpic from "./assets/MotionCLIP.png";
import MotionDiffusepic from "./assets/MotionDiffuse.png";
import MotionGPT3pic from "./assets/MotionGPT3.png";
import T2MGPTpic from "./assets/T2MGPT.png";

// ── Types ─────────────────────────────────────────────────────────────────────

type ChatMessage = {
  id: string;
  align: "left" | "right";
  type: "text" | "video" | "gif";
  text?: string;
  name?: string;
  url?: string;
};

type ScriptSection = {
  id: string;
  title: string;
  bg: string;
  messages: ChatMessage[];
  draft: string;
  files: string[];
  videos: string[];
  image?: string;
  acceptsFile: boolean;
  multiModal: boolean;
  inBetween: boolean;
  inBetweenAcceptsFile: boolean;
};

// ── Message factories ─────────────────────────────────────────────────────────

export function createTextMessage(
  text: string,
  align: "left" | "right" = "right"
): ChatMessage {
  return { id: crypto.randomUUID(), align, type: "text", text };
}

// ── Section data ──────────────────────────────────────────────────────────────

export function createBaseSections() {
  return [
    { id: "01", title: "MotionDiffuse", bg: "bg-slate-100",  image: MotionDiffusepic, acceptsFile: false, multiModal: false, inBetween: false, inBetweenAcceptsFile: false },
    { id: "02", title: "MotionGPT3",    bg: "bg-violet-50",  image: MotionGPT3pic,    acceptsFile: true,  multiModal: true, inBetween: false, inBetweenAcceptsFile: false },
    { id: "03", title: "MDM",           bg: "bg-emerald-50", image: MDMpic,           acceptsFile: false, multiModal: false, inBetween: true, inBetweenAcceptsFile: false },
    { id: "04", title: "T2M-GPT",       bg: "bg-amber-50",   image: T2MGPTpic,        acceptsFile: false, multiModal: false, inBetween: false, inBetweenAcceptsFile: false },
    { id: "05", title: "MO-Mask",       bg: "bg-rose-50",    image: MoMASKpic,        acceptsFile: false,  multiModal: false, inBetween: true, inBetweenAcceptsFile: true },
    { id: "06", title: "MotionCLIP",    bg: "bg-sky-50",     image: MotionCLIPpic,    acceptsFile: false,  multiModal: false, inBetween: false, inBetweenAcceptsFile: false },
    { id: "07", title: "MotionAgent",   bg: "bg-fuchsia-50", image: MotionAgentpic,   acceptsFile: true,  multiModal: false, inBetween: false, inBetweenAcceptsFile: false },
  ];
}

export function buildInitialSections(): ScriptSection[] {
  return createBaseSections().map((section) => ({
    ...section,
    messages: [],
    draft: "",
    files: [],
    videos: [],
  }));
}

// ── Sub-components ────────────────────────────────────────────────────────────

function MessageBubble({
  align = "left", children,
}: {
  align?: "left" | "right";
  children: React.ReactNode;
}) {
  const right = align === "right";
  return (
    <div className={`flex ${right ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[82%] rounded-2xl px-4 py-3 text-sm shadow-sm ${
          right
            ? "bg-blue-600 text-white"
            : "border border-slate-200 bg-white text-slate-700"
        }`}
      >
        {children}
      </div>
    </div>
  );
}

async function forceDownload(url: string, filename: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Failed to download file");
  }

  const blob = await response.blob();
  const objectUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(objectUrl);
}

function ensureExtension(filename: string, ext: string) {
  return filename.toLowerCase().endsWith(ext) ? filename : `${filename}${ext}`;
}

function VideoCard({ name, url }: { name: string; url?: string }) {
  if (url) {
    return (
      <div className="w-full max-w-sm overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <video controls autoPlay muted playsInline loop className="w-full aspect-video bg-slate-900">
          <source src={url} type="video/mp4" />
        </video>
        <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-3 py-2 text-sm text-slate-700">
          <span className="truncate">{name}</span>
          <button
            type="button"
            onClick={() => { void forceDownload(url, ensureExtension(name, ".mp4")); }}
            className="shrink-0 rounded-lg border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-100"
          >
            Download
          </button>
        </div>
      </div>
    );
  }
  return (
    <div className="w-full max-w-sm overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex aspect-video items-center justify-center bg-slate-900 text-4xl text-white">
        ▶
      </div>
      <div className="border-t border-slate-200 px-3 py-2 text-sm text-slate-700">{name}</div>
    </div>
  );
}

function GifCard({ name, url }: { name: string; url: string }) {
  return (
    <div className="w-full max-w-sm overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <img src={url} alt={name} className="w-full" />
      <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-3 py-2 text-sm text-slate-700">
        <span className="truncate">{name}</span>
        <button
          type="button"
          onClick={() => { void forceDownload(url, ensureExtension(name, ".gif")); }}
          className="shrink-0 rounded-lg border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-100"
        >
          Download
        </button>
      </div>
    </div>
  );
}

function MessageBubbleContent({ message }: { message: ChatMessage }) {
  if (message.type === "text") return <span>{message.text}</span>;
  if (message.type === "video") return <VideoCard name={message.name || "output.mp4"} url={message.url} />;
  if (message.type === "gif") return <GifCard name={message.name || "output.gif"} url={message.url!} />;
  return null;
}

// ── ScriptCard ────────────────────────────────────────────────────────────────

function ScriptCard({
  section, tall = false, onDraftChange, onSend, onIncoming, onFilenameSelected, userid,active, onRegisterSend, onReady
}: {
  section: ScriptSection;
  tall?: boolean;
  onDraftChange: (sectionId: string, value: string) => void;
  onSend: (sectionId: string) => void;
  onIncoming: (sectionId: string, raw: string) => void;
  onFilenameSelected: (sectionId: string, filename: string, videoPath?: string) => void;
  userid: number;
  active: boolean;
  onRegisterSend: (sectionId: string, sendFn: (msg: string) => void) => void;
  onReady: (sectionId: string) => void;
}) {
  const [pickerOpen, setPickerOpen] = React.useState(false);
  const [ready, setReady] = React.useState(false);
  const [processing, setProcessing] = React.useState(false);
  const [pendingFile, setPendingFile] = React.useState<string | null>(null);
  const [pendingDatasetId, setPendingDatasetId] = React.useState<number | null>(null);
  const disabled = !ready || processing;
  const fileDisabled = disabled || !section.acceptsFile;

  const { send } = useSectionSocket(
    section.id, userid, active,
    (text) => onIncoming(section.id, text),
    () => { setReady(true); onReady(section.id); },
    () => setProcessing(false)
  );

  const [showTaskPicker, setShowTaskPicker] = React.useState(false);

  function handleSend() {
    const hasFile = !!pendingFile;
    const hasText = !!section.draft.trim();
    if (!hasFile && !hasText || !ready || processing) return;

    // Only show task picker if section supports multiModal
    if (hasFile && hasText && section.multiModal) {
      setShowTaskPicker(true);
      return;
    }

    submitMessage(null);
  }

  function submitMessage(task: "t2m" | "m2t" | null) {
    const hasFile = !!pendingFile;
    const hasText = !!section.draft.trim();

    let parts = [];
    parts.push("MODE:direct");
    if (hasFile) parts.push(`FILE:${pendingFile}|DATASET:${pendingDatasetId}`);
    if (hasText) parts.push(`TEXT:${section.draft.trim()}`);
    if (task)    parts.push(`TASK:${task}`);

    send(parts.join("|"));

    if (hasFile) onFilenameSelected(section.id, pendingFile!);
    if (hasText) onSend(section.id);

    setProcessing(true);  // grey out until response arrives
    setPendingFile(null);
    setShowTaskPicker(false);
  }

  async function handleFilePicked(filename: string, datasetId: number) {
    setPendingFile(filename);
    setPendingDatasetId(datasetId);
    setPickerOpen(false);

    // Fetch rendered visualization
    const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
    try {
      const res = await fetch(`${apiUrl}/render?filename=${encodeURIComponent(filename)}&dataset_id=${datasetId}`);
      if (res.ok) {
        const { path } = await res.json();
        onFilenameSelected(section.id, filename, path);
      } else {
        onFilenameSelected(section.id, filename); // fallback to text
      }
    } catch {
      onFilenameSelected(section.id, filename); // fallback to text
    }
  }

  React.useEffect(() => {
    onRegisterSend(section.id, send);
  }, [send]);


  return (
    <section className={`${section.bg} flex h-[640px] flex-col rounded-3xl border border-white/70 shadow-[0_18px_50px_rgba(15,23,42,0.08)] backdrop-blur ${tall ? "md:col-span-2" : ""}`}>
      {/* Header */}
      <div className="flex items-center justify-between gap-4 rounded-t-3xl border-b border-slate-200/70 bg-white/65 px-4 py-3 backdrop-blur">
        <div className="font-mono text-sm font-semibold text-slate-700">{section.title}</div>
      </div>

      {/* Messages */}
      <div className="flex flex-1 flex-col-reverse overflow-y-auto min-h-0 relative px-3 py-2">
        {section.image && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage: `url(${section.image})`,
              backgroundSize: "contain",
              backgroundRepeat: "no-repeat",
              backgroundPosition: "center",
              opacity: 0.15,
            }}
          />
        )}
        <div className="relative flex flex-col gap-4">
          {section.messages.map((message) => (
            <MessageBubble key={message.id} align={message.align}>
              <MessageBubbleContent message={message} />
            </MessageBubble>
          ))}
        </div>
      </div>

      {/* Pending file chip */}
      {pendingFile && (
        <div className="flex items-center gap-2 px-3 pt-2">
          <div className="flex items-center gap-2 rounded-2xl border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs text-blue-700 font-medium">
            <span>📄 {pendingFile}</span>
            <button
              onClick={() => setPendingFile(null)}
              className="text-blue-400 hover:text-blue-700 transition-colors leading-none"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Input row */}
      <div className="relative flex items-center gap-2 border-t border-slate-200/70 px-3 py-3">
        {showTaskPicker && (
          <div className="absolute bottom-[calc(100%+8px)] left-0 right-0 z-50 mx-3 rounded-2xl border border-slate-200 bg-white/95 backdrop-blur shadow-xl p-3">
            <p className="text-xs font-semibold text-slate-600 mb-2">Select task</p>
            <div className="flex gap-2">
              <button
                onClick={() => submitMessage("t2m")}
                className="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300 transition-all"
              >
                🏃 T2M
                <span className="block text-xs text-slate-400 font-normal">Text → Motion</span>
              </button>
              <button
                onClick={() => submitMessage("m2t")}
                className="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300 transition-all"
              >
                📝 M2T
                <span className="block text-xs text-slate-400 font-normal">Motion → Text</span>
              </button>
            </div>
          </div>
        )}
        {pickerOpen && (
          <div className="absolute bottom-[calc(100%+8px)] left-0 right-0 z-50">
            <DatasetPicker
              onFilePicked={handleFilePicked}
              onClose={() => setPickerOpen(false)}
            />
          </div>
        )}

        <button
          onClick={() => !fileDisabled && setPickerOpen((v) => !v)}
          className={`flex h-10 w-10 items-center justify-center rounded-2xl border shadow-sm transition-all ${
            fileDisabled
              ? "border-slate-200 bg-slate-100 text-slate-300 cursor-not-allowed opacity-40"
              : pickerOpen || pendingFile
              ? "border-blue-400 bg-blue-50 text-blue-600"
              : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          📎
        </button>

        <input
          value={section.draft}
          onChange={(e) => onDraftChange(section.id, e.target.value)}
          placeholder={pendingFile ? "Add a text prompt (optional)..." : "Type a message..."}
          className="flex-1 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-inner outline-none placeholder:text-slate-400"
        />
        <button
          onClick={handleSend}
          disabled={disabled}
          className={`flex h-10 w-10 items-center justify-center rounded-2xl ${disabled ? 'bg-slate-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white shadow-sm hover:bg-blue-700`}
        >
          ➜
        </button>
      </div>
    </section>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function PyChatGridFrontend() {
  const [activeSections, setActiveSections] = React.useState<string[]>([]);
  const [userid] = React.useState(() => Math.floor(Math.random() * 90000) + 10000);
  const [broadcastText, setBroadcastText] = React.useState("");
  const [broadcastPendingFile, setBroadcastPendingFile] = React.useState<string | null>(null);
  const [broadcastMode, setBroadcastMode] = React.useState<"broadcast" | "inbetween">("broadcast");
  const [broadcastDatasetId, setBroadcastDatasetId] = React.useState<number | null>(null);
  const [broadcastPickerOpen, setBroadcastPickerOpen] = React.useState(false);
  const [broadcastTaskPicker, setBroadcastTaskPicker] = React.useState(false);
  const sendRefs = React.useRef<Record<string, (msg: string) => void>>({});
  const renderedFiles = React.useRef<Set<string>>(new Set());
  const onRegisterSend = React.useCallback((sectionId: string, sendFn: (msg: string) => void) => {
  sendRefs.current[sectionId] = sendFn;
  }, []);
  const [sections, setSections] = React.useState<ScriptSection[]>(() =>
    buildInitialSections()
  );
  const hasActivity = sections.some((s) => s.messages.length > 0);
  const [readySections, setReadySections] = React.useState<Set<string>>(new Set());
  const broadcastReady = activeSections.length > 0 && 
    activeSections.every((id) => readySections.has(id));

  // State for feedback dialog
  const [showFeedback, setShowFeedback] = React.useState(false);
  const [feedbackRating, setFeedbackRating] = React.useState<"good" | "ok" | "bad" | null>(null);
  const [feedbackComment, setFeedbackComment] = React.useState("");

  async function submitFeedback() {
    const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
    await fetch(`${apiUrl}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userid, rating: feedbackRating, comment: feedbackComment }),
    });
    setShowFeedback(false);
    //window.close();
    // If window.close() didn't work, show message after a short delay
    setTimeout(() => {
      alert("Session ended. You can now close this tab.");
    }, 300);
  }

  React.useEffect(() => {
    const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
    fetch(`${apiUrl}/sections`)
      .then((r) => r.json())
      .then((data) => setActiveSections(data.active));
  }, []);

  function updateDraft(sectionId: string, value: string) {
    setSections((prev) =>
      prev.map((s) => (s.id === sectionId ? { ...s, draft: value } : s))
    );
  }

  const MAX_MESSAGES = 61; // 20 queries × 2 (input + output)

  function trimMessages(messages: ChatMessage[]): ChatMessage[] {
    if (messages.length > MAX_MESSAGES) {
      return messages.slice(messages.length - MAX_MESSAGES);
    }
    return messages;
  }

  function sendSectionMessage(sectionId: string) {
    setSections((prev) =>
      prev.map((s) => {
        if (s.id !== sectionId || !s.draft.trim()) return s;
        return {
          ...s,
          messages: trimMessages([...s.messages, createTextMessage(s.draft.trim())]),
          draft: "",
        };
      })
    );
  }

  // Called when user picks a filename from the dataset picker
  function addFilenameMessage(sectionId: string, filename: string, videoPath?: string) {
    setSections((prev) =>
      prev.map((s) => {
        if (s.id !== sectionId) return s;
        const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
        const renderedName = videoPath?.split("/").pop() || ensureExtension(filename, ".mp4");
        const message: ChatMessage = videoPath ? {
          id: crypto.randomUUID(),
          align: "right",
          type: "video",
          name: renderedName,
          url: `${apiUrl}/video?path=${encodeURIComponent(videoPath)}`,
        } : {
          id: crypto.randomUUID(),
          align: "right",
          type: "text",
          text: `📄 ${filename}`,
        };
        return {
          ...s,
          messages: trimMessages([...s.messages, message]),
        };
      })
    );
  }

  // Called when Python sends a line via stdout
  function addIncomingMessage(sectionId: string, raw: string) {
    if (raw === "TEXT:ready") return; // silently ignored
    if (raw.startsWith("ERROR:")) return;
    setSections((prev) =>
      prev.map((s) => {
        if (s.id !== sectionId) return s;

        let message: ChatMessage;

        if (raw.startsWith("FILE:mp4:")) {
          const path = raw.slice("FILE:mp4:".length);
          const apiUrl =
            (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
          message = {
            id: crypto.randomUUID(),
            align: "left",
            type: "video",
            name: path.split("/").pop() || "output.mp4",
            url: `${apiUrl}/video?path=${encodeURIComponent(path)}`,
          };
        } else if (raw.startsWith("FILE:gif:")) {
          const path = raw.slice("FILE:gif:".length);
          const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
          message = {
            id: crypto.randomUUID(),
            align: "left",
            type: "gif",
            name: path.split("/").pop() || "output.gif",
            url: `${apiUrl}/video?path=${encodeURIComponent(path)}`,
          };
        } else {
          const text = raw.startsWith("TEXT:") ? raw.slice(5) : raw;
          message = createTextMessage(text, "left");
        }

        return { ...s, messages: trimMessages([...s.messages, message]) };
      })
    );
  }

  function handleBroadcast() {
    setBroadcastMode("broadcast");
    const hasFile = !!broadcastPendingFile;
    const hasText = !!broadcastText.trim();
    if (!hasFile && !hasText) return;

    const hasMultiModal = sections.some((s) => s.multiModal && activeSections.includes(s.id));
    if (hasFile && hasText && hasMultiModal) {
      setBroadcastTaskPicker(true);
      return;
    }
    submitBroadcast(null);
  }

  function submitBroadcast(task: "t2m" | "m2t" | null) {
    const hasFile = !!broadcastPendingFile;
    const hasText = !!broadcastText.trim();

    sections.forEach((s) => {
      if (!activeSections.includes(s.id)) return;
      if (hasFile && !s.acceptsFile) return; // for testing only
      let parts = [];
      if (s.acceptsFile) {
        parts.push("MODE:broadcast");
        if (hasFile) parts.push(`FILE:${broadcastPendingFile}|DATASET:${broadcastDatasetId}`);
        if (hasText) parts.push(`TEXT:${broadcastText.trim()}`);
        if (task && s.multiModal) parts.push(`TASK:${task}`);
      } else {
        parts.push("MODE:broadcast");
        if (hasText) parts.push(`TEXT:${broadcastText.trim()}`);
      }
      if (parts.length > 0) sendRefs.current[s.id]?.(parts.join("|"));
    });

    setSections((prev) =>
      prev.map((s) => {
        if (!activeSections.includes(s.id)) return s;
        const hasFileForThis = hasFile && s.acceptsFile;
        if (!hasFileForThis && !hasText) return s;

        let newMessages = [...s.messages];
        if (hasFileForThis) {
          if (!renderedFiles.current.has(broadcastPendingFile!)) {
            newMessages = trimMessages([...newMessages, {
              id: crypto.randomUUID(),
              align: "right" as const,
              type: "text" as const,
              text: `📄 ${broadcastPendingFile}`,
            }]);
          }
        }
        if (hasText) {
          newMessages = trimMessages([...newMessages, createTextMessage(broadcastText.trim(), "right")]);
        }
        return { ...s, messages: newMessages };
      })
    );

    setBroadcastText("");
    setBroadcastPendingFile(null);
    setBroadcastTaskPicker(false);
  }

  function handleInBetween() {
    setBroadcastMode("inbetween");
    const hasFile = !!broadcastPendingFile;
    const hasText = !!broadcastText.trim();
    if (!hasFile && !hasText) return;

    const hasMultiModal = sections.some((s) => 
      s.inBetween && s.multiModal && activeSections.includes(s.id)
    );
    if (hasFile && hasText && hasMultiModal) {
      setBroadcastTaskPicker(true);  // reuse same task picker
      return;
    }
    submitInBetween(null);
  }

  function submitInBetween(task: "t2m" | "m2t" | null) {
    const hasFile = !!broadcastPendingFile;
    const hasText = !!broadcastText.trim();

    sections.forEach((s) => {
      if (!activeSections.includes(s.id)) return;
      if (!s.inBetween) return;  // only in-between models

      let parts = [];
      parts.push("MODE:inbetween");
      if (s.inBetweenAcceptsFile) {
        if (hasFile) parts.push(`FILE:${broadcastPendingFile}|DATASET:${broadcastDatasetId}`);
        if (hasText) parts.push(`TEXT:${broadcastText.trim()}`);
        if (task && s.multiModal) parts.push(`TASK:${task}`);
      } else {
        if (hasText) parts.push(`TEXT:${broadcastText.trim()}`);
      }
      if (parts.length > 0) sendRefs.current[s.id]?.(parts.join("|"));
    });

    setSections((prev) =>
      prev.map((s) => {
        if (!activeSections.includes(s.id) || !s.inBetween) return s;
        const hasFileForThis = hasFile && s.inBetweenAcceptsFile;
        if (!hasFileForThis && !hasText) return s;

        let newMessages = [...s.messages];
        if (hasFileForThis && !renderedFiles.current.has(broadcastPendingFile!)) {
          newMessages = trimMessages([...newMessages, {
            id: crypto.randomUUID(),
            align: "right" as const,
            type: "text" as const,
            text: `📄 ${broadcastPendingFile}`,
          }]);
        }
        if (hasText) {
          newMessages = trimMessages([...newMessages, createTextMessage(broadcastText.trim(), "right")]);
        }
        return { ...s, messages: newMessages };
      })
    );

    setBroadcastText("");
    setBroadcastPendingFile(null);
    setBroadcastTaskPicker(false);
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.09),_transparent_28%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_100%)] text-slate-900">
      <header className="sticky top-0 z-50 border-b border-slate-200/70 bg-slate-950/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] items-center gap-4 px-5 py-4">
          <div className="flex min-w-fit items-center gap-3 text-white">
            <button
              onClick={() => { if (hasActivity) setShowFeedback(true); }}
              className={`rounded-2xl border px-3 py-2 text-sm font-semibold transition ${
                hasActivity
                  ? "border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20"
                  : "border-white/5 bg-white/5 text-slate-600 cursor-not-allowed"
              }`}
            >
              End Session
            </button>
          </div>
          <div className="flex flex-1 items-center gap-3 relative">
            {broadcastPickerOpen && (
              <div className="fixed top-[79px] right-4 z-[100] w-72">
                <DatasetPicker
                  onFilePicked={async (filename, datasetId) => {
                    setBroadcastPendingFile(filename);
                    setBroadcastDatasetId(datasetId);
                    setBroadcastPickerOpen(false);
                    const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
                    try {
                      const res = await fetch(`${apiUrl}/render?filename=${encodeURIComponent(filename)}&dataset_id=${datasetId}`);
                      if (res.ok) {
                        renderedFiles.current.add(filename);
                        // show video in acceptsFile sections
                        const { path } = await res.json();
                        setSections((prev) =>
                          prev.map((s) => {
                            if ((!s.acceptsFile && !s.inBetweenAcceptsFile) || !activeSections.includes(s.id)) return s;
                            return {
                              ...s,
                              messages: trimMessages([...s.messages, {
                                id: crypto.randomUUID(),
                                align: "right" as const,
                                type: "video" as const,
                                name: filename,
                                url: `${apiUrl}/video?path=${encodeURIComponent(path)}`,
                              }]),
                            };
                          })
                        );
                      }
                    } catch {}
                  }}
                  onClose={() => setBroadcastPickerOpen(false)}
                />
              </div>
            )}

            {broadcastTaskPicker && (
              <div className="absolute top-[calc(100%+8px)] left-0 right-0 z-[100] mx-3 rounded-2xl border border-slate-200 bg-white/95 backdrop-blur shadow-xl p-3">
                <p className="text-xs font-semibold text-slate-600 mb-2">Select task for motion models</p>
                <div className="flex gap-2">
                  <button onClick={() => broadcastMode === "broadcast" ? submitBroadcast("t2m") : submitInBetween("t2m")} className="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300 transition-all">
                    🏃 T2M
                    <span className="block text-xs text-slate-400 font-normal">Text → Motion</span>
                  </button>
                  <button onClick={() => broadcastMode === "broadcast" ? submitBroadcast("m2t") : submitInBetween("m2t")} className="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300 transition-all">
                    📝 M2T
                    <span className="block text-xs text-slate-400 font-normal">Motion → Text</span>
                  </button>
                </div>
              </div>
            )}

            {broadcastPendingFile && (
              <div className="flex items-center gap-2 rounded-2xl border border-blue-200 bg-blue-50/20 px-3 py-2 text-xs text-blue-300 font-medium">
                <span>📄 {broadcastPendingFile}</span>
                <button onClick={() => setBroadcastPendingFile(null)} className="text-blue-400 hover:text-white leading-none">✕</button>
              </div>
            )}

            <input
              value={broadcastText}
              onChange={(e) => setBroadcastText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleBroadcast(); }}
              placeholder="Broadcast to all sections..."
              className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 shadow-inner outline-none placeholder:text-slate-400"
            />

            <button
              onClick={() => broadcastReady && setBroadcastPickerOpen((v) => !v)}
              className={`... ${!broadcastReady ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              📎
            </button>

            <button
              onClick={broadcastReady ? handleBroadcast : undefined}
              disabled={!broadcastReady}
              className={`rounded-2xl px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-950/30 transition ${
                broadcastReady
                  ? "bg-blue-600 hover:bg-blue-500 cursor-pointer"
                  : "bg-slate-600 cursor-not-allowed opacity-40"
              }`}
            >
              Broadcast
            </button>

            <button
              onClick={handleInBetween}
              disabled={!broadcastReady}
              className={`rounded-2xl px-5 py-3 text-sm font-semibold text-white shadow-lg transition ${
                broadcastReady
                  ? "bg-purple-600 hover:bg-purple-500"
                  : "bg-slate-600 cursor-not-allowed opacity-40"
              }`}
            >
              In-Between
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-5 py-6">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {sections.slice(0, 6).map((section) => (
            <ScriptCard
              key={section.id}
              section={section}
              onDraftChange={updateDraft}
              onSend={sendSectionMessage}
              onIncoming={addIncomingMessage}
              onFilenameSelected={addFilenameMessage}
              userid={userid}
              active={activeSections.includes(section.id)}
              onRegisterSend={onRegisterSend}
              onReady={(sectionId) => setReadySections((prev) => new Set([...prev, sectionId]))}
            />
          ))}
          {sections[6] && (
            <ScriptCard
              section={sections[6]}
              tall
              onDraftChange={updateDraft}
              onSend={sendSectionMessage}
              onIncoming={addIncomingMessage}
              onFilenameSelected={addFilenameMessage}
              userid={userid}
              active={activeSections.includes(sections[6].id)}
              onRegisterSend={onRegisterSend}
              onReady={(sectionId) => setReadySections((prev) => new Set([...prev, sectionId]))}
            />
          )}
        </div>
      </main>

      {showFeedback && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-3xl shadow-2xl p-6 w-full max-w-md mx-4 flex flex-col gap-4">
            <h2 className="text-lg font-semibold text-slate-800">How was your experience?</h2>
            
            {/* Rating thumbs */}
            <div className="flex gap-3 justify-center">
              {(["good", "ok", "bad"] as const).map((r) => (
                <button
                  key={r}
                  onClick={() => setFeedbackRating(r)}
                  className={`flex-1 py-3 rounded-2xl border text-sm font-medium transition-all ${
                    feedbackRating === r
                      ? "border-blue-400 bg-blue-50 text-blue-700"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  {r === "good" ? "👍 Good" : r === "ok" ? "👌 Ok" : "👎 Bad"}
                </button>
              ))}
            </div>

            {/* Comment box — only for bad or ok */}
            {(feedbackRating === "bad" || feedbackRating === "ok") && (
              <textarea
                value={feedbackComment}
                onChange={(e) => setFeedbackComment(e.target.value)}
                placeholder="Tell us what could be improved..."
                className="rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-700 outline-none resize-none h-24 shadow-inner"
              />
            )}

            <div className="flex gap-2">
              <button
                onClick={() => setShowFeedback(false)}
                className="flex-1 rounded-2xl border border-slate-200 py-2 text-sm text-slate-500 hover:bg-slate-50"
              >
                Skip
              </button>
              <button
                onClick={submitFeedback}
                disabled={!feedbackRating}
                className={`flex-1 rounded-2xl py-2 text-sm font-semibold text-white transition-all ${
                  feedbackRating ? "bg-blue-600 hover:bg-blue-700" : "bg-slate-300 cursor-not-allowed"
                }`}
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}