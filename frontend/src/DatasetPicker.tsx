import * as React from "react";

// ── Types ────────────────────────────────────────────────────────────────────

export type CmuRow = {
  outside: number;
  origname: string;
  sequence_labels: string;
  frame_labels: string;
};

export type SimpleRow = {
  origname: string;
};

export type Datasets = {
  cmu: CmuRow[];
  jump_handsup: SimpleRow[];
  jump_vertical: SimpleRow[];
  run: SimpleRow[];
  sit: SimpleRow[];
  walk: SimpleRow[];
  upperstatic: CmuRow[];
};

type PickerView =
  | { kind: "datasets" }
  | { kind: "cmu_outsides"; dataset: string }
  | { kind: "cmu_rows"; outside: number; dataset: string }
  | { kind: "simple_list"; dataset: keyof Omit<Datasets, "cmu" | "upperstatic"> };

// ── Hook: fetch datasets once globally ───────────────────────────────────────

let _cache: Datasets | null = null;
let _promise: Promise<Datasets> | null = null;

export function useDatasets() {
  const [datasets, setDatasets] = React.useState<Datasets | null>(_cache);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (_cache) { setDatasets(_cache); return; }
    if (!_promise) {
      const apiUrl = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
      _promise = fetch(`${apiUrl}/datasets`).then((r) => r.json());
    }
    _promise
      .then((data) => { _cache = data; setDatasets(data); })
      .catch(() => setError("Could not load datasets"));
  }, []);

  return { datasets, error };
}

// ── Sub-components ────────────────────────────────────────────────────────────

const DATASET_LABELS: Record<string, string> = {
  cmu:           "CMU",
  jump_handsup:  "Jump\nHands Up",
  jump_vertical: "Jump\nVertical",
  run:           "Run",
  sit:           "Sit",
  walk:          "Walk",
  upperstatic:   "Upper\nStatic",
};

const DATASET_COLORS: Record<string, string> = {
  cmu:           "from-slate-100 to-slate-200 border-slate-300 text-slate-700",
  jump_handsup:  "from-violet-50 to-violet-100 border-violet-300 text-violet-700",
  jump_vertical: "from-emerald-50 to-emerald-100 border-emerald-300 text-emerald-700",
  run:           "from-amber-50 to-amber-100 border-amber-300 text-amber-700",
  sit:           "from-rose-50 to-rose-100 border-rose-300 text-rose-700",
  walk:          "from-sky-50 to-sky-100 border-sky-300 text-sky-700",
  upperstatic: "from-orange-50 to-orange-100 border-orange-300 text-orange-700"
};

function BackButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800 transition-colors mb-2"
    >
      ← Back
    </button>
  );
}

// 6-box grid for dataset selection
function DatasetGrid({
  datasets,
  onSelect,
}: {
  datasets: Datasets;
  onSelect: (key: string) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {Object.keys(DATASET_LABELS).map((key) => (
        <button
          key={key}
          onClick={() => onSelect(key)}
          className={`bg-gradient-to-br ${DATASET_COLORS[key]} border rounded-xl py-3 px-2 text-center text-xs font-semibold whitespace-pre-line leading-tight transition-all hover:scale-[1.03] hover:shadow-md active:scale-[0.97]`}
        >
          {DATASET_LABELS[key]}
        </button>
      ))}
    </div>
  );
}

// CMU: list of unique outside numbers
function CmuOutsideList({
  rows,
  onSelect,
}: {
  rows: CmuRow[];
  onSelect: (outside: number) => void;
}) {
  const outsides = [...new Set(rows.map((r) => r.outside))].sort((a, b) => a - b);
  return (
    <div className="flex flex-wrap gap-2">
      {outsides.map((o) => (
        <button
          key={o}
          onClick={() => onSelect(o)}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-mono font-semibold text-slate-700 shadow-sm hover:bg-slate-50 hover:shadow transition-all"
        >
          {o}
        </button>
      ))}
    </div>
  );
}

// CMU: two-column table of sequence_labels + frame_labels
function CmuRowTable({
  rows,
  onSelect,
}: {
  rows: CmuRow[];
  onSelect: (origname: string, datasetKey: string) => void;
}) {
  return (
    <div className="overflow-y-auto max-h-[180px] rounded-xl border border-slate-200 bg-white text-xs">
      <table className="w-full border-collapse">
        <thead className="sticky top-0 bg-slate-50 z-10">
          <tr>
            <th className="px-3 py-2 text-left font-semibold text-slate-500 border-b border-slate-200">Sequence</th>
            <th className="px-3 py-2 text-left font-semibold text-slate-500 border-b border-slate-200">Frame labels</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.origname + i}
              onClick={() => onSelect(row.origname, "cmu")}
              className="cursor-pointer hover:bg-blue-50 transition-colors border-b border-slate-100 last:border-0"
            >
              <td className="px-3 py-2 text-slate-700 font-medium">{row.sequence_labels}</td>
              <td className="px-3 py-2 text-slate-500 italic">{row.frame_labels}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Simple datasets: flat list of filenames
function SimpleFileList({
  rows,
  datasetKey,
  onSelect,
}: {
  rows: SimpleRow[];
  datasetKey: string;
  onSelect: (origname: string, datasetKey: string) => void;
}) {
  return (
    <div className="overflow-y-auto max-h-[120px] flex flex-col gap-1">
      {rows.map((row, i) => (
        <button
          key={row.origname + i}
          onClick={() => onSelect(row.origname, datasetKey)}
          className="text-left rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 font-mono hover:bg-blue-50 hover:border-blue-300 transition-all"
        >
          📄 {row.origname}
        </button>
      ))}
    </div>
  );
}

// ── Main DatasetPicker component ──────────────────────────────────────────────

export function DatasetPicker({
  onFilePicked,
  onClose,
}: {
  onFilePicked: (filename: string, datasetId: number) => void;
  onClose: () => void;
}) {
  const { datasets, error } = useDatasets();
  const [view, setView] = React.useState<PickerView>({ kind: "datasets" });

  function handleDatasetSelect(key: string) {
    if (key === "cmu" || key === "upperstatic") {
      setView({ kind: "cmu_outsides", dataset: key });
    } else {
      setView({ kind: "simple_list", dataset: key as keyof Omit<Datasets, "cmu" | "upperstatic"> });
    }
  }

  // Map dataset keys to IDs
const DATASET_IDS: Record<string, number> = {
  cmu:           0,
  jump_handsup:  0,
  jump_vertical: 0,
  run:           0,
  sit:           0,
  walk:          0,
  upperstatic:   1,
};

function handleFilePick(filename: string, datasetKey: string) {
  onFilePicked(filename, DATASET_IDS[datasetKey]);
  onClose();
}

  const title =
    view.kind === "datasets"      ? "Select dataset" :
    view.kind === "cmu_outsides"  ? "CMU — select subject" :
    view.kind === "cmu_rows"      ? `CMU — subject ${view.outside}` :
    view.kind === "simple_list"   ? DATASET_LABELS[view.dataset] :
    "";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/95 backdrop-blur shadow-xl p-3 flex flex-col gap-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-600">{title}</span>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-700 text-sm leading-none px-1"
        >
          ✕
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{error}</div>
      )}

      {/* Loading state */}
      {!datasets && !error && (
        <div className="text-xs text-slate-400 text-center py-4">Loading datasets…</div>
      )}

      {/* Views */}
      {datasets && (
        <>
          {view.kind === "datasets" && (
            <DatasetGrid datasets={datasets} onSelect={handleDatasetSelect} />
          )}

          {view.kind === "cmu_outsides" && (
            <>
              <BackButton onClick={() => setView({ kind: "datasets" })} />
              <CmuOutsideList
                rows={datasets[view.dataset as "cmu" | "upperstatic"]}
                onSelect={(outside) => setView({ kind: "cmu_rows", outside, dataset: view.dataset })}
              />
            </>
          )}

          {view.kind === "cmu_rows" && (
            <>
              <BackButton onClick={() => setView({ kind: "cmu_outsides", dataset: view.dataset })} />
              <CmuRowTable
                rows={datasets[view.dataset as "cmu" | "upperstatic"].filter((r) => r.outside === view.outside)}
                onSelect={(origname) => handleFilePick(origname, view.dataset)}
              />
            </>
          )}

          {view.kind === "simple_list" && (
            <>
              <BackButton onClick={() => setView({ kind: "datasets" })} />
              <SimpleFileList
                rows={datasets[view.dataset]}
                datasetKey={view.dataset}
                onSelect={(origname, key) => handleFilePick(origname, key)}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}