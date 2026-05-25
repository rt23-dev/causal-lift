import { AlertCircle, FileText, Sparkles, Upload as UploadIcon } from "lucide-react";
import React, { useState } from "react";

interface Props {
  onUpload: (spendFile: File, salesFile: File, contributionMargin: number) => void;
  onUseSample: (contributionMargin: number) => void;
  loading: boolean;
  error: string | null;
}

interface FileSlot {
  file: File | null;
  dragging: boolean;
}

// ── FileZone hoisted to module level to prevent remount on parent re-render ──

interface FileZoneProps {
  label: string;
  hint: string;
  slot: FileSlot;
  inputId: string;
  onFile: (f: File) => void;
  onDragChange: (dragging: boolean) => void;
}

function FileZone({ label, hint, slot, inputId, onFile, onDragChange }: FileZoneProps) {
  return (
    <label
      htmlFor={inputId}
      className={`relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 cursor-pointer transition-colors
        ${
          slot.dragging
            ? "border-indigo-400 bg-indigo-50"
            : slot.file
            ? "border-emerald-400 bg-emerald-50"
            : "border-slate-200 bg-white hover:border-indigo-300 hover:bg-slate-50"
        }`}
      onDragOver={(e) => {
        e.preventDefault();
        onDragChange(true);
      }}
      onDragLeave={() => onDragChange(false)}
      onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
        onDragChange(false);
      }}
    >
      <input
        id={inputId}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {slot.file ? (
        <>
          <FileText className="w-8 h-8 text-emerald-500" />
          <div className="text-center">
            <p className="font-semibold text-slate-800 text-sm">{slot.file.name}</p>
            <p className="text-xs text-slate-500 mt-0.5">
              {(slot.file.size / 1024).toFixed(1)} KB · click to replace
            </p>
          </div>
        </>
      ) : (
        <>
          <UploadIcon className="w-8 h-8 text-slate-400" />
          <div className="text-center">
            <p className="font-semibold text-slate-700 text-sm">{label}</p>
            <p className="text-xs text-slate-500 mt-1">{hint}</p>
            <p className="text-xs text-slate-400 mt-2">drag & drop or click to browse</p>
          </div>
        </>
      )}
    </label>
  );
}

// ── Upload page ───────────────────────────────────────────────────────────────

export default function Upload({ onUpload, onUseSample, loading, error }: Props) {
  const [spend, setSpend] = useState<FileSlot>({ file: null, dragging: false });
  const [sales, setSales] = useState<FileSlot>({ file: null, dragging: false });
  const [marginPct, setMarginPct] = useState<string>("30");

  const parsedMargin = Math.max(5, Math.min(95, parseFloat(marginPct) || 30)) / 100;
  const breakeven = (1 / parsedMargin).toFixed(1);
  const canSubmit = spend.file && sales.file && !loading;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (spend.file && sales.file) onUpload(spend.file, sales.file, parsedMargin);
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      {/* Header */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 bg-indigo-100 text-indigo-700 text-xs font-semibold px-3 py-1 rounded-full mb-4">
          <Sparkles className="w-3 h-3" />
          Causal incrementality · not vanity ROAS
        </div>
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">
          Incremental<span className="text-indigo-600">IQ</span>
        </h1>
        <p className="mt-3 text-slate-500 text-lg max-w-lg mx-auto">
          Upload your ad spend and sales data. We'll tell you which channels are actually
          driving revenue — and which ones are just taking credit.
        </p>
      </div>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-2xl bg-white rounded-3xl shadow-xl shadow-slate-200 p-8 space-y-6"
      >
        {/* File zones */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Ad Spend
            </p>
            <FileZone
              label="spend.csv"
              hint="date · channel · spend"
              slot={spend}
              inputId="spend-input"
              onFile={(f) => setSpend({ file: f, dragging: false })}
              onDragChange={(d) => setSpend((s) => ({ ...s, dragging: d }))}
            />
          </div>
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Sales
            </p>
            <FileZone
              label="sales.csv"
              hint="date · revenue · orders (optional)"
              slot={sales}
              inputId="sales-input"
              onFile={(f) => setSales({ file: f, dragging: false })}
              onDragChange={(d) => setSales((s) => ({ ...s, dragging: d }))}
            />
          </div>
        </div>

        {/* Contribution margin */}
        <div className="bg-slate-50 rounded-2xl p-4 space-y-3">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-slate-700">Contribution Margin</p>
              <p className="text-xs text-slate-400 mt-0.5">
                Revenue minus COGS, fulfilment & returns. Used to calculate your breakeven iROAS.
              </p>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <input
                type="number"
                min={5}
                max={95}
                step={1}
                value={marginPct}
                onChange={(e) => setMarginPct(e.target.value)}
                className="w-16 text-center border border-slate-200 rounded-lg py-1.5 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-300"
              />
              <span className="text-sm text-slate-500">%</span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-indigo-600 bg-indigo-50 rounded-lg px-3 py-2">
            <span>📍</span>
            <span>
              At {marginPct}% margin your breakeven iROAS is{" "}
              <strong>{breakeven}x</strong> — channels below this are destroying value.
            </span>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 rounded-xl bg-red-50 border border-red-200 p-4 text-sm text-red-700">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="submit"
            disabled={!canSubmit}
            className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-xl transition-colors"
          >
            {loading ? "Uploading…" : "Analyze My Data"}
          </button>
          <button
            type="button"
            onClick={() => onUseSample(parsedMargin)}
            disabled={loading}
            className="flex-1 sm:flex-none border border-slate-200 hover:bg-slate-50 disabled:opacity-50 text-slate-700 font-semibold py-3 px-6 rounded-xl transition-colors text-sm"
          >
            Use Sample Data
          </button>
        </div>

        <p className="text-center text-xs text-slate-400">
          Data is processed in memory and never stored on our servers.
        </p>
      </form>

      {/* Format reference */}
      <div className="mt-8 w-full max-w-2xl grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
        {[
          {
            title: "spend.csv",
            cols: ["date", "channel", "spend"],
            sample: [
              ["2024-01-01", "facebook", "3200"],
              ["2024-01-01", "google", "1800"],
              ["2024-01-02", "facebook", "3400"],
            ],
          },
          {
            title: "sales.csv",
            cols: ["date", "revenue", "orders"],
            sample: [
              ["2024-01-01", "42000", "312"],
              ["2024-01-02", "38500", "287"],
              ["2024-01-03", "44100", "328"],
            ],
          },
        ].map(({ title, cols, sample }) => (
          <div key={title} className="bg-white rounded-xl border border-slate-100 p-4">
            <p className="font-semibold text-slate-700 mb-2">{title}</p>
            <table className="w-full font-mono">
              <thead>
                <tr>
                  {cols.map((c) => (
                    <th key={c} className="text-left text-slate-400 font-normal pb-1">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sample.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j} className="text-slate-600 pr-3">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
