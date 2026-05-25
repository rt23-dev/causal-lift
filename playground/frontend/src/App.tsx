import { useState } from "react";
import { analyze, loadSampleData, uploadCSVs } from "./api";
import Dashboard from "./components/Dashboard";
import Upload from "./components/Upload";
import type { AnalysisResult, AppStage, ParsedData } from "./types";

export default function App() {
  const [stage, setStage] = useState<AppStage>({ kind: "upload" });

  async function handleUpload(
    spendFile: File,
    salesFile: File,
    contributionMargin: number
  ) {
    setStage({ kind: "uploading" });
    try {
      const data = await uploadCSVs(spendFile, salesFile);
      await runAnalysis(data, contributionMargin);
    } catch (e) {
      setStage({ kind: "error", message: (e as Error).message });
    }
  }

  async function handleSampleData(contributionMargin: number) {
    setStage({ kind: "uploading" });
    try {
      const data = await loadSampleData();
      await runAnalysis(data, contributionMargin);
    } catch (e) {
      setStage({ kind: "error", message: (e as Error).message });
    }
  }

  async function runAnalysis(data: ParsedData, contributionMargin: number) {
    setStage({ kind: "analyzing", data, contributionMargin });
    try {
      const result: AnalysisResult = await analyze(data, contributionMargin);
      // Pass ground truth through for sample-data verification
      if (data._ground_truth) result._ground_truth = data._ground_truth;
      setStage({ kind: "results", data, result });
    } catch (e) {
      setStage({ kind: "error", message: (e as Error).message });
    }
  }

  function reset() {
    setStage({ kind: "upload" });
  }

  if (stage.kind === "uploading") {
    return <Spinner label="Parsing your CSV files…" sub="Validating columns and types." />;
  }

  if (stage.kind === "analyzing") {
    return (
      <Spinner
        label="Running causal analysis…"
        sub={`Fitting Diff-in-Diff model on ${stage.data.summary.days} days × ${stage.data.summary.channels.length} channels.`}
      />
    );
  }

  if (stage.kind === "results") {
    return <Dashboard result={stage.result} data={stage.data} onReset={reset} />;
  }

  const error = stage.kind === "error" ? stage.message : null;

  return (
    <Upload
      onUpload={handleUpload}
      onUseSample={handleSampleData}
      loading={false}
      error={error}
    />
  );
}

function Spinner({ label, sub }: { label: string; sub: string }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-4">
      <div className="w-10 h-10 rounded-full border-4 border-indigo-200 border-t-indigo-600 animate-spin" />
      <p className="font-semibold text-slate-700">{label}</p>
      <p className="text-sm text-slate-400">{sub}</p>
    </div>
  );
}
