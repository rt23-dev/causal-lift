import type { AnalysisResult, ParsedData } from "./types";

// Configurable via VITE_API_BASE_URL env var — set in .env.local for non-localhost deployments
const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg: string;
    try {
      const body = await res.json();
      msg = body.detail ?? JSON.stringify(body);
    } catch {
      msg = await res.text();
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function uploadCSVs(spendFile: File, salesFile: File): Promise<ParsedData> {
  const form = new FormData();
  form.append("spend_file", spendFile);
  form.append("sales_file", salesFile);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  return handleResponse<ParsedData>(res);
}

export async function analyze(
  data: ParsedData,
  contributionMargin: number
): Promise<AnalysisResult> {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spend_data: data.spend_data,
      sales_data: data.sales_data,
      contribution_margin: contributionMargin,
    }),
  });
  return handleResponse<AnalysisResult>(res);
}

export async function loadSampleData(): Promise<ParsedData> {
  const res = await fetch(`${BASE}/sample-data`);
  return handleResponse<ParsedData>(res);
}
