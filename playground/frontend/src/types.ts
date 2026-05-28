export type Recommendation = "SCALE" | "HOLD" | "CUT" | "INCONCLUSIVE";

export interface ChannelResult {
  channel: string;
  total_spend: number;
  /** Proportional daily attribution proxy — NOT platform-reported ROAS */
  attribution_proxy_roas: number;
  incremental_roas: number;
  incremental_revenue: number;
  confidence_interval: [number, number];
  recommendation: Recommendation;
  recommendation_reason: string;
  model_fit: number;
  /** Variance Inflation Factor — > 10 means estimate is unreliable */
  vif_score: number | null;
  raw_coef: number;
}

export interface AnalysisResult {
  channels: ChannelResult[];
  method_used: string;
  total_revenue: number;
  total_spend: number;
  r_squared: number;
  observations: number;
  contribution_margin: number;
  breakeven_roas: number;
  durbin_watson: number;
  warnings: string[];
  /** Only present when loaded via /sample-data */
  _ground_truth?: Record<string, number>;
}

export interface DataSummary {
  channels: string[];
  date_range: { start: string; end: string };
  days: number;
}

export interface ParsedData {
  spend_data: Array<{ date: string; channel: string; spend: number }>;
  sales_data: Array<{ date: string; revenue: number; orders?: number }>;
  summary: DataSummary;
  /** Only present in sample-data mode */
  _ground_truth?: Record<string, number>;
}

// ── App state machine ────────────────────────────────────────────────────────
// Note: no "ready" intermediate state — goes directly upload → analyzing → results

export type AppStage =
  | { kind: "upload" }
  | { kind: "uploading" }
  | { kind: "analyzing"; data: ParsedData; contributionMargin: number }
  | { kind: "results"; data: ParsedData; result: AnalysisResult }
  | { kind: "error"; message: string };
