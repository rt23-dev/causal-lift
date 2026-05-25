import { ArrowLeft, Download, TrendingDown, TrendingUp } from "lucide-react";
import type { AnalysisResult, ParsedData } from "../types";
import { downloadCSV, fmt, fmtCurrency, fmtPct } from "../utils";
import ChannelCard from "./ChannelCard";
import RoasChart from "./RoasChart";
import SpendRevenueChart from "./SpendRevenueChart";

interface Props {
  result: AnalysisResult;
  data: ParsedData;
  onReset: () => void;
}

function exportResults(result: AnalysisResult) {
  const rows = result.channels.map((ch) => ({
    channel: ch.channel,
    total_spend: ch.total_spend,
    attribution_proxy_roas: ch.attribution_proxy_roas,
    incremental_roas: ch.incremental_roas,
    ci_lower: ch.confidence_interval[0],
    ci_upper: ch.confidence_interval[1],
    incremental_revenue: ch.incremental_revenue,
    recommendation: ch.recommendation,
    vif_score: ch.vif_score ?? "",
    breakeven_roas: result.breakeven_roas,
    contribution_margin_pct: (result.contribution_margin * 100).toFixed(0),
    model_r2: result.r_squared,
    observations: result.observations,
    durbin_watson: result.durbin_watson,
  }));
  downloadCSV(rows, "incrementaliq_results.csv");
}

export default function Dashboard({ result, data, onReset }: Props) {
  const groundTruth = result._ground_truth ?? data._ground_truth;
  const isSampleData = !!groundTruth;

  // Weighted blended ROAS — guard against zero spend
  const totalSpend = result.total_spend > 0 ? result.total_spend : 1;
  const blendedProxy =
    result.channels.reduce((s, c) => s + c.attribution_proxy_roas * c.total_spend, 0) / totalSpend;
  const blendedIncremental =
    result.channels.reduce((s, c) => s + c.incremental_roas * c.total_spend, 0) / totalSpend;
  const totalIncrementalRevenue = result.channels.reduce((s, c) => s + c.incremental_revenue, 0);

  // Only show the "you're being overcharged" banner when model is credible
  const modelIsCredible = result.r_squared >= 0.4 && result.observations >= 30;
  const hasHighVif = result.channels.some((c) => c.vif_score !== null && (c.vif_score ?? 0) > 10);

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-100 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onReset}
              className="text-slate-400 hover:text-slate-600 transition-colors"
              aria-label="Back to upload"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="font-extrabold text-slate-900 tracking-tight">
                Incremental<span className="text-indigo-600">IQ</span>
              </h1>
              <p className="text-xs text-slate-400">
                {data.summary.date_range.start} → {data.summary.date_range.end} ·{" "}
                {result.observations} days · {fmtPct(result.contribution_margin)} margin ·{" "}
                {fmt(result.breakeven_roas)}x breakeven
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isSampleData && (
              <span className="text-xs font-semibold bg-violet-100 text-violet-600 px-3 py-1 rounded-full">
                Sample Data
              </span>
            )}
            <button
              onClick={() => exportResults(result)}
              className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 border border-slate-200 hover:border-slate-300 px-3 py-1.5 rounded-lg transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">

        {/* ── Warnings ─────────────────────────────────────────────────────── */}
        {result.warnings.length > 0 && (
          <div className="space-y-2">
            {result.warnings.map((w, i) => {
              const isCritical = w.startsWith("⚠️");
              return (
                <div
                  key={i}
                  className={`flex items-start gap-3 rounded-xl px-4 py-3 text-sm border ${
                    isCritical
                      ? "bg-amber-50 border-amber-200 text-amber-800"
                      : "bg-slate-50 border-slate-200 text-slate-600"
                  }`}
                >
                  <span className="flex-shrink-0 mt-0.5">{isCritical ? "⚠️" : "ℹ️"}</span>
                  <span className="leading-relaxed">{w.replace(/^⚠️\s*/, "")}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* ── Hero metrics ─────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            {
              label: "Total Ad Spend",
              value: fmtCurrency(result.total_spend),
              sub: `across ${result.channels.length} channels`,
              icon: null,
              color: "text-slate-800",
            },
            {
              label: "Total Revenue",
              value: fmtCurrency(result.total_revenue),
              sub: `${result.observations} day window`,
              icon: null,
              color: "text-slate-800",
            },
            {
              label: "Attribution Proxy",
              value: `${fmt(blendedProxy)}x`,
              sub: "naive proportional",
              icon: <TrendingUp className="w-4 h-4 text-slate-400" />,
              color: "text-slate-500",
            },
            {
              label: "Incremental ROAS",
              value: `${fmt(blendedIncremental)}x`,
              sub: `breakeven = ${fmt(result.breakeven_roas)}x`,
              icon: <TrendingDown className="w-4 h-4 text-indigo-400" />,
              color: "text-indigo-700",
            },
          ].map(({ label, value, sub, icon, color }) => (
            <div
              key={label}
              className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5"
            >
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  {label}
                </p>
                {icon}
              </div>
              <p className={`text-2xl font-bold ${color}`}>{value}</p>
              <p className="text-xs text-slate-400 mt-1">{sub}</p>
            </div>
          ))}
        </div>

        {/* ── Call-out banner — only when model is credible ────────────────── */}
        {modelIsCredible && !hasHighVif ? (
          <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-2xl p-6 text-white">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div>
                <p className="font-bold text-lg">
                  Only {fmtPct(totalIncrementalRevenue / result.total_revenue)} of revenue appears
                  incremental.
                </p>
                <p className="text-indigo-200 text-sm mt-1">
                  Causal estimate: <strong className="text-white">{fmt(blendedIncremental)}x</strong>{" "}
                  vs naive proxy: <strong className="text-white">{fmt(blendedProxy)}x</strong>.{" "}
                  {blendedIncremental < result.breakeven_roas ? (
                    <span className="text-yellow-300 font-semibold">
                      Blended iROAS is below your {fmt(result.breakeven_roas)}x breakeven — paid
                      media may be net-negative at current margins.
                    </span>
                  ) : (
                    <span>
                      Blended iROAS is above breakeven — but channel mix matters. See cards below.
                    </span>
                  )}
                </p>
                <p className="text-indigo-300 text-xs mt-2">
                  ⚠️ Assumes spend variation is budget-driven. Algorithmic bidding may bias these
                  estimates upward. Treat as directional.
                </p>
              </div>
              <div className="flex-shrink-0 text-center bg-white/10 rounded-xl px-6 py-3">
                <p className="text-xs text-indigo-200">Incremental Revenue</p>
                <p className="text-3xl font-extrabold">{fmtCurrency(totalIncrementalRevenue)}</p>
                <p className="text-xs text-indigo-200">causally attributed</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5">
            <p className="font-semibold text-amber-800">
              {hasHighVif
                ? "High collinearity detected — channel-level estimates are unreliable."
                : `Model R²=${fmt(result.r_squared)} is low — interpret estimates with caution.`}
            </p>
            <p className="text-sm text-amber-700 mt-1">
              {hasHighVif
                ? "Channels with correlated spend cannot be individually identified by regression alone. Run budget holdout experiments to get credible per-channel estimates."
                : "Revenue may be driven by factors outside the uploaded spend data. More data, controlled experiments, or additional covariates (promo flags, seasonality) would improve model quality."}
            </p>
          </div>
        )}

        {/* ── Time-series chart ─────────────────────────────────────────────── */}
        <SpendRevenueChart
          spendData={data.spend_data}
          salesData={data.sales_data}
          channels={data.summary.channels}
        />

        {/* ── ROAS comparison chart ─────────────────────────────────────────── */}
        <RoasChart channels={result.channels} breakeven_roas={result.breakeven_roas} />

        {/* ── Channel cards ─────────────────────────────────────────────────── */}
        <div>
          <h2 className="font-bold text-slate-900 mb-4">Channel Breakdown</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {result.channels.map((ch) => (
              <ChannelCard
                key={ch.channel}
                ch={ch}
                breakeven_roas={result.breakeven_roas}
                groundTruth={groundTruth?.[ch.channel]}
              />
            ))}
          </div>
        </div>

        {/* ── Model footer ──────────────────────────────────────────────────── */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5 text-sm">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
            <div>
              <p className="font-semibold text-slate-700">Analysis method</p>
              <p className="text-slate-400 text-xs mt-0.5">{result.method_used}</p>
            </div>
            <div className="flex flex-wrap gap-6 text-center">
              {[
                ["Model R²", fmt(result.r_squared)],
                ["Observations", String(result.observations)],
                ["Channels", String(result.channels.length)],
                ["Durbin-Watson", fmt(result.durbin_watson)],
                ["HAC lags", String(Math.max(1, Math.floor(4 * Math.pow(result.observations / 100, 2 / 9))))],
              ].map(([label, value]) => (
                <div key={label}>
                  <p className="text-xs text-slate-400">{label}</p>
                  <p className="font-bold text-slate-700">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {isSampleData && (
            <p className="mt-3 pt-3 border-t border-slate-100 text-xs text-violet-500">
              Ground truth values shown in purple — true iROAS baked into the synthetic DGP.
              Each channel has a deliberate budget experiment to enable causal identification.
              Without budget experiments, per-channel estimates are not reliable (see VIF scores).
            </p>
          )}

          <p className="mt-3 pt-3 border-t border-slate-100 text-xs text-slate-400">
            <strong>Limitations:</strong> This model assumes spend variation is exogenous
            (budget-driven). Algorithmic bidding violates this and will bias iROAS upward.
            No adstock or carryover effects are modelled — brand-awareness channels may appear
            weaker than they are. Confidence intervals use HAC (Newey-West) SEs to correct for
            autocorrelation. VIF scores flag multicollinearity; channels with VIF&nbsp;&#62;&nbsp;10
            should not be acted on without an experiment.
          </p>
        </div>
      </main>
    </div>
  );
}
