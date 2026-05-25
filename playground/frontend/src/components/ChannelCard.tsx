import { AlertTriangle } from "lucide-react";
import { fmt, fmtCurrency } from "../utils";
import type { ChannelResult } from "../types";

const REC_STYLES = {
  SCALE: {
    badge: "bg-emerald-100 text-emerald-700 border border-emerald-200",
    accent: "bg-emerald-500",
    border: "border-emerald-200",
  },
  HOLD: {
    badge: "bg-amber-100 text-amber-700 border border-amber-200",
    accent: "bg-amber-400",
    border: "border-amber-200",
  },
  CUT: {
    badge: "bg-red-100 text-red-700 border border-red-200",
    accent: "bg-red-500",
    border: "border-red-200",
  },
};

const CHANNEL_COLORS: Record<string, string> = {
  facebook: "bg-blue-500",
  meta: "bg-blue-500",
  google: "bg-yellow-500",
  tiktok: "bg-pink-500",
  snapchat: "bg-yellow-400",
  twitter: "bg-sky-400",
  pinterest: "bg-red-500",
  youtube: "bg-red-600",
  linkedin: "bg-blue-700",
  display: "bg-teal-500",
  email: "bg-violet-500",
  affiliates: "bg-orange-500",
};

interface Props {
  ch: ChannelResult;
  breakeven_roas: number;
  groundTruth?: number;
}

export default function ChannelCard({ ch, breakeven_roas, groundTruth }: Props) {
  const style = REC_STYLES[ch.recommendation];
  const channelColor =
    CHANNEL_COLORS[ch.channel.toLowerCase()] ?? "bg-slate-500";

  const highVif = ch.vif_score !== null && ch.vif_score !== undefined && ch.vif_score > 10;
  const modVif = ch.vif_score !== null && ch.vif_score !== undefined && ch.vif_score > 5 && ch.vif_score <= 10;

  // CI bar: map to a visual scale of 0..max
  const scale = Math.max(8, breakeven_roas * 2.5);
  const pctLow = Math.max(0, (ch.confidence_interval[0] / scale) * 100);
  const pctHigh = Math.min(100, (ch.confidence_interval[1] / scale) * 100);
  const pctPoint = Math.min(100, Math.max(0, (ch.incremental_roas / scale) * 100));
  const pctBreak = Math.min(100, (breakeven_roas / scale) * 100);

  const proxyOverclaim =
    ch.attribution_proxy_roas > 0
      ? (((ch.attribution_proxy_roas - ch.incremental_roas) / ch.attribution_proxy_roas) * 100).toFixed(0)
      : null;

  return (
    <div className={`bg-white rounded-2xl border ${style.border} shadow-sm overflow-hidden`}>
      <div className={`h-1 ${style.accent}`} />

      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className={`w-9 h-9 rounded-xl ${channelColor} flex items-center justify-center text-white font-bold text-sm uppercase`}
            >
              {ch.channel[0]}
            </div>
            <div>
              <p className="font-bold text-slate-900 capitalize">{ch.channel}</p>
              <p className="text-xs text-slate-400">{fmtCurrency(ch.total_spend)} spent</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {(highVif || modVif) && (
              <span
                title={`VIF=${ch.vif_score?.toFixed(1)} — ${highVif ? "high collinearity, estimate unreliable" : "moderate collinearity, treat with caution"}`}
                className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                  highVif ? "bg-red-100 text-red-600" : "bg-amber-100 text-amber-600"
                }`}
              >
                <AlertTriangle className="w-3 h-3" />
                VIF {ch.vif_score?.toFixed(0)}
              </span>
            )}
            <span className={`text-xs font-bold px-3 py-1 rounded-full ${style.badge}`}>
              {ch.recommendation}
            </span>
          </div>
        </div>

        {/* ROAS comparison */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="rounded-xl bg-slate-50 p-3">
            <p className="text-xs text-slate-400 mb-1">Attribution Proxy</p>
            <p className="text-2xl font-bold text-slate-500">{fmt(ch.attribution_proxy_roas)}x</p>
            <p className="text-xs text-slate-400 mt-0.5">naive proportional</p>
          </div>
          <div className="rounded-xl bg-indigo-50 p-3">
            <p className="text-xs text-indigo-500 mb-1">Incremental ROAS</p>
            <p className="text-2xl font-bold text-indigo-700">{fmt(ch.incremental_roas)}x</p>
            <p className="text-xs text-indigo-400 mt-0.5">causal estimate</p>
          </div>
        </div>

        {/* Breakeven callout */}
        <div className="flex items-center justify-between text-xs mb-4 bg-slate-50 rounded-lg px-3 py-2">
          <span className="text-slate-500">
            Breakeven at your margin:{" "}
            <strong className="text-slate-700">{fmt(breakeven_roas)}x</strong>
          </span>
          {ch.incremental_roas >= breakeven_roas ? (
            <span className="text-emerald-600 font-semibold">above ✓</span>
          ) : (
            <span className="text-red-500 font-semibold">below ✗</span>
          )}
        </div>

        {proxyOverclaim !== null && Number(proxyOverclaim) > 10 && (
          <p className="text-xs text-slate-500 mb-4">
            Proxy overclaims by{" "}
            <span className="font-semibold text-red-500">{proxyOverclaim}%</span> vs causal estimate.{" "}
            <span className="text-slate-400">
              (Compare to your actual platform dashboard — these will differ.)
            </span>
          </p>
        )}

        {/* CI visualisation */}
        <div className="mb-4">
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>95% Confidence Interval (HAC)</span>
            <span>
              [{fmt(ch.confidence_interval[0])}x, {fmt(ch.confidence_interval[1])}x]
            </span>
          </div>
          <div className="relative h-3 bg-slate-100 rounded-full">
            {/* CI range */}
            <div
              className="absolute top-0 h-full bg-indigo-200 rounded-full"
              style={{ left: `${pctLow}%`, width: `${Math.max(0, pctHigh - pctLow)}%` }}
            />
            {/* Breakeven line */}
            <div
              className="absolute top-[-3px] h-[calc(100%+6px)] w-0.5 bg-red-400"
              style={{ left: `${pctBreak}%` }}
              title={`Breakeven ${fmt(breakeven_roas)}x`}
            />
            {/* Point estimate */}
            <div
              className="absolute top-[-2px] w-3 h-3 rounded-full bg-indigo-600 border-2 border-white shadow"
              style={{ left: `calc(${pctPoint}% - 6px)` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-300 mt-1">
            <span>0x</span>
            <span className="text-red-400">{fmt(breakeven_roas)}x breakeven</span>
            <span>{fmt(scale)}x</span>
          </div>
        </div>

        {/* Incremental revenue */}
        <div className="flex items-center justify-between text-sm border-t border-slate-100 pt-4">
          <span className="text-slate-500">Incremental Revenue</span>
          <span className="font-semibold text-slate-800">
            {fmtCurrency(ch.incremental_revenue)}
          </span>
        </div>

        {/* VIF warning detail */}
        {highVif && (
          <div className="mt-3 text-xs bg-red-50 border border-red-100 rounded-lg px-3 py-2 text-red-600">
            ⚠️ High collinearity (VIF={ch.vif_score?.toFixed(1)}) — this channel's spend moves
            with others, making individual attribution unreliable. Run a budget holdout to isolate
            its effect.
          </div>
        )}

        {/* Ground truth (sample data only) */}
        {groundTruth !== undefined && (
          <div className="mt-3 flex items-center justify-between text-xs bg-violet-50 rounded-lg px-3 py-2">
            <span className="text-violet-500">True iROAS (DGP ground truth)</span>
            <span className="font-bold text-violet-700">{groundTruth.toFixed(1)}x</span>
          </div>
        )}

        {/* Reason */}
        <p className="mt-4 text-xs text-slate-400 leading-relaxed">
          {ch.recommendation_reason}
        </p>
      </div>
    </div>
  );
}
