import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChannelResult } from "../types";
import { fmt } from "../utils";

interface Props {
  channels: ChannelResult[];
  breakeven_roas: number;
}

const REC_COLORS: Record<string, string> = {
  SCALE: "#10b981",
  HOLD: "#f59e0b",
  CUT: "#ef4444",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 shadow-lg rounded-xl p-3 text-sm">
      <p className="font-semibold capitalize text-slate-800 mb-2">{label}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: p.fill || p.color }} />
          <span className="text-slate-500">{p.name}:</span>
          <span className="font-semibold text-slate-800">{fmt(p.value)}x</span>
        </div>
      ))}
    </div>
  );
}

export default function RoasChart({ channels, breakeven_roas }: Props) {
  const data = channels.map((ch) => ({
    channel: ch.channel,
    "Attribution Proxy": ch.attribution_proxy_roas,
    "Incremental ROAS": ch.incremental_roas,
    recommendation: ch.recommendation,
  }));

  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="mb-2">
        <h3 className="font-bold text-slate-900">Attribution Proxy vs Incremental ROAS</h3>
        <p className="text-sm text-slate-400 mt-0.5">
          Grey bars = naive proportional attribution (similar mechanism to platform overcounting).
          Coloured bars = causal estimate. Red line = your breakeven at current margin.
        </p>
      </div>
      <p className="text-xs text-amber-600 bg-amber-50 rounded-lg px-3 py-2 mb-4">
        ⚠️ The grey "Attribution Proxy" is <strong>not</strong> your actual platform-reported ROAS
        — it's a modelled naive baseline. Compare the coloured bars to your own dashboards.
      </p>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} barCategoryGap="30%" barGap={4}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="channel"
            tick={{ fontSize: 12, fill: "#64748b" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v) => `${v}x`}
            tick={{ fontSize: 11, fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#f8fafc" }} />
          <ReferenceLine
            y={breakeven_roas}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{
              value: `breakeven ${fmt(breakeven_roas)}x`,
              position: "right",
              fontSize: 10,
              fill: "#ef4444",
            }}
          />

          {/* Attribution proxy — grey */}
          <Bar dataKey="Attribution Proxy" fill="#cbd5e1" radius={[6, 6, 0, 0]}>
            <LabelList
              dataKey="Attribution Proxy"
              position="top"
              formatter={(v: number) => `${fmt(v, 1)}x`}
              style={{ fontSize: 10, fill: "#94a3b8" }}
            />
          </Bar>

          {/* Incremental ROAS — coloured by recommendation */}
          <Bar dataKey="Incremental ROAS" radius={[6, 6, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={REC_COLORS[entry.recommendation]} />
            ))}
            <LabelList
              dataKey="Incremental ROAS"
              position="top"
              formatter={(v: number) => `${fmt(v, 1)}x`}
              style={{ fontSize: 10, fill: "#475569", fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap gap-4 mt-3 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-slate-300" />
          Attribution proxy (naive)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-emerald-500" />
          Incremental — SCALE
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-amber-400" />
          Incremental — HOLD
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-red-500" />
          Incremental — CUT
        </span>
      </div>
    </div>
  );
}
