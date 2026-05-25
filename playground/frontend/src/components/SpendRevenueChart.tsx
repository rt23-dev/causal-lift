import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtCurrency } from "../utils";

interface SpendRow {
  date: string;
  channel: string;
  spend: number;
}
interface SalesRow {
  date: string;
  revenue: number;
}

interface Props {
  spendData: SpendRow[];
  salesData: SalesRow[];
  channels: string[];
}

const CHANNEL_COLORS: Record<string, string> = {
  facebook: "#3b82f6",
  meta: "#3b82f6",
  google: "#f59e0b",
  tiktok: "#ec4899",
  snapchat: "#facc15",
  twitter: "#38bdf8",
  pinterest: "#ef4444",
  youtube: "#dc2626",
};
function channelColor(ch: string): string {
  return CHANNEL_COLORS[ch.toLowerCase()] ?? "#6366f1";
}

// Aggregate to weekly buckets for readability
function toWeekly<T extends { date: string }>(
  rows: T[],
  aggregate: (bucket: T[]) => Record<string, number>
): Array<Record<string, number | string>> {
  const buckets: Record<string, T[]> = {};
  for (const row of rows) {
    const d = new Date(row.date);
    // ISO week start (Monday)
    const day = d.getDay();
    const offset = day === 0 ? -6 : 1 - day;
    const monday = new Date(d);
    monday.setDate(d.getDate() + offset);
    const key = monday.toISOString().slice(0, 10);
    if (!buckets[key]) buckets[key] = [];
    buckets[key].push(row);
  }
  return Object.entries(buckets)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, bucket]) => ({ date, ...aggregate(bucket) }));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function RevenueTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 shadow-lg rounded-xl p-3 text-sm">
      <p className="font-semibold text-slate-700 mb-1">Week of {label}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-500">{p.name}:</span>
          <span className="font-semibold">{fmtCurrency(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

export default function SpendRevenueChart({ spendData, salesData, channels }: Props) {
  const weeklyRevenue = useMemo(
    () =>
      toWeekly(salesData, (rows) => ({
        revenue: rows.reduce((s, r) => s + r.revenue, 0),
      })),
    [salesData]
  );

  const weeklySpend = useMemo(
    () =>
      toWeekly(spendData, (rows) => {
        const out: Record<string, number> = {};
        for (const ch of channels) out[ch] = 0;
        for (const r of rows) {
          if (out[r.channel] !== undefined) out[r.channel] += r.spend;
        }
        return out;
      }),
    [spendData, channels]
  );

  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-6">
      <div>
        <h3 className="font-bold text-slate-900">Revenue & Spend Over Time</h3>
        <p className="text-sm text-slate-400 mt-0.5">
          Weekly aggregates. Spend spikes = budget experiments that help identify causal effects.
        </p>
      </div>

      {/* Revenue chart */}
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Weekly Revenue
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={weeklyRevenue} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => v.slice(5)} // MM-DD
            />
            <YAxis
              tickFormatter={(v) => fmtCurrency(v)}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              width={52}
            />
            <Tooltip content={<RevenueTooltip />} />
            <Area
              type="monotone"
              dataKey="revenue"
              name="Revenue"
              stroke="#6366f1"
              strokeWidth={2}
              fill="url(#revGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Per-channel spend chart */}
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Weekly Ad Spend by Channel
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={weeklySpend} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => v.slice(5)}
            />
            <YAxis
              tickFormatter={(v) => fmtCurrency(v)}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              width={52}
            />
            <Tooltip content={<RevenueTooltip />} />
            <Legend
              iconType="circle"
              iconSize={8}
              wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
            />
            {channels.map((ch) => (
              <Bar
                key={ch}
                dataKey={ch}
                name={ch}
                stackId="spend"
                fill={channelColor(ch)}
                radius={channels.indexOf(ch) === channels.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
