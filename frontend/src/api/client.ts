export interface MacroSignal {
  date: string;
  predicted_return: number;
  confidence: number;
  signal: "BUY" | "SELL" | "NEUTRAL";
  detail: string | null;
}

export interface StockSignal {
  date: string;
  code: string;
  name: string;
  expected_return: number;
  beta: number;
  signal: "BUY" | "SELL" | "HOLD";
  reason: string;
}

export interface TodaySignals {
  macro_signal: MacroSignal | null;
  stock_signals: StockSignal[];
}

export type OutlookDirection = "STRONG_UP" | "UP" | "FLAT" | "DOWN" | "STRONG_DOWN";

export interface MorningOutlook {
  date: string;
  direction: OutlookDirection;
  expected_move: number;
  confidence: number;
  implied_gap: number | null;
  model_return: number | null;
  nikkei_prev_close: number | null;
  nikkei_futures: number | null;
  implied_open_level: number | null;
  futures_source: string | null;
  us_detail: string | null;
  us_market: UsMarket | null;
  narrative: string | null;
}

export interface UsMarket {
  date: string | null;
  sp500_return: number | null;
  nasdaq_return: number | null;
  dow_return: number | null;
  vix_close: number | null;
  vix_change: number | null;
  usdjpy_return: number | null;
  sentiment: "risk_on" | "risk_off" | "neutral";
}

export interface OutlookRunResult {
  date: string;
  collection_results: Record<string, number>;
  direction: OutlookDirection;
  expected_move: number;
  confidence: number;
  stock_signal_count: number;
  notification_sent: boolean;
}

export interface TargetStock {
  code: string;
  name: string;
  edinet_company_name: string | null;
}

export interface RunResult {
  date: string;
  collection_results: Record<string, number>;
  macro_signal: string;
  stock_signal_count: number;
  notification_sent: boolean;
}

const BASE_URL = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API request failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getLatestOutlook: () => request<MorningOutlook | null>("/outlook/latest"),
  getOutlookHistory: (limit = 30) => request<MorningOutlook[]>(`/outlook/history?limit=${limit}`),
  runOutlook: () => request<OutlookRunResult>("/outlook/run", { method: "POST" }),
  getTodaySignals: () => request<TodaySignals>("/signals/today"),
  getMacroHistory: (limit = 30) => request<MacroSignal[]>(`/signals/macro/history?limit=${limit}`),
  getStockHistory: (code: string, limit = 30) =>
    request<StockSignal[]>(`/signals/stock/${code}/history?limit=${limit}`),
  getTargetStocks: () => request<TargetStock[]>("/stocks"),
  addTargetStock: (stock: TargetStock) =>
    request<TargetStock>("/stocks", { method: "POST", body: JSON.stringify(stock) }),
  removeTargetStock: (code: string) => request<{ removed: boolean }>(`/stocks/${code}`, { method: "DELETE" }),
  runDaily: () => request<RunResult>("/run/daily", { method: "POST" }),
};
