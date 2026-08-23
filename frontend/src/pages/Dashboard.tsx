import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { api, MacroSignal, MorningOutlook, OutlookDirection, StockSignal } from "../api/client";

function signalClass(signal: string): string {
  switch (signal) {
    case "BUY":
      return "signal-buy";
    case "SELL":
      return "signal-sell";
    default:
      return "signal-neutral";
  }
}

const DIRECTION_LABELS: Record<OutlookDirection, string> = {
  STRONG_UP: "大幅上昇",
  UP: "上昇",
  FLAT: "ほぼ横ばい",
  DOWN: "下落",
  STRONG_DOWN: "大幅下落",
};

const DIRECTION_EMOJI: Record<OutlookDirection, string> = {
  STRONG_UP: "🚀",
  UP: "📈",
  FLAT: "➡️",
  DOWN: "📉",
  STRONG_DOWN: "⚠️",
};

function directionClass(direction: OutlookDirection): string {
  if (direction === "STRONG_UP" || direction === "UP") return "outlook-up";
  if (direction === "STRONG_DOWN" || direction === "DOWN") return "outlook-down";
  return "outlook-flat";
}

function pct(value: number | null | undefined, digits = 2): string {
  return value == null ? "N/A" : `${(value * 100).toFixed(digits)}%`;
}

export default function Dashboard() {
  const [outlook, setOutlook] = useState<MorningOutlook | null>(null);
  const [outlookHistory, setOutlookHistory] = useState<MorningOutlook[]>([]);
  const [macroSignal, setMacroSignal] = useState<MacroSignal | null>(null);
  const [stockSignals, setStockSignals] = useState<StockSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [latest, history, today] = await Promise.all([
        api.getLatestOutlook(),
        api.getOutlookHistory(30),
        api.getTodaySignals(),
      ]);
      setOutlook(latest);
      setOutlookHistory(history.slice().reverse());
      setMacroSignal(today.macro_signal);
      setStockSignals(today.stock_signals);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      await api.runOutlook();
      await loadData();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const chartData = outlookHistory.map((o) => ({
    date: o.date,
    expected_move: Number((o.expected_move * 100).toFixed(3)),
  }));

  return (
    <div className="dashboard">
      <div className="toolbar">
        <button onClick={handleRun} disabled={running}>
          {running ? "分析中..." : "寄り付き前アウトルックを更新"}
        </button>
        <button onClick={loadData} disabled={loading}>
          再読み込み
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div>読み込み中...</div>}

      {!loading && (
        <>
          <section className="morning-outlook">
            <h2>東京市場 寄り付き前アウトルック</h2>
            {outlook ? (
              <div className={`outlook-card ${directionClass(outlook.direction)}`}>
                <div className="outlook-badge">
                  <div className="outlook-emoji">{DIRECTION_EMOJI[outlook.direction]}</div>
                  <div className="outlook-label">{DIRECTION_LABELS[outlook.direction]}</div>
                </div>
                <div className="outlook-body">
                  <div className="outlook-headline">
                    予想寄り付き: <strong>{pct(outlook.expected_move)}</strong>
                    <span className="outlook-conf"> (信頼度 {outlook.confidence.toFixed(2)})</span>
                  </div>
                  <div className="outlook-grid">
                    <div>
                      <span className="k">日付</span>
                      <span className="v">{outlook.date}</span>
                    </div>
                    <div>
                      <span className="k">日経225 前日終値</span>
                      <span className="v">
                        {outlook.nikkei_prev_close != null ? outlook.nikkei_prev_close.toLocaleString() : "N/A"}
                      </span>
                    </div>
                    <div>
                      <span className="k">予想寄り付き水準</span>
                      <span className="v">
                        {outlook.implied_open_level != null ? Math.round(outlook.implied_open_level).toLocaleString() : "N/A"}
                      </span>
                    </div>
                    <div>
                      <span className="k">先物ギャップ(主指標)</span>
                      <span className="v">{pct(outlook.implied_gap)}</span>
                    </div>
                    <div>
                      <span className="k">回帰モデル予測</span>
                      <span className="v">{pct(outlook.model_return)}</span>
                    </div>
                    <div>
                      <span className="k">日経先物</span>
                      <span className="v">
                        {outlook.nikkei_futures != null ? outlook.nikkei_futures.toLocaleString() : "N/A"}
                        {outlook.futures_source ? ` (${outlook.futures_source})` : ""}
                      </span>
                    </div>
                  </div>
                  {outlook.us_detail && <div className="detail">{outlook.us_detail}</div>}
                  {outlook.narrative && <pre className="outlook-narrative">{outlook.narrative}</pre>}
                </div>
              </div>
            ) : (
              <p>まだアウトルックがありません。「寄り付き前アウトルックを更新」を押してください。</p>
            )}
          </section>

          {chartData.length > 0 && (
            <section className="outlook-chart">
              <h2>アウトルック履歴(予想寄り付きリターン %)</h2>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <ReferenceLine y={0} stroke="#999" />
                  <Line type="monotone" dataKey="expected_move" stroke="#2563eb" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </section>
          )}

          {macroSignal && (
            <section className="macro-signal">
              <h2>参考: マクロシグナル(翌営業日TOPIX回帰)</h2>
              <div className={`signal-card ${signalClass(macroSignal.signal)}`}>
                <div className="signal-badge">{macroSignal.signal}</div>
                <div>
                  <div>日付: {macroSignal.date}</div>
                  <div>予測TOPIXリターン: {(macroSignal.predicted_return * 100).toFixed(3)}%</div>
                  <div>信頼度 (R²): {macroSignal.confidence.toFixed(3)}</div>
                </div>
              </div>
            </section>
          )}

          {stockSignals.length > 0 && (
            <section className="stock-signals">
              <h2>参考: 銘柄別判断</h2>
              <table>
                <thead>
                  <tr>
                    <th>コード</th>
                    <th>銘柄名</th>
                    <th>判断</th>
                    <th>期待リターン</th>
                    <th>β</th>
                    <th>根拠</th>
                  </tr>
                </thead>
                <tbody>
                  {stockSignals
                    .slice()
                    .sort((a, b) => b.expected_return - a.expected_return)
                    .map((s) => (
                      <tr key={s.code}>
                        <td>{s.code}</td>
                        <td>{s.name}</td>
                        <td>
                          <span className={`signal-pill ${signalClass(s.signal)}`}>{s.signal}</span>
                        </td>
                        <td>{(s.expected_return * 100).toFixed(3)}%</td>
                        <td>{s.beta.toFixed(2)}</td>
                        <td className="reason">{s.reason}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </div>
  );
}
