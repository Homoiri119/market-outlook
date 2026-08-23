import { FormEvent, useEffect, useState } from "react";
import { api, TargetStock } from "../api/client";

export default function Settings() {
  const [stocks, setStocks] = useState<TargetStock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [name, setName] = useState("");
  const [edinetName, setEdinetName] = useState("");

  const loadStocks = async () => {
    setLoading(true);
    setError(null);
    try {
      setStocks(await api.getTargetStocks());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStocks();
  }, []);

  const handleAdd = async (e: FormEvent) => {
    e.preventDefault();
    if (!code || !name) return;
    setError(null);
    try {
      await api.addTargetStock({ code, name, edinet_company_name: edinetName || null });
      setCode("");
      setName("");
      setEdinetName("");
      await loadStocks();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleRemove = async (stockCode: string) => {
    setError(null);
    try {
      await api.removeTargetStock(stockCode);
      await loadStocks();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="settings">
      <h2>対象銘柄の設定</h2>
      {error && <div className="error">{error}</div>}

      <form className="add-stock-form" onSubmit={handleAdd}>
        <input placeholder="銘柄コード (例: 7203)" value={code} onChange={(e) => setCode(e.target.value)} />
        <input placeholder="銘柄名" value={name} onChange={(e) => setName(e.target.value)} />
        <input
          placeholder="EDINET提出者名 (任意)"
          value={edinetName}
          onChange={(e) => setEdinetName(e.target.value)}
        />
        <button type="submit">追加</button>
      </form>

      <p className="hint">
        ※ 通知設定 (Discord Webhook) や J-Quants / EDINET の認証情報は backend の <code>.env</code> ファイルで設定します。
      </p>

      {loading ? (
        <div>読み込み中...</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>コード</th>
              <th>銘柄名</th>
              <th>EDINET提出者名</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {stocks.map((s) => (
              <tr key={s.code}>
                <td>{s.code}</td>
                <td>{s.name}</td>
                <td>{s.edinet_company_name ?? "-"}</td>
                <td>
                  <button onClick={() => handleRemove(s.code)}>削除</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
