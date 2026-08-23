import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Settings from "./pages/Settings";

type Page = "dashboard" | "settings";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");

  return (
    <div className="app">
      <header className="app-header">
        <h1>日本株 売買判断ダッシュボード</h1>
        <nav>
          <button className={page === "dashboard" ? "active" : ""} onClick={() => setPage("dashboard")}>
            ダッシュボード
          </button>
          <button className={page === "settings" ? "active" : ""} onClick={() => setPage("settings")}>
            設定
          </button>
        </nav>
      </header>
      <main>{page === "dashboard" ? <Dashboard /> : <Settings />}</main>
    </div>
  );
}
