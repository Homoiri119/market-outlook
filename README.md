# 東京市場 寄り付き前アウトルック アプリ

**米国市場のクローズと日経225先物のナイトセッションを分析し、東京市場が開く前(平日 AM8:00 JST)に
「日経225がどの水準・どの方向で寄り付くか」の見通しを自動で出す**アプリです。

中核となる指標は **オーバーナイトの日経225先物ギャップ** です。AM8:00時点で、CMEの日経225先物は
すでに米国市場のクローズを織り込んでおり、
`予想寄り付きギャップ = (現在の先物価格 − 前日の日経225現物終値) ÷ 前日終値`
は「東京がどこで寄り付くか」に対する市場自身の見積もりになります。これに
「米国市場 → 翌日日経リターン」の線形回帰モデルの予測をブレンドし、
**大幅上昇 / 上昇 / ほぼ横ばい / 下落 / 大幅下落** に分類します。

- 中核データは **yfinance のみ(APIキー不要)**。日経先物が取れれば履歴が浅くても初日から動きます。
- J-Quants / EDINET を設定すると、補助的に**銘柄別判断**もブリーフに併記されます(任意)。

- バックエンド: Python (FastAPI) + SQLite + APScheduler
- フロントエンド: React + TypeScript + Vite
- 通知: Discord Webhook(平日 AM8:00 にモーニング・ブリーフを自動送信)

## 構成

```
backend/   FastAPI アプリ・データ取得・分析・スケジューラ
frontend/  React ダッシュボード(マクロシグナル・銘柄別判断・履歴チャート)
```

## セットアップ

### 1. 認証情報の準備

寄り付き前アウトルック(中核機能)は **yfinance のみで動くため、APIキーは不要**です。
以下はすべて任意の設定です。

- **Discord**(推奨): 通知を送りたいチャンネルで「Incoming Webhook」を作成し、URLを取得します。
  AM8:00のモーニング・ブリーフを受け取れます。
- **J-Quants**(任意): https://jpx-jquants.com/ でアカウント登録。設定すると補助的な銘柄別判断が有効になります。
- **EDINET**(任意): https://disclosure2dl.edinet-fsa.go.jp/ で無料のAPIキーを取得。開示情報の注意フラグに使います。

### 2. バックエンド

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# .env を編集して JQUANTS_MAIL / JQUANTS_PASSWORD (または JQUANTS_REFRESH_TOKEN),
# EDINET_API_KEY, DISCORD_WEBHOOK_URL を設定する

uvicorn app.main:app --reload
```

起動すると以下が行われます:
- SQLite DB (`backend/app.db`) の作成
- `data/target_stocks.json` から対象銘柄(TOPIX Core30相当)の初期登録
- APScheduler起動(平日 `.env` の `DAILY_RUN_HOUR:DAILY_RUN_MINUTE`(デフォルト **8:00 JST**、
  東京市場が開く9:00の直前)に**寄り付き前アウトルック**を自動実行・Discord通知)

### 3. フロントエンド

```powershell
cd frontend
npm install
npm run dev
```

ブラウザで http://localhost:5173 を開きます。Viteの開発サーバーが `/api` をbackend(`http://localhost:8000`)へプロキシします。

## 使い方

1. ダッシュボード上部に**寄り付き前アウトルック**が表示されます。「寄り付き前アウトルックを更新」を
   押すと、以下が実行されます:
   - 米国市場(yfinance)・日経225現物(yfinance)の取得と、**日経225先物のオーバーナイト価格**の取得
   - 予想寄り付きギャップ・回帰モデル予測をブレンドした見通しの算出
   - Discordへのモーニング・ブリーフ送信(Webhook未設定時はログ出力のみ)
   - (J-Quants設定時のみ)補助的な銘柄別判断の併記
2. 平日 AM8:00 には同じ処理がスケジューラで自動実行されます。
3. 「設定」タブで(銘柄別判断を使う場合の)対象銘柄の追加・削除ができます。

## 仕組み(分析ロジック)

### 寄り付き前アウトルック(中核)

1. **先物ギャップ(主指標)**: AM8:00時点のCME日経225先物(`NIY=F`、取得不可なら`NKD=F`)と
   前日の日経225現物終値(`^N225`)から
   `ギャップ = (先物 − 前日終値) ÷ 前日終値` を計算します。先物は米国クローズを織り込んで
   夜間取引されているため、これが「東京の寄り付き」に対する市場の見積もりです。
2. **回帰モデル(補助)**: 「米国市場の前営業日リターン・VIX変化・USD/JPY変化」を特徴量に、
   「翌日の日経225リターン」を予測する線形回帰モデル(直近`HISTORY_DAYS`日で毎回再学習)。
   履歴が20件未満のうちは先物ギャップのみで動作します。
3. **ブレンドと分類**: `期待寄り付き = FUTURES_GAP_WEIGHT × ギャップ + (1−重み) × モデル予測`。
   `OUTLOOK_STRONG_THRESHOLD`(±0.8%)・`OUTLOOK_FLAT_THRESHOLD`(±0.2%)で
   **大幅上昇 / 上昇 / ほぼ横ばい / 下落 / 大幅下落** に分類し、日本語の解説文を生成します。

### ダッシュボードの主な表示項目(熟練投資家向け強化)

- **予測精度トラッキング**: 過去の「予想寄り付き」と、実際の東京市場の寄り付き(`^N225`のOpen)を突き合わせ、
  **方向的中率**と**平均誤差(MAE)**を直近20営業日で表示。予測 vs 実績を折れ線で重ね描きします。
- **予想寄り付きレンジ**: 点推定だけでなく、直近の寄り付きギャップのボラティリティ(±1σ)から
  レンジ(円建て + %)を提示。
- **拡張オーバーナイト指標**: 米国3指数・VIX・USD/JPYに加え、
  **SOXフィラデルフィア半導体指数**(東エレク・アドバンテスト等に影響)、
  **米10年債利回り**(前日比bp)、**WTI原油**を表示。
- **市場コンテキスト**: 日経225の**25日移動平均かい離**(トレンド)と、**VIXレジーム**(平穏/やや警戒/警戒/恐怖)。

### 補助: 銘柄別シグナル(J-Quants設定時のみ)

各銘柄の過去リターンとTOPIXリターンから回帰でβ(感応度)を算出し、
`期待リターン = マクロ予測リターン × β` でBUY/SELL/HOLDを判定します。直近のEDINET開示があれば
根拠コメントに注意フラグとして付加されます。J-Quantsが未設定・データ不足の場合は自動的にスキップされ、
アウトルック本体には影響しません。

## クラウド実行(PC不要・GitHub Actions・無料)

PCを起動しなくても、**GitHub Actions が毎朝 AM8:00 JST に自動で** ①アウトルックを計算 → ②Discord通知 →
③ダッシュボードを **GitHub Pages** に公開します。スマホからURLを開くだけで最新の見通しが見られます。
クラウド側はDB不要のステートレス実行(毎回 yfinance から取得して計算)なので、サーバー管理は不要です。

含まれるファイル:
- [.github/workflows/morning-outlook.yml](.github/workflows/morning-outlook.yml) — 毎朝のcronワークフロー
- [backend/scripts/run_cloud.py](backend/scripts/run_cloud.py) — 計算 → Discord → `docs/` にダッシュボード生成
- `docs/index.html` — 生成される静的ダッシュボード(GitHub Pagesで公開)

### セットアップ手順(GitHubアカウント作成から)

1. **GitHubアカウントを作成**: https://github.com/signup (無料)。
2. **新しいリポジトリを作成**: 右上「+」→「New repository」→ 名前を付けて「Create」。
   Private でも Public でも可(Public ならPagesは無料、Privateも個人なら無料枠でPages可)。
3. **このプロジェクトをpush**: プロジェクトフォルダで(初回のみ):
   ```bash
   git init
   git add .
   git commit -m "initial"
   git branch -M main
   git remote add origin https://github.com/<あなたのユーザー名>/<リポジトリ名>.git
   git push -u origin main
   ```
4. **Discord Webhookを登録(通知が欲しい場合)**: リポジトリの
   `Settings → Secrets and variables → Actions → New repository secret` で
   名前 `DISCORD_WEBHOOK_URL`、値にWebhook URLを設定。
5. **GitHub Pagesを有効化**: `Settings → Pages` →
   Source を「Deploy from a branch」、Branch を `main` / フォルダ `/docs` にして Save。
   数分後、`https://<ユーザー名>.github.io/<リポジトリ名>/` でダッシュボードが公開されます。
6. **動作確認**: `Actions` タブ →「Morning Outlook」→「Run workflow」で手動実行できます。
   以後は平日 AM8:00 JST に自動実行されます。

> 補足:
> - GitHubのcronはUTC基準です。ワークフローは `0 23 * * 0-4`(=日〜木 23:00 UTC = 月〜金 8:00 JST)。
> - スケジュール実行は数分遅れることがありますが、東京の寄り付き(9:00)前には十分間に合います。
> - 60日間リポジトリの更新が無いと、GitHubはスケジュールを自動停止します(通常は毎日の自動コミットで更新されるため問題ありません)。

## API(主要エンドポイント)

- `POST /api/outlook/run` — 寄り付き前アウトルックを今すぐ実行(Discord通知含む)
- `GET  /api/outlook/latest` — 最新のアウトルック
- `GET  /api/outlook/history?limit=30` — アウトルック履歴

## 注意

- 先物ギャップは現物終値との比較のため、先物のベーシス(配当・金利による理論差)ぶんの
  わずかなズレを含みます(通常0.3%未満)。方向感の参考値としてご利用ください。
- 中核のアウトルックはyfinanceのみで動きます。J-Quants / EDINET未設定時は銘柄別判断がスキップされるだけです。
- 本アプリの見通しは統計モデルによる参考情報であり、投資判断の最終的な責任は利用者本人にあります。
