# auto-mosaic

街中散歩動画（iPhone 4K60fps等）の**顔**や**車のナンバープレート**に自動でモザイク／ぼかしを適用する、高速・完全ローカル動作のAIデスクトップアプリ（GUI / CLI両対応）です。

- **完全ローカル完結**：クラウド送信なし（プライバシー・著作権配慮）
- **高精度AI検出（顔 ＆ ナンバープレート対応）**：SCRFD顔検出 ＋ YOLOv11nナンバープレート検出（ONNX Runtime / OpenVINO / CoreML / DirectML 高速推論）
- **PC・Mac負荷最適化（モザイク対象の選択）**：「顔のみ」「ナンバープレートのみ」「顔＋ナンバー両方」を用途に応じて選択可能。不要なAIモデルをロード・実行しないことでCPU/GPU/NPU・メモリ負荷を最小化
- **自然な丸型モザイク ＆ プレート専用矩形モザイク**：顔には輪郭にフィットする楕円・丸型モザイク、ナンバープレートには四隅の数字まで確実に隠す矩形モザイクを自動適用
- **ちらつき防止・前後フレームパディング**：IoU＋速度予測追跡 ＋ 前後Nフレームへのぼかし延長
- **誤検知防止（トラック単位判定）**：イラスト・アニメ顔除外判定、YCbCr肌色判定（彫像・銅像除外）、動物除外判定、写真・ポスター除外（ナンバープレートは顔用フィルタを自動バイパス）
- **色再現性（カラーマネジメント）**：BT.2020 / BT.709 静的メタデータタグ・レンジの完全継承（色味変化の完全防止）
- **バッチ処理**：単一動画／複数指定／フォルダ一括処理、エラー時スキップ＆サマリー出力

---

## 1. 必要要件

- **macOS** (Apple Silicon / Intel) または **Windows**
- **Python 3.10以上**
- **ffmpeg**
  - macOS: `brew install ffmpeg`
  - Windows: `choco install ffmpeg` または `winget install Gyan.FFmpeg`

---

## 2. セットアップ手順

### 2.1 Windows の場合（ワンクリック自動セットアップ推奨）

Windows環境では、配布ZIPを展開して `start-app.bat` をダブルクリックするだけで、**Python環境（未導入時はポータブル版Python）、仮想環境（venv）、FFmpeg、SCRFD顔検出モデルのダウンロードからGUI起動まで全自動**で行われます。
初回実行時はセットアップ進捗が画面に表示され、2回目以降はバックグラウンドで即座にGUIが起動します。

```cmd
# リポジトリ直下のバッチファイルをダブルクリックまたはコマンドラインから実行
start-app.bat
```

手動で構築する場合：
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/requirements-windows.txt
python scripts/download_models.py
```

### 2.2 macOS の場合（ワンクリック自動セットアップ推奨）

macOS環境では、配布ZIPを展開して `start-app.command` をダブルクリックするだけで、**Python環境チェック、仮想環境（venv）、FFmpegチェック、SCRFD顔検出モデルのダウンロードからGUI起動まで全自動**で行われます。
*(詳細は後述の「5. Gatekeeper警告の回避方法 (Mac版)」をご確認ください)*

```bash
# ターミナルから直接実行する場合
./start-app.command
# または
bash start-app.command
```

手動で構築する場合：

```bash
git clone https://github.com/kh813/auto-mosaic.git
cd auto-mosaic

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージ導入
pip install --upgrade pip
pip install -r requirements/requirements-mac.txt
```

#### SCRFDモデルのダウンロード

自動ダウンロードスクリプトを実行してONNXモデルを取得します。

```bash
python scripts/download_models.py
```

`models/` ディレクトリに以下のモデルが配置されます：
- `scrfd_2.5g_bnkps.onnx` (顔検出：高速・60+ FPS)
- `scrfd_10g_bnkps.onnx` (顔検出：標準・推奨デフォルト)
- `scrfd_34g_gnkps.onnx` (顔検出：高精度)
- `license-plate-finetune-v1n.onnx` (ナンバープレート検出：YOLOv11n 軽量・高精度)

---

## 3. 使い方（GUI操作ガイド）

本ツールは直感的なデスクトップGUI画面から、マウス操作だけで動画のモザイク処理を行えます。

- **Windows**: `start-app.bat` をダブルクリック（初回は進捗表示、2回目以降はサイレント起動）
- **macOS**: `start-app.command` をダブルクリック（初回セットアップおよびGUI起動）

### 3.1 かんたん3ステップ操作

1. **動画を追加する**:
   - `+ Add File(s)...` または `+ Add Folder...` ボタンをクリックして動画を選択します。
   - エクスプローラーやFinderから動画ファイル・フォルダを直接ウィンドウ内へ**ドラッグ＆ドロップ**しても追加できます。
2. **設定を確認・調整する**:
   - 基本的には**初期設定（デフォルト）のままで最適**に動作します（AIモデルもお使いのPCスペックに合わせて自動選択されます）。
   - モザイク方式（ガウスぼかし / ピクセルモザイク）や形状（丸型 / 四角型）をお好みに合わせて変更できます。
3. **変換を開始する**:
   - `▶ Start Processing` ボタンをクリックするとバッチ処理がスタートします。
   - 右側のプレビュー画面でリアルタイムに処理映像（Pass 1: 顔検出枠表示 / Pass 2: 実際のモザイク描画）を確認できます。
   - 処理が完了すると、鮮やかなエメラルドグリーンの **`▶ Open Processed Video`**（完成動画を再生）および **`📁 Open Output Folder`**（保存先フォルダを開く）が有効化されます。

---

### 3.2 設定パラメータ一覧（直感的なレイアウト）

設定パネルは、思考順序および処理パイプライン（モザイク対象 → AI顔検出 → 範囲・強度 → 形状 → 時間補間 → 出力品質）に沿って論理的に配置されています：

| 行 | 設定項目 | 説明・推奨値 |
|---|---|---|
| **Row 0** | **Mosaic Target**<br>モザイク対象（負荷最適化） | `Faces Only (顔のみ)`、`Faces + Plates (顔＋ナンバー両方)`、`Plates Only (ナンバーのみ)` から選択。不要なモデルを動かさないことでPC/Macの負荷を最適化します。 |
| | **Plate Confidence**<br>ナンバー検出信頼度 | AIが「ナンバープレートである」と判定する閾値（既定: `0.40`）。ナンバープレート検出有効時のみアクティブになります。 |
| **Row 1** | **Model**<br>AI顔検出モデル | 顔検出を行うSCRFDモデル。ほとんどの用途には標準の `scrfd_10g` が推奨されます。PCスペックに応じて自動選択されます。 |
| | **Detection Confidence**<br>顔検出信頼度 | AIが「顔である」と判定する閾値（既定: `0.50`）。数値を下げると遠方の小顔も検出しやすくなり、上げると誤検出を抑制します。 |
| **Row 2** | **Face Margin (%)**<br>顔拡大マージン | 検出された顔のバウンディングボックスを周囲に何%拡大してモザイクをかけるか（既定: `50%` ★推奨）。頭部・輪郭・髪型まで確実にカバーします。 |
| | **Blur Strength**<br>ぼかし強度／サイズ | ガウスぼかし時のカーネルサイズ（奇数・既定: `51`）またはピクセルモザイク時のブロック幅（既定: `28`）。数値が大きいほど強力にぼかします。 |
| **Row 3** | **Blur Type**<br>ぼかし方式 | `Gaussian`（滑らかなガウスぼかし）または `Mosaic`（定番のブロックモザイク）。 |
| | **Blur Shape**<br>モザイク形状 | `Round (Ellipse) ★推奨`（自然な丸型・四隅の余計な背景を巻き込まない）または `Rectangle (Square)`（従来の四角形）。※ナンバープレートは常に四隅まで隠れる四角形が適用されます。 |
| **Row 4** | **Pad Backward**<br>後方パディング | 顔やナンバーが最初に検出されたフレームより何フレーム前から先行モザイクをかけるか（既定: `12` フレーム）。接近する被写体の見逃しを防ぎます。 |
| | **Pad Forward**<br>前方パディング | 顔やナンバーが最後に検出されたフレームより何フレーム後までモザイクを継続するか（既定: `24` フレーム）。一時的に隠れたり振り向いた場合の途切れを防止。動画のfps（60/120/240fps）に合わせて自動スケーリングされます。 |
| **Row 5** | **Output Codec**<br>出力コーデック | `Auto (元動画と同じ) ★推奨`（元動画のコーデック HEVC / H.264、コンテナ、ビット深度、色空間をそのまま維持し、Windows / Mac / iPad / iPhone すべてで確実に再生可能）、`HEVC (H.265)`（高圧縮・小サイズ化）、または `H.264`（レガシー機器互換）。 |

| | **CRF Quality**<br>画質・容量バランス | 映像品質（既定: `23` ★推奨）。`標準 (CRF 23: 見た目の劣化なし＆容量約50%減)`、`最高画質 (CRF 18: 大容量)`、`容量優先 (CRF 26)` からワンクリック選択可能。数値が小さいほど高画質。 |
| | **Bit Depth**<br>ビット深度 | `Auto (元動画に合わせる)`、`8-bit (互換性・サイズ削減優先)`、`10-bit (高階調維持)`。iPhone HDR動画を8-bit化することで容量削減と再生互換性を両立。 |
| | **FastStart & Params**<br>最適化＆命名規則 | MP4の即時再生最適化（`-movflags +faststart`）を常時自動適用。設定パラメータを出力ファイル名に自動埋め込み可能（例: `_blurred_(G51-M50-CRF23).mp4`）。 |

---

### 3.3 便利なGUI機能

- **マウスドラッグ＆ボタンによる処理順序の並び替え**:
  - キュー一覧の動画行をマウスで直接ドラッグ＆ドロップして、希望の位置へ直感的に並び替えられます。
  - `▲ Up` / `▼ Down` ボタンやショートカットキー（`Alt+Up` / `Alt+Down`）にも対応しています。
  - 右クリックメニューから `⤒ Move to Top`（先頭へ移動）や `⤓ Move to Bottom`（末尾へ移動）も即座に実行できます。
  - **実行中の動的順序同期**: バックグラウンドで変換が実行中であっても、待機中の動画の順番を並び替えると、次の処理順序へリアルタイムに即座反映されます。
- **選択した動画だけのピンポイント実行**:
  - キュー内の行をクリックして選択すると、ボタンが `▶ Start Processing (1 selected)` に変化し、選択した動画のみを実行できます。未選択時はキュー内の未完了動画を順次一括処理します。
- **きめ細やかなキャンセル・停止機能**:
  - `⏭ Skip / Cancel Current File`: 現在処理中の動画のみを即座にスキップし、キューに残っている次の動画へ安全に進みます。
  - `⏹ Stop Processing`: 実行中の変換を安全に中断します（多重起動を防ぐステートマシン制御）。
- **最後に開いたフォルダの自動記憶**:
  - ファイルやフォルダを追加したディレクトリを自動記憶（`QSettings`）。次回起動時やダイアログ表示時に同じフォルダを素早く開けます。
- **実行ログ＆トラブルシューティング支援**:
  - 画面下部のログ欄に加え、`logs/face_mosaic.log` に詳細ログを自動記録。「📋 Copy Log」でクリップボードへ一括コピー、「📁 Open Log File」でログファイルを直接開けます。
- **cmd.exe / ターミナルのポップアップ完全抑制（Windows）**:
  - 変換中の FFmpeg / FFprobe / QSV コーデック判定などの子プロセス呼び出し時にも一切コンソールウィンドウを出現させない完全サイレント設計（`CREATE_NO_WINDOW` & `SW_HIDE`）。

---

### 3.4 中断・再開機能（チェックポイント＆レジューム）

4K 60fps の長尺動画や長時間のバッチ処理中に予期せぬ中断（強制終了・電源断・キャンセル）が発生しても、**最初からやり直すことなく安全に途中再開できるチェックポイント機構**を備えています。

- **超軽量設計（数十GBの不要キャッシュなし）**:
  - 保存するのは顔検出の座標情報（JSONL）のみ。4K長尺動画でもチェックポイント容量はわずか数MB〜十数MBに収まり、ディスクを圧迫しません。
- **自動クリーンアップ**:
  - 処理が 100% 完了した瞬間に中間作業フォルダ `.{出力ファイル名}.checkpoint/` は **自動的に完全消去** されます。
- **自動レジューム**:
  - Pass 1（顔検出）の途中で中断した場合は記録されたフレームから再開、Pass 2（モザイク描画）の途中で中断した場合は Pass 1 を丸ごとスキップして Pass 2 から即座に再開します。

### 3.5 自動アップデート機能（Self-Update）

auto-mosaic には、ワンクリックで最新リリースを確認・取得できるセルフアップデート機能が備わっています。

- **GUIからの更新**:
  - メニューバーの `Help` → `🔄 Check for Updates...` または画面下部のログボタン横にある `🔄 Check Update` ボタンをクリック。
  - アプリ起動2秒後にバックグラウンドで最新バージョンを静かに確認し、新バージョンがある場合はボタンが鮮やかな緑色（`🎉 Update to vX.X.X`）に変わり、更新ダイアログが表示されます。
  - 「はい」を選択するだけで、進捗インジケーター付きで自動更新が行われます。
- **Gitリポジトリ／スタンドアロン自動判別**:
  - Gitリポジトリで運用している場合は自動で `git pull` ＋ 仮想環境（venv）の依存ライブラリ更新を実行。
  - 配布ZIP（スタンドアロン版）の場合はGitHub Releasesから最新アーカイブを自動取得し、安全に上書き配置します。

---

## 4. 上級者向け機能：CLI（コマンドライン）での実行

スクリプトによるバッチ自動化や、サーバー・ヘッドレス環境で一括処理を行いたい上級者向けのコマンドラインインターフェースです。

### 4.1 基本的な実行

```bash
# 単一ファイルの処理
python -m face_mosaic.cli path/to/walk_video.mov

# 複数ファイルの一括指定
python -m face_mosaic.cli video1.mov video2.mp4 video3.mov

# フォルダ内の動画を一括バッチ処理（再帰的探索）
python -m face_mosaic.cli /path/to/videos/ --output-dir /path/to/output/
```

### 4.2 オプション指定例

```bash
# 顔と車のナンバープレートの両方にモザイクをかける
python -m face_mosaic.cli video.mp4 --target both

# 車のナンバープレートのみにモザイクをかける（車載動画等・PC/Mac負荷軽減）
python -m face_mosaic.cli video.mp4 --plates-only --plate-conf 0.40

# 顔のみにモザイクをかける（デフォルト・PC/Mac負荷軽減）
python -m face_mosaic.cli video.mp4 --faces-only

# ピクセルモザイク方式・ブロックサイズ指定
python -m face_mosaic.cli video.mp4 --blur-type mosaic --strength 28 --shape ellipse

# ガウスぼかし強度・検出閾値・前後パディングフレーム数指定
python -m face_mosaic.cli video.mp4 --blur-type gaussian --strength 51 --conf 0.5 --pad-back 12 --pad-fwd 24

# 高速モデル (2.5G)・H.264エンコーダ指定
python -m face_mosaic.cli video.mp4 --model scrfd_2.5g_bnkps.onnx --codec h264 --crf 18

# フォルダ監視モード（新規動画が追加されたら自動処理）
python -m face_mosaic.cli /path/to/watch_folder --watch

# 設定ファイル (YAML) を指定して実行
python -m face_mosaic.cli video.mp4 --config configs/config.default.yaml

# アップデート確認・実行 (CLI)
python -m face_mosaic.cli --check-update
python -m face_mosaic.cli --update
```

### 4.3 コマンドライン引数一覧

| 引数 | 説明 | デフォルト値 |
|---|---|---|
| `inputs` | 処理対象の動画ファイルまたはフォルダ（複数指定可） | 必須 |
| `--target, -t` | モザイク対象 (`faces`: 顔のみ, `plates`: ナンバーのみ, `both`: 両方) | `faces` |
| `--faces-only` | 顔モザイクのみ実行 (`--target faces` のショートカット) | `False` |
| `--plates-only` | ナンバープレートモザイクのみ実行 (`--target plates` のショートカット) | `False` |
| `--both` | 顔とナンバープレートの両方を実行 (`--target both` のショートカット) | `False` |
| `--plate-conf` | ナンバープレート検出の信頼度閾値 (0.0〜1.0) | `0.40` |
| `--plate-model` | 使用するナンバープレート検出ONNXモデル名 | `license-plate-finetune-v1n.onnx` |
| `--config, -c` | 設定ファイル（YAML）のパス | `configs/config.default.yaml` |
| `--model, -m` | 使用するSCRFDモデル名 | `scrfd_10g_bnkps.onnx` |
| `--blur-type, -b` | ぼかし方式 (`gaussian` または `mosaic`) | `gaussian` |
| `--shape` | モザイク・ぼかしの形状 (`ellipse` 丸型 または `rect` 四角形) | `ellipse` |
| `--strength, -s` | ぼかし強度（奇数整数）またはモザイクブロックサイズ | `51` / `28` |
| `--conf` | 顔検出の信頼度閾値 (0.0〜1.0) | `0.5` |
| `--pad-back` | 顔検出前のぼかし延長フレーム数 (N)（事前モザイク） | `12` |
| `--pad-fwd` | 顔検出後のぼかし延長フレーム数 (M) | `24` |
| `--no-illustration-filter` | イラスト・アニメ顔の自動除外フィルターを無効化 | `False` |
| `--codec` | 出力動画コーデック (`auto`: 元動画と同じ, `hevc`, または `h264`) | `auto` |
| `--crf` | CRF品質パラメータ（数値が小さいほど高画質） | `18` |
| `--output-dir, -o` | 出力フォルダ（省略時は入力と同じフォルダ） | `""` |
| `--no-recursive` | フォルダ探索時に再帰探索を無効化 | `False` |
| `--watch` | フォルダ監視モード（新規動画を検知して自動処理） | `False` |
| `--check-update` | GitHub Releasesから最新バージョンを確認 | `False` |
| `--update, --self-update` | auto-mosaic を最新バージョンに自動更新 | `False` |


---

## 5. Gatekeeper警告の回避方法 (Mac版配布ZIP利用時)

GitHub Releases等からダウンロードしたZIPをmacOSで展開し、`start-app.command` を初めて実行する際、macOSのセキュリティ保護機能（Gatekeeper）によって「開発元が未確認のため開けません」または「悪質なソフトウェアかどうかを検証できないため開けません」という警告が表示される場合があります。

以下のいずれかの方法で簡単に開くことができます：

### 方法A: 右クリックから開く（最も簡単・おすすめ）
1. Finderで `start-app.command` を **「Controlキーを押しながらクリック」（または右クリック）** します。
2. 表示されるコンテキストメニューから **「開く」** を選択します。
3. 確認ダイアログが表示されるので、**「開く」** ボタンをクリックします。
   *(一度この操作を行うとmacOSに許可が記録され、次回以降は通常通りダブルクリックだけで起動します)*

### 方法B: ターミナルから隔離属性（quarantine）を解除する
ターミナルを開き、展開したフォルダ内で以下のコマンドを実行します：
```bash
xattr -d com.apple.quarantine start-app.command
```
または、直接 bash から起動することも可能です：
```bash
bash start-app.command
```

> [!TIP]
> 配布ZIPアーカイブはGitHub Actions上で実行可能権限（`chmod +x`）を付与した状態でパッケージ化されています。通常の「アーカイブユーティリティ」（ダブルクリック展開）で解凍した場合、手動で `chmod +x` を実行する必要はありません。

---

## 6. テスト・ベンチマークの実行

```bash
# 全単体・結合テストの実行（全75テスト）
pytest -v

# SCRFD推論速度のベンチマーク計測
python scripts/benchmark.py
```

---

## 7. ハードウェア自動検知と最適化仕様 (Intel Arc / Apple Silicon / AMD Ryzen)

本ツールは起動時にマシンの CPU / GPU / NPU アーキテクチャおよび RAM 容量を自動診断し、それぞれのハードウェアに特化した推論アクセラレータおよびハードウェアエンコーダーへ自動分岐します。

### ハードウェア別 最適化マトリックス

| ハードウェア環境 | Pass 1 顔検出推論 | Pass 2 レンダリング（動画エンコード） | 特徴・最適化内容 |
|---|---|---|---|
| **Intel Arc / Core Ultra** (Windows) | **OpenVINO (`MULTI:GPU,NPU`)** | **Intel QSV (`hevc_qsv` / `h264_qsv`)** | Arc GPU と AI Boost NPU の協調処理（約 135〜155 FPS）。メディアエンジンによる 4K60fps 爆速書き出し。 |
| **Apple Silicon (M1/M2/M3/M4)** (macOS) | **CoreML (`CoreMLExecutionProvider`)** | **Apple VideoToolbox (`hevc_videotoolbox`)** | Apple Neural Engine (ANE) & Metal GPU 推論。Mac 専用メディアエンジンによる高速・超低発熱エンコード（10-bit HDR 対応）。 |
| **AMD Ryzen / Radeon** (Windows/Linux) | **DirectML (Radeon) / AVX2・AVX-512** | **AMD AMF (`hevc_amf`) / `-threads 0`** | Radeon GPU での DirectML 推論。CPU 時は Zen アーキテクチャのマルチスレッド（16〜32スレッド）をフル活用。 |
| **汎用 CPU / NVIDIA GPU** | **CUDA / CPUExecutionProvider** | **NVENC (`hevc_nvenc`) / `libx265`** | NVIDIA GPU 時は NVENC、CPU 時はマルチスレッド並列処理で安全にフォールバック。 |

---

### 7.1 Intel Arc / Core Ultra での最適化 (既存仕様完全維持)

Windows（Intel Core Ultra 環境）では、ネイティブ OpenVINO ランタイムを通じて **Intel Arc GPU と Intel AI Boost NPU の協調処理 (`MULTI:GPU,NPU`)** を自動活用します。
また、動画書き出しには **Intel Quick Sync Video (`hevc_qsv`)** を使用します。

- **実機での推論ベンチマーク性能（Intel Core Ultra 5 225H / Arc 130T GPU）**:
  - **OpenVINO GPU / NPU 推論**: **約 135〜155 FPS**（4K 60fps 動画の推論速度を 2.5倍以上上回るリアルタイム超高速処理）
  - **DirectML (DirectX 12) / ONNX Runtime**: 約 16〜43 FPS（DirectML 環境への自動安全フォールバック対応）
- **長尺動画での OOM 防止（メモリリーク完全遮断）**:
  - 4K 生フレーム（1フレーム約 25MB）への NumPy スライス参照を遮断し、顔トラッキング用クロップ画像を最大 128px にリサイズした上で独立メモリバッファとして保持。
  - 数万フレーム（数十GB規模）の 4K 60fps 動画であっても、メモリ使用量が数GB以内に一定制御され、クラッシュなく安定して完走します。

### 7.2 Apple Silicon (M1/M2/M3/M4) での最適化

macOS（Apple Silicon 環境）では、Apple 専用のアクセラレータハードウェアを自動検知してパイプライン全体を最適化します：

- **Pass 1 推論：Apple Neural Engine (ANE) & Metal GPU 協調 (`CoreMLExecutionProvider`)**:
  - ONNX Runtime において `CoreMLExecutionProvider` を最優先でロード。
  - M1 / M2 / M3 / M4 チップの Neural Engine および Apple GPU をフル稼働させ、極めて低消費電力かつリアルタイムを大きく上回る速度で顔検出を実行します。
  - 統合メモリ（Unified Memory）を考慮し、16GB+ マシンでは高精度モデル（34G）を自動選択。
- **Pass 2 レンダリング：Apple VideoToolbox ハードウェアエンコード (`hevc_videotoolbox`)**:
  - Mac 専用のハードウェアメディアエンジン（VideoToolbox）を自動判別して直接パイプラインに接続。
  - ソフトウェア（CPU）エンコードと比較して書き出し速度が数倍に向上し、発熱やファン回転、バッテリー消費を大幅に抑制します。
  - **10-bit HDR / BT.2020 保持**: `-profile:v main10 -pix_fmt p010le` を自動適用し、macOS QuickTime Player や Final Cut Pro 等での色調ズレを完全に防止します。

### 7.3 AMD Ryzen / Radeon での最適化

AMD プロセッサおよびグラフィックス環境では、Zen アーキテクチャの強力な並列計算能力を最大限に引き出します：

- **Pass 1 推論：Radeon GPU (DirectML) ＆ Zen AVX2/AVX-512 マルチスレッド**:
  - AMD Radeon GPU（内蔵 APU または外付け GPU）環境では、DirectX 12 経由の `DmlExecutionProvider` による GPU 加速を実行。
  - CPU 実行時は、Ryzen のマルチコア（16〜32スレッド）に最適化した `intra_op_num_threads` 設定により、AVX2 / AVX-512 命令セットをフル活用して推論を行います。
- **Pass 2 レンダリング：AMD AMF ハードウェアエンコード (`hevc_amf`) ＆ マルチスレッド CPU**:
  - Radeon GPU 搭載時は **AMD Advanced Media Framework (`hevc_amf`)** を自動活用。
  - CPU ソフトウェアエンコード時（`libx265`）でも明示的に `-threads 0` を指定し、Ryzen の全コア・全スレッドをフル稼働させて高速書き出しを行います。

### 7.4 Pass 1（顔検出）の動画デコード超高速化（全環境共通）

4K 動画（3840×2160）をそのままデコードすると、プロセス間パイプに毎秒約 1.5GB の非圧縮データが流れ込み、デコード速度が約 2.4 FPS でボトルネックになっていました。
本ツールでは、顔検出モデルの入力サイズ（640×640）に合わせて **FFmpeg 内部でスケールダウン（`-vf scale=640:360`）した上でパイプ転送し、マルチスレッド（`-threads 0`）を適用** しています。
- パイプ転送量を 1/36 に削減し、Pass 1 の動画読み込み速度が **約 2.4 FPS → 約 60 FPS（約25倍）** に飛躍的に高速化されます。

### 7.5 Pass 2（モザイク描画・レンダリング）の 3ステージ非同期パイプライン高速化（全環境共通）


Pass 2（レンダリング）において CPU 使用率が 15% 前後で停滞していたボトルネックは、Windows の匿名パイプ（Anonymous Pipe）を経由して 4K 非圧縮フレーム（1フレーム約 25MB、60fps で毎秒 1.5GB）を同期転送する際に発生する OS レベルのコンテキストスイッチ待ちでした。

本ツールでは、**3ステージ並行非同期パイプライン** を導入することで、Pass 2 の書き出し速度を **約 2.5 FPS → 約 20〜30+ FPS（最大10倍以上）** に高速化しました：
1. **ステージ 1（バックグラウンド・デコーダー）**: `cv2.VideoCapture` によるプロセス内ハードウェア高速デコード（Direct3D 11 / Media Foundation）で、OS パイプを介さず直接フレームをプリフェッチキューへ先読み。
2. **ステージ 2（メインスレッド・モザイク演算）**: メモリ上でゼロコピー演算を行い、モザイク・ぼかしを高速描画（80+ FPS）。
3. **ステージ 3（バックグラウンド・エンコーダー）**: 非同期キュー経由で Intel Arc GPU のメディアエンジン（`hevc_qsv` / `h264_qsv`）へストリーミング転送し、ハードウェアエンコード。

### 7.6 色再現性（カラーマネジメント）の完全保証：色味変更の完全防止

iPhone 4K 10-bit HDR（BT.2020 / HLG）などの広色域・高ダイナミックレンジ映像において、**再エンコードによる色褪せ・変色・白飛びを 100% 防止** します：
- **10-bit 階調の維持 (`p010le`)**: 入力動画が 10-bit（`yuv420p10le`）の場合、Intel QSV / Apple VideoToolbox / AMD AMF / CPU のいずれにおいても自動的に 10-bit ピクセルフォーマット（`p010le` 等）を適用し、8-bit 圧縮によるバンディングや色階調の破綻を防ぎます。
- **HEVC VUI ビットストリーム注入 (`hevc_metadata`)**:
  コンテナタグだけでなく、HEVC ビットストリームの NAL ユニット内部（VUI: Video Usability Information）に直接 **ITU-T H.273 規格コード** を埋め込みます：
  - `colour_primaries=9` (BT.2020)
  - `transfer_characteristics=18` (HLG / ARIB STD-B67)
  - `matrix_coefficients=9` (BT.2020 non-constant luminance)
  - `color_range=tv` (Limited Range)
  これにより、YouTube や QuickTime Player、VLC 等のあらゆる再生環境で、元動画と完全に同一の発色・色域・コントラストが正確に再現されます。

---

## 8. 利用規約・免責事項 (Disclaimer & Terms of Use)

本ソフトウェアをご利用になる前に、以下の免責事項を必ずお読みください。本ソフトウェアを使用した時点で、本免責事項に同意したものとみなされます。

> [!IMPORTANT]
> **目視による最終確認の義務**
> 本ソフトウェアは、AI（コンピュータビジョン・機械学習モデル）を用いて顔検出およびモザイク／ぼかし処理を**自動化・補助するツール**です。
> カメラアングル、被写体の移動速度、遮蔽物、極端な照明変化、解像度等により、**顔の検出漏れやモザイクの途切れが発生する可能性を技術的に 100% 排除することはできません**。
>
> 処理後の動画をインターネット（YouTube、SNS等）へ公開・共有する前に、**利用者の責任において必ず全編の目視確認を行い、必要に応じて手動での追加修正を行ってください**。

1. **保証の否認 (Disclaimer of Warranty)**:
   本ソフトウェアは Apache License 2.0 に基づき「現状有姿 (AS IS)」で提供され、商品性、特定目的への適合性、権利非侵害、および検出の完全性を含め、明示的にも黙示的にもいかなる保証も行いません。
2. **責任の制限 (Limitation of Liability)**:
   本ソフトウェアの利用、誤動作、検出漏れ、または利用不能に起因して生じたいかなる損害（肖像権侵害、プライバシー侵害、名誉毀損、第三者との紛争、営業損害、データの消失、その他直接的・間接的・派生的損害を含む）について、開発者およびコントリビューターは理由の如何を問わず一切の法的責任を負いません。
3. **第三者権利侵害の責任**:
   動画を撮影・加工・公開・配信することに伴うすべての法的責任（肖像権、プライバシー権、著作権等の遵守）は、本ソフトウェアの利用者が単独で負うものとします。

---

## 9. ライセンス (License)

本プロジェクトは **[Apache License, Version 2.0](LICENSE)** の下で公開されています。
詳細はリポジトリ直下の `LICENSE` ファイルをご参照ください。

