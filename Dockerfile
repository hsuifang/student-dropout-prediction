# 學生退學預測 — Streamlit 推論介面
# 行為等同本機：打包訓練好的 models/model.pt（含 EnsembleMLP）+ preprocessor.joblib。
FROM python:3.12-slim

WORKDIR /app

# 更新底層 OS 套件，清掉基底映像可修補的 CVE
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# 先裝 CPU-only torch（伺服器多為 CPU 推論；避免抓進數 GB 的 CUDA 函式庫）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 再裝其餘相依（torch 已滿足，不會重裝 CUDA 版）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案。若 build context 的 models/ 已含 export_model 產出的真實模型，會一併打包。
# （原始/處理後的學生資料由 .dockerignore 排除，不會進映像）
COPY . .

# 確保映像內有可用模型：
#   - models/model.pt 已存在（你先跑過 export_model）→ 直接用真實 ensemble
#   - 否則 fallback 烤一個 placeholder，讓映像仍能啟動 demo
RUN [ -f models/model.pt ] || python -m scripts.train_placeholder

# inference log 目錄設為可寫（HF Spaces 等以非 root 執行時也能寫入）
RUN mkdir -p results && chmod -R 777 results

EXPOSE 8501

# 伺服器對外服務需綁 0.0.0.0；headless 避免容器內嘗試開瀏覽器
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
