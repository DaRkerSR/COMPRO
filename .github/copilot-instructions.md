# Copilot instructions for this repository

This repository is a small Flask app that recommends recipes based on input ingredients using sentence-transformers embeddings. Keep instructions concise and actionable — focus on the concrete patterns and files below so an AI agent can be productive immediately.

## Big picture
- **Service**: single-process Flask web app in `app.py`. Routes and behavior are implemented directly in that file.
- **Data**: recipes live in `data/resep.json`. Favorites are persisted to `data/favorit.json` (created if missing).
- **ML component**: uses `sentence-transformers` model `paraphrase-multilingual-MiniLM-L12-v2` to embed cleaned ingredient lists. Embeddings are computed at import/startup and stored in `resep_embeddings`.

## Key files and examples
- `app.py` — main entrypoint. Important routes:
  - `GET /` renders `templates/index.html` (input form)
  - `POST /rekomendasi` takes form field `bahan`, computes embedding, returns `templates/hasil.html`
  - `POST /simpan` saves a recipe to `data/favorit.json`
  - `GET /favorit` shows saved recipes using `templates/favorit.html`
  - `POST /hapus` removes a favorite
- `data/resep.json` — list of recipe objects with keys: `nama`, `bahan` (array), `langkah`, `gambar`. Example entry: `{ "nama": "Nasi Goreng", "bahan": ["nasi","telur"], ... }`.
- `templates/` — Jinja templates (`index.html`, `hasil.html`, `favorit.html`) use Indonesian copy and expect `gambar` either as an external URL or a path relative to `static/` (`images/...`). Templates call `url_for('static', filename=r.gambar)`.

## Observable patterns & conventions
- Text preprocessing: `clean_text()` in `app.py` lowercases, strips non-a-z characters and removes Indonesian stopwords (NLTK). This function is used for both recipe texts and user input before embedding.
- Embeddings: `embedder.encode(..., normalize_embeddings=True)` is used; cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) ranks recipes. Top-3 matches are returned; a score threshold of `0.3` is used to detect low-confidence matches.
- Persistence: favorites are simple JSON list objects written with `json.dump(..., ensure_ascii=False, indent=4)`.
- Startup cost: the model is loaded and all recipe embeddings are computed at import time — expect a noticeable delay and network download the first time the model is used.

## Developer workflows (how to run & debug)
1. Create / activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies (from project root or `resep_app`):
```powershell
cd resep_app
pip install -r requirements.txt
```
3. Run locally:
```powershell
python app.py
# app runs on http://127.0.0.1:5000/ by default (debug=True)
```
4. Quick functional test: open browser and submit ingredients like `nasi, telur` on the homepage form.

## What to watch for when editing
- Avoid moving embedding computation into request handlers without caching — current approach avoids recomputing embeddings for the recipe corpus on every request.
- Template image handling: `r.gambar` may be an external URL or a path under `static/`. Use `url_for('static', filename=...)` when referencing local static paths.
- Data paths are relative: running the app from `resep_app` makes `data/resep.json` accessible via the relative paths in `app.py`.

## Useful quick fixes an agent can make
- If startup time is an issue, persist `resep_embeddings` to disk and load them on startup instead of re-encoding every run (note: this repo currently does not persist embeddings).
- Validate `r.gambar` values and normalize local vs external images to avoid broken images in templates.

## Tests and CI
- There are no tests or CI configs in the repository. Keep changes small and test locally using the instructions above.

---
If any section is unclear or you want more detail (for example, a suggested change to persist embeddings or a sample unit test for `clean_text()`), tell me which part to expand.
