from flask import Flask, render_template, request, redirect
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import json
import re
import nltk

# --- Pastikan stopwords Indonesia tersedia ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words = set(stopwords.words("indonesian"))

app = Flask(__name__)

# --- PREPROCESSING TEKS ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- MODEL EMBEDDING MULTIBAHASA ---
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Load data resep ---
with open("data/resep.json", "r", encoding="utf-8") as f:
    resep_data = json.load(f)

# Siapkan embedding semua bahan resep
resep_bahan_texts = [" ".join(r["bahan"]) for r in resep_data]
resep_bahan_clean = [clean_text(t) for t in resep_bahan_texts]
resep_embeddings = embedder.encode(resep_bahan_clean, normalize_embeddings=True)

# --- Load atau buat data favorit ---
favorit_file = "data/favorit.json"
try:
    with open(favorit_file, "r", encoding="utf-8") as f:
        favorit_data = json.load(f)
except FileNotFoundError:
    favorit_data = []
    with open(favorit_file, "w", encoding="utf-8") as f:
        json.dump(favorit_data, f, ensure_ascii=False, indent=4)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rekomendasi", methods=["POST"])
def rekomendasi():
    # Ambil input bahan dari user
    bahan_user = request.form["bahan"]
    bahan_bersih = clean_text(bahan_user)

    # Buat embedding untuk input user
    user_embed = embedder.encode([bahan_bersih], normalize_embeddings=True)

    # Hitung similarity antar bahan user vs semua resep
    sim_scores = cosine_similarity(user_embed, resep_embeddings).flatten()
    top_idx = sim_scores.argsort()[-3:][::-1]

    # Ambil 3 resep paling mirip
    hasil_resep = [resep_data[i] for i in top_idx]
    skor_tertinggi = sim_scores[top_idx[0]]

    # Kalau similarity rendah (AI ragu)
    if skor_tertinggi < 0.3:
        return render_template(
            "hasil.html",
            hasil=[],
            bahan=bahan_user.split(","),
            prediksi=None,
            rekomendasi=[r["nama"] for r in hasil_resep],
            pesan="🤔 AI belum yakin, tapi ini beberapa resep dengan bahan paling mirip:"
        )

    # Kalau similarity cukup tinggi
    hasil_utama = hasil_resep[0]
    rekomendasi_menu = [r["nama"] for r in hasil_resep[1:]]

    return render_template(
        "hasil.html",
        hasil=[hasil_utama],
        bahan=bahan_user.split(","),
        prediksi=hasil_utama["nama"],
        rekomendasi=rekomendasi_menu,
        pesan=None
    )


@app.route("/simpan", methods=["POST"])
def simpan():
    bahan = request.form["bahan"]
    resep = request.form["resep"]
    if resep and resep not in [f["nama"] for f in favorit_data]:
        favorit_data.append({"nama": resep, "bahan": bahan})
        with open(favorit_file, "w", encoding="utf-8") as f:
            json.dump(favorit_data, f, ensure_ascii=False, indent=4)
    return redirect("/favorit")


@app.route("/favorit")
def favorit():
    favorit_resep = []
    for fav in favorit_data:
        for r in resep_data:
            if r["nama"] == fav["nama"]:
                favorit_resep.append(r)
                break
    return render_template("favorit.html", favorit=favorit_resep)


@app.route("/hapus", methods=["POST"])
def hapus():
    nama = request.form["nama"]
    global favorit_data
    favorit_data = [f for f in favorit_data if f["nama"] != nama]
    with open(favorit_file, "w", encoding="utf-8") as f:
        json.dump(favorit_data, f, ensure_ascii=False, indent=4)
    return redirect("/favorit")


if __name__ == "__main__":
    app.run(debug=True)
