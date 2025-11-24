from flask import Flask, render_template, request, redirect, session, url_for, flash
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import json
import re
import nltk
import os
from werkzeug.security import generate_password_hash, check_password_hash

# --- Pastikan stopwords Indonesia tersedia ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words = set(stopwords.words("indonesian"))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

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

# --- global token sets for recipes (used by chatbot quick replies)
resep_tokens_global = [set(clean_text(" ".join(r.get("bahan", []))).split()) for r in resep_data]

# --- Load atau buat data favorit ---
favorit_file = "data/favorit.json"
try:
    with open(favorit_file, "r", encoding="utf-8") as f:
        favorit_data = json.load(f)
        # migrate old entries that lack 'username'
        changed = False
        for item in favorit_data:
            if isinstance(item, dict) and "username" not in item:
                item.setdefault("username", "public")
                changed = True
        if changed:
            with open(favorit_file, "w", encoding="utf-8") as wf:
                json.dump(favorit_data, wf, ensure_ascii=False, indent=4)
except FileNotFoundError:
    favorit_data = []
    with open(favorit_file, "w", encoding="utf-8") as f:
        json.dump(favorit_data, f, ensure_ascii=False, indent=4)

# --- Users (simple file-based user store) ---
users_file = "data/users.json"
try:
    with open(users_file, "r", encoding="utf-8") as f:
        users = json.load(f)
except FileNotFoundError:
    users = []

# Create a default admin if no users exist (password: 'admin')
if not users:
    users.append({
        "username": "admin",
        "password": generate_password_hash("admin"),
        "role": "admin",
    })
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = next((u for u in users if u["username"] == username), None)
        if user and check_password_hash(user["password"], password):
            session["username"] = username
            session["role"] = user.get("role", "user")
            flash("Login berhasil.", "success")
            return redirect(url_for("index"))
        flash("Username atau kata sandi salah.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Isi username dan password.", "error")
            return render_template("register.html")
        if any(u["username"] == username for u in users):
            flash("Username sudah digunakan.", "error")
            return render_template("register.html")
        user = {
            "username": username,
            "password": generate_password_hash(password),
            "role": "user",
        }
        users.append(user)
        with open(users_file, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
        flash("Pendaftaran berhasil. Silakan login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/rekomendasi", methods=["POST"])
def rekomendasi():
    # Ambil input bahan dari user
    bahan_user = request.form["bahan"]
    # parse comma-separated user ingredients
    user_inputs = [b.strip().lower() for b in bahan_user.split(",") if b.strip()]

    # Allow forcing semantic fallback even when confidence is low
    force = bool(request.form.get("force"))

    # Build a set of all ingredient tokens in the corpus (normalized)
    def tokens_from_list(bahan_list):
        # join list into text then clean and split
        text = " ".join(bahan_list)
        return set(clean_text(text).split())

    all_tokens = set()
    resep_tokens = []
    for r in resep_data:
        toks = tokens_from_list(r.get("bahan", []))
        resep_tokens.append(toks)
        all_tokens.update(toks)

    # Check which user ingredients are present (by normalized token overlap) and suggest close matches
    missing = []
    suggestions = {}
    present = []
    # create token set for user inputs
    user_tokens = set()
    user_input_tokens = {}
    for ui in user_inputs:
        toks = set(clean_text(ui).split())
        user_input_tokens[ui] = toks
        user_tokens.update(toks)

    for ui, toks in user_input_tokens.items():
        if toks & all_tokens:
            present.append(ui)
        else:
            missing.append(ui)
            # suggestions based on close matches of raw ui against all_tokens joined strings
            close = get_close_matches(ui, [" ".join([t for t in all_tokens])], n=3, cutoff=0.6)
            # fallback: use close matches on the raw ingredient names (un-normalized) for better suggestions
            if not close:
                close = get_close_matches(ui, list({ing.lower() for r in resep_data for ing in r.get("bahan", [])}), n=3, cutoff=0.6)
            if close:
                suggestions[ui] = close

    # Compute exact-match score per recipe using normalized token overlap
    exact_scores = []
    for idx, toks in enumerate(resep_tokens):
        score = len(user_tokens & toks)
        exact_scores.append((idx, score))

    # If any recipe has a positive exact-match score, prefer those results
    exact_scores_sorted = sorted(exact_scores, key=lambda x: x[1], reverse=True)
    if exact_scores_sorted and exact_scores_sorted[0][1] > 0:
        # take top 3 by exact score; break ties by semantic similarity
        top_exact = [t for t in exact_scores_sorted if t[1] > 0]
        # compute semantic scores only for tied recipes if needed
        # prepare user embedding for tiebreakers
        bahan_bersih = clean_text(" ".join(user_inputs))
        user_embed = embedder.encode([bahan_bersih], normalize_embeddings=True)
        sim_scores = cosine_similarity(user_embed, resep_embeddings).flatten()

        # sort by (exact score, semantic score)
        top_exact_sorted = sorted(top_exact, key=lambda t: (t[1], sim_scores[t[0]]), reverse=True)
        top_idx = [t[0] for t in top_exact_sorted[:3]]
        hasil_resep = [resep_data[i] for i in top_idx]

        # Prepare messages
        pesan = None
        if missing:
            pesan = f"Beberapa bahan tidak ditemukan: {', '.join(missing)}."
        return render_template(
            "hasil.html",
            hasil=hasil_resep[:1],
            bahan=user_inputs,
            prediksi=hasil_resep[0]["nama"] if hasil_resep else None,
            rekomendasi=[r["nama"] for r in hasil_resep[1:]],
            pesan=pesan,
            missing=missing,
            suggestions=suggestions,
        )

    # Fallback: semantic similarity as before
    bahan_bersih = clean_text(" ".join(user_inputs))
    user_embed = embedder.encode([bahan_bersih], normalize_embeddings=True)
    sim_scores = cosine_similarity(user_embed, resep_embeddings).flatten()
    top_idx = sim_scores.argsort()[-3:][::-1]
    hasil_resep = [resep_data[i] for i in top_idx]
    skor_tertinggi = sim_scores[top_idx[0]]

    # If similarity is low and not forced, show recommendations but indicate uncertain and missing ingredients
    hasil_utama = hasil_resep[0]
    rekomendasi_menu = [r["nama"] for r in hasil_resep[1:]]

    if skor_tertinggi < 0.3 and not force:
        pesan = "🤔 AI belum yakin, tapi ini beberapa resep dengan bahan paling mirip:"
        if missing:
            pesan += f" Beberapa bahan tidak ditemukan: {', '.join(missing)}."
        return render_template(
            "hasil.html",
            hasil=[],
            bahan=user_inputs,
            prediksi=None,
            rekomendasi=[r["nama"] for r in hasil_resep],
            pesan=pesan,
            missing=missing,
            suggestions=suggestions,
            forced=False,
        )

    # If forced or similarity is high, show semantic results; if forced, mark forced=True so template can show a banner
    forced_flag = force
    pesan_forced = None
    if force and skor_tertinggi < 0.3:
        pesan_forced = "Menampilkan hasil meskipun confidence rendah (semantic fallback)."

    return render_template(
        "hasil.html",
        hasil=[hasil_utama],
        bahan=user_inputs,
        prediksi=hasil_utama["nama"],
        rekomendasi=rekomendasi_menu,
        pesan=pesan_forced,
        missing=missing,
        suggestions=suggestions,
        forced=forced_flag,
    )



@app.route('/chat', methods=['POST'])
def chat():
    # simple rule-based chatbot reply (JSON)
    data = request.get_json(force=True)
    msg = (data.get('message') or '').strip()
    if not msg:
        return json.dumps({'reply': 'Silakan tulis pertanyaan.'}), 200, {'Content-Type': 'application/json'}

    # normalize and compute token overlap with recipes
    msg_tokens = set(clean_text(msg).split())
    scores = []
    for idx, toks in enumerate(resep_tokens_global):
        scores.append((idx, len(msg_tokens & toks)))
    scores.sort(key=lambda x: x[1], reverse=True)

    # if any recipe shares tokens, suggest top 3 names
    if scores and scores[0][1] > 0:
        top = [resep_data[i]['nama'] for i, s in scores if s > 0][:3]
        reply = f"Saya menemukan resep yang mengandung bahan tersebut: {', '.join(top)}. Ketik nama resep untuk detail atau minta rekomendasi lain."
    else:
        reply = f"Hai — saya SmartchefAI. Saya belum terhubung ke model eksternal. Saya menerima: '{msg}'. Coba tanya nama bahan atau minta rekomendasi sederhana."

    return json.dumps({'reply': reply}), 200, {'Content-Type': 'application/json'}


@app.route("/simpan", methods=["POST"])
def simpan():
    if "username" not in session:
        flash("Silakan login untuk menyimpan favorit.", "error")
        return redirect(url_for("login"))
    bahan = request.form["bahan"]
    resep = request.form["resep"]
    username = session["username"]
    # prevent duplicate per-user
    existing = [f for f in favorit_data if f.get("nama") == resep and f.get("username") == username]
    if resep and not existing:
        favorit_data.append({"username": username, "nama": resep, "bahan": bahan})
        with open(favorit_file, "w", encoding="utf-8") as f:
            json.dump(favorit_data, f, ensure_ascii=False, indent=4)
    return redirect(url_for("favorit"))


@app.route("/favorit")
def favorit():
    if "username" not in session:
        flash("Silakan login untuk melihat favorit.", "error")
        return redirect(url_for("login"))

    username = session["username"]
    role = session.get("role", "user")
    # Admin sees all favorites; users see their own
    if role == "admin":
        visible = favorit_data
    else:
        visible = [f for f in favorit_data if f.get("username") == username]

    favorit_resep = []
    for fav in visible:
        for r in resep_data:
            if r["nama"] == fav["nama"]:
                item = r.copy()
                item["saved_by"] = fav.get("username")
                favorit_resep.append(item)
                break
    return render_template("favorit.html", favorit=favorit_resep)


@app.route("/admin")
def admin_panel():
    # admin-only view of users and favorites
    if "username" not in session or session.get("role") != "admin":
        flash("Akses ditolak: harus admin.", "error")
        return redirect(url_for("index"))

    # reload users & favorites from files to show latest
    try:
        with open(users_file, "r", encoding="utf-8") as f:
            current_users = json.load(f)
    except FileNotFoundError:
        current_users = []

    try:
        with open(favorit_file, "r", encoding="utf-8") as f:
            current_favs = json.load(f)
    except FileNotFoundError:
        current_favs = []

    return render_template("admin.html", users=current_users, favorites=current_favs)


@app.route("/admin/user_action", methods=["POST"])
def admin_user_action():
    if "username" not in session or session.get("role") != "admin":
        flash("Akses ditolak: harus admin.", "error")
        return redirect(url_for("index"))

    action = request.form.get("action")
    target = request.form.get("username")
    if not target or not action:
        flash("Parameter tidak lengkap.", "error")
        return redirect(url_for("admin_panel"))

    # load users
    with open(users_file, "r", encoding="utf-8") as f:
        current_users = json.load(f)

    if action == "promote":
        for u in current_users:
            if u.get("username") == target:
                u["role"] = "admin"
                flash(f"{target} di-promote menjadi admin.", "success")
                break
    elif action == "demote":
        for u in current_users:
            if u.get("username") == target:
                u["role"] = "user"
                flash(f"{target} dikembalikan menjadi user.", "success")
                break
    elif action == "delete":
        # remove user and their favorites
        current_users = [u for u in current_users if u.get("username") != target]
        with open(favorit_file, "r", encoding="utf-8") as f:
            favs = json.load(f)
        favs = [f for f in favs if f.get("username") != target]
        with open(favorit_file, "w", encoding="utf-8") as f:
            json.dump(favs, f, ensure_ascii=False, indent=4)
        flash(f"User {target} dan semua favoritnya dihapus.", "success")

    # save users back
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(current_users, f, ensure_ascii=False, indent=4)

    return redirect(url_for("admin_panel"))


@app.route("/hapus", methods=["POST"])
def hapus():
    if "username" not in session:
        flash("Silakan login untuk menghapus favorit.", "error")
        return redirect(url_for("login"))
    nama = request.form["nama"]
    username = session["username"]
    role = session.get("role", "user")
    global favorit_data
    if role == "admin":
        favorit_data = [f for f in favorit_data if f.get("nama") != nama]
    else:
        favorit_data = [f for f in favorit_data if not (f.get("nama") == nama and f.get("username") == username)]
    with open(favorit_file, "w", encoding="utf-8") as f:
        json.dump(favorit_data, f, ensure_ascii=False, indent=4)
    return redirect(url_for("favorit"))


if __name__ == "__main__":
    app.run(debug=True)
