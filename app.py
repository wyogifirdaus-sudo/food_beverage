# ============================================================
# FOODS AND BEVERAGES ADVISOR
# Chatbot Perbandingan Produk untuk Tim Sales & Marketing
# Powered by LangChain + Groq + FAISS
# ============================================================
#
# CARA MENJALANKAN:
#   streamlit run app.py
#
# ============================================================

import streamlit as st
from rag_pipeline import build_rag_pipeline

# ── Konfigurasi Halaman ────────────────────────────────────────────────
st.set_page_config(
    page_title="Foods and Bevarages Advisor",
    page_icon="🏬",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────
st.title("🏬 Foods and Bevarages Advisor")
st.caption(
    "Asisten AI untuk tim sales & marketing — "
    "rekomendasi menu yang dapat dipilih sesuai katalog resmi"
)

# ── Load RAG Pipeline ──────────────────────────────────────────────────
# Menggunakan st.cache_resource agar pipeline hanya dibangun sekali.
# Tanpa ini, pipeline akan dibangun ulang setiap ada interaksi pengguna.
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_rag_pipeline()

# Tampilkan proses loading kepada pengguna
if "pipeline_loaded" not in st.session_state:
    with st.status("Memuat sistem AI...", expanded=True) as status:
        st.write("Membaca katalog produk...")
        st.write("Membangun vector store...")
        st.write("Menginisialisasi model bahasa...")
        chain, num_chunks = load_pipeline()
        st.session_state.chain = chain
        st.session_state.num_chunks = num_chunks
        st.session_state.pipeline_loaded = True
        status.update(
            label=f"Sistem siap! {num_chunks} potongan dokumen berhasil diindeks.",
            state="complete"
        )

chain = st.session_state.chain

# ── Inisialisasi Riwayat Chat ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Tampilkan Contoh Pertanyaan (hanya saat belum ada chat) ───────────
if not st.session_state.messages:
    st.info(
        "**Contoh pertanyaan yang bisa Anda ajukan:**\n\n"
        "- Rekomendasikan main menu\n"
        "- Rekomendasikan vegetarian menu\n"
        "- Menu apa yang cocok untuk penyuka seafood?\n"
        "- Apa menu yang paling murah di Main Menu?\n"
        "- Menu apa yang paling cocok untuk penyuka mie?\n"
    )

# ── Tampilkan Riwayat Chat ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Input Pengguna ─────────────────────────────────────────────────────
if user_input := st.chat_input("Tanyakan sesuatu tentang menu..."):

    # Simpan dan tampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate jawaban dari RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi di katalog..."):
            result = chain.invoke({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

        st.markdown(answer)

        # Tampilkan referensi dokumen sumber (bisa di-collapse)
        with st.expander("Lihat referensi dari katalog"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Referensi {i}:**")
                st.text(doc.page_content[:300] + "...")
                st.divider()

    # Simpan jawaban ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini menggunakan teknologi **RAG** "
        "_(Retrieval-Augmented Generation)_ untuk menjawab "
        "pertanyaan berdasarkan katalog produk resmi.\n\n"
        "Jawaban didasarkan **hanya** pada dokumen katalog, "
        "bukan pengetahuan umum AI."
    )

    st.divider()

    st.subheader("📜 Produk Tersedia")
    st.markdown(
        "Main Menu\n"
        "1. Nasi Goreng Spesial\n"
        "2. Ayam Bakar Madu\n"
        "3. Ikan Nila Goreng\n"
        "4. Sate Ayam (10 tusuk)\n"
        "5. Sate Kambing (10 tusuk)\n"
        "6. Capcay Goreng\n"
        "7. Gurame Asam Manis\n"
        "8. Mie Goreng Jawa\n"
        "Vegetarian Menu\n"
        "1. Nasi Goreng Vegetarian\n"
        "2. Tumis Kangkung Tahu\n"
        "Drinks\n"
        "1. Es Teh Manis - Teh hitam manis dengan es batu - Rp8.000\n"
        "2. Teh Hangat - Teh hitam hangat, gula tersedia terpisah - Rp7.000\n"
        "3. Jus Jeruk Segar - Jeruk peras segar, tanpa gula tambahan - Rp18.000\n"
        "4. Es Campur Segar - Campuran buah, cincau, susu, sirup merah - Rp20.000\n"
        "5. Air Mineral - Air mineral kemasan 600ml - Rp5.000\n"
        "6. Smoothie Mangga - Mangga, yogurt, madu, es batu - Rp28.000\n"
        "7. Milkshake Cokelat - Susu, es krim cokelat, sirup cokelat - Rp30.000\n"
        "8. Es Krim cup - 1  scoop es krim vanilla/cokelat - Rp10.000\n"
        "9. Susu Cokelat Hangat - Susu cokelat, disajikan hangat - Rp15.000\n"
        "Value Pack\n"
        "1. Paket Duo - 2 Nasi Goreng Spesial + 2 Es Teh Manis - Rp70.000\n"
        "2. Paket Keluarga - 4 Nasi Goreng Spesial + 2 Ayam Bakar Madu + 4 Es Teh Manis + 2 Jus Jeruk - Rp200.000\n"
        "3. Paket Vegetarian - 2 Nasi Goreng Vegetarian + 2 Teh Hangat - Rp67.000"
    )

    st.divider()

    st.subheader("⚙️ Arsitektur Sistem")
    st.markdown(
        "```\n"
        "Katalog Produk (TXT)\n"
        "       ↓\n"
        "  Document Loader\n"
        "       ↓\n"
        "  Text Splitter\n"
        "       ↓\n"
        "HuggingFace Embeddings\n"
        "       ↓\n"
        "  FAISS Vector Store\n"
        "       ↓\n"
        "    Retriever\n"
        "       ↓\n"
        " Groq LLM (Llama 3.3)\n"
        "       ↓\n"
        "  Jawaban Final\n"
        "```"
    )

    st.divider()

    if st.button("🔄 Reset Percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
