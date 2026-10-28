import os
import tempfile
import time
from pathlib import Path
import streamlit as st
import mlx_whisper

# Set page config
st.set_page_config(
    page_title="Voice2TZ — Транскрибация & ТЗ из голосовых",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/clean modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Настройки")

    model_choice = st.selectbox(
        "🧠 Модель Whisper",
        options=[
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/whisper-large-v3",
            "mlx-community/whisper-medium",
            "mlx-community/whisper-small"
        ],
        index=0,
        help="whisper-large-v3-turbo оптимизирована под Apple Silicon M3 Pro: работает молниеносно при максимальной точности."
    )

    language_choice = st.selectbox(
        "🌐 Язык аудио",
        options=["Автоопределение", "Русский (ru)", "Казахский (kk)", "Английский (en)"],
        index=1
    )

    lang_code = {
        "Автоопределение": None,
        "Русский (ru)": "ru",
        "Казахский (kk)": "kk",
        "Английский (en)": "en"
    }[language_choice]

    st.markdown("---")
    st.subheader("🤖 Настройки AI-анализа")

    ai_provider = st.selectbox(
        "Провайдер AI для ТЗ",
        options=["Локальный Ollama", "OpenAI (ChatGPT)", "Anthropic (Claude)", "Без LLM (Только расшифровка)"],
        index=0
    )

    ollama_model = "qwen2.5:latest"
    api_key = ""
    openai_model = "gpt-4o-mini"
    claude_model = "claude-3-5-sonnet-20241022"
    ollama_url = "http://localhost:11434"

    if ai_provider == "Локальный Ollama":
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        ollama_model = st.text_input("Модель Ollama", value="qwen2.5:latest", help="Например: qwen2.5, llama3.1, mistral, deepseek-r1")
    elif ai_provider == "OpenAI (ChatGPT)":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        openai_model = st.selectbox("Модель OpenAI", ["gpt-4o", "gpt-4o-mini"], index=1)
    elif ai_provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        claude_model = st.selectbox("Модель Claude", ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"], index=0)

    st.markdown("---")
    analysis_type = st.radio(
        "🎯 Формат итогового документа:",
        options=[
            "📋 Подробное ТЗ и план работ",
            "✅ Список задач и Action Items",
            "📌 Краткая выжимка (Саммари)",
            "📝 Только стенограмма"
        ],
        index=0
    )

# Main UI
st.markdown('<div class="main-header">🎙️ Voice2TZ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Локальная пакетная транскрибация голосовых сообщений и генерация ТЗ на Apple M3 Pro</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Закиньте сюда 1 или сразу 5–10 голосовых сообщений (.ogg, .opus, .m4a, .mp3, .wav, .mp4, .mov)",
    type=["ogg", "opus", "m4a", "mp3", "wav", "mp4", "mov", "aac", "webm", "flac"],
    accept_multiple_files=True,
    help="Вы можете выделить и перетащить сразу пачку файлов из Telegram, WhatsApp или диктофона"
)

if uploaded_files:
    st.info(f"📁 Загружено файлов: **{len(uploaded_files)}**")

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        start_btn = st.button("🚀 Расшифровать и составить ТЗ", type="primary", use_container_width=True)

    if start_btn:
        transcripts = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        # Step 1: Transcribe each file
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"⏳ Расшифровка ({idx + 1}/{len(uploaded_files)}): {uploaded_file.name}...")

            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                kwargs = {"path_or_hf_repo": model_choice}
                if lang_code:
                    kwargs["language"] = lang_code

                result = mlx_whisper.transcribe(tmp_path, **kwargs)
                text = result.get("text", "").strip()

                transcripts.append({
                    "filename": uploaded_file.name,
                    "text": text
                })
            except Exception as e:
                transcripts.append({
                    "filename": uploaded_file.name,
                    "text": f"⚠️ Ошибка расшифровки: {e}"
                })
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        elapsed = time.time() - start_time
        status_text.success(f"✅ Расшифровано {len(uploaded_files)} файлов за {elapsed:.1f} сек на Metal GPU!")

        # Combine full transcript
        full_transcript_text = ""
        for i, item in enumerate(transcripts, 1):
            full_transcript_text += f"### Голосовое #{i} (`{item['filename']}`):\n{item['text']}\n\n"

        # Step 2: AI Analysis / TZ Generation
        final_doc = ""

        prompt_instructions = {
            "📋 Подробное ТЗ и план работ": """
Ты — ведущий системный аналитик и проджект-менеджер.
На основе расшифровки голосовых сообщений составь качественное, подробное Техническое Задание (ТЗ) и план реализации проекта.

Структура документа:
# 📋 Техническое Задание (ТЗ) и План реализации

## 1. 📌 Краткое резюме проекта (Executive Summary)
- Суть проекта и основная цель в 2-3 предложениях.

## 2. 🎯 Цели и ключевые вводные требования
- Основные требования и пожелания заказчика/спикера.

## 3. ⚙️ Функциональные требования
- Детальный список функций и модулей системы.

## 4. 📋 План работ и этапы реализации
- Этап 1: Подготовка и проектирование
- Этап 2: Разработка и интеграция
- Этап 3: Тестирование и сдача

## 5. ⚠️ Риски, открытые вопросы и блокеры
- Вопросы, требующие уточнения у заказчика.
""",
            "✅ Список задач и Action Items": """
Ты — персональный ассистент и технический координатор.
Извлеки из расшифровки голосовых сообщений чёткий список задач и договоренностей.

Структура:
# ✅ Список задач и договорённостей

## 📌 Краткая суть обсуждения
## 🎯 Список конкретных задач (Action Items) с чек-боксами
## 🤝 Договорённости и обязательства сторон
## ❓ Вопросы на уточнение
""",
            "📌 Краткая выжимка (Саммари)": """
Составь максимально лаконичную и точную выжимку (Summary) из переданных голосовых сообщений. Выдели главное, убери воду.
""",
            "📝 Только стенограмма": None
        }[analysis_type]

        if prompt_instructions and ai_provider != "Без LLM (Только расшифровка)":
            with st.spinner("🤖 Генерация аналитики и ТЗ через AI..."):
                try:
                    if ai_provider == "Локальный Ollama":
                        import requests
                        res = requests.post(
                            f"{ollama_url.rstrip('/')}/api/generate",
                            json={
                                "model": ollama_model,
                                "prompt": f"{prompt_instructions}\n\nИсходная стенограмма сообщений:\n{full_transcript_text}",
                                "stream": False
                            },
                            timeout=120
                        )
                        if res.status_code == 200:
                            final_doc = res.json().get("response", "")
                        else:
                            st.warning(f"Ollama вернул ошибку ({res.status_code}). Убедитесь, что Ollama запущен (`ollama serve`).")

                    elif ai_provider == "OpenAI (ChatGPT)":
                        import openai
                        client = openai.OpenAI(api_key=api_key)
                        resp = client.chat.completions.create(
                            model=openai_model,
                            messages=[
                                {"role": "system", "content": prompt_instructions},
                                {"role": "user", "content": f"Вот стенограмма голосовых сообщений:\n\n{full_transcript_text}"}
                            ]
                        )
                        final_doc = resp.choices[0].message.content

                    elif ai_provider == "Anthropic (Claude)":
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)
                        resp = client.messages.create(
                            model=claude_model,
                            max_tokens=4000,
                            system=prompt_instructions,
                            messages=[
                                {"role": "user", "content": f"Вот стенограмма голосовых сообщений:\n\n{full_transcript_text}"}
                            ]
                        )
                        final_doc = resp.content[0].text
                except Exception as e:
                    st.error(f"Ошибка при обращении к AI-провайдеру: {e}")

        # Assemble tabs
        st.markdown("---")
        tab1, tab2 = st.tabs(["📄 Результат / ТЗ", "📝 Исходные стенограммы файлов"])

        with tab1:
            if final_doc:
                st.markdown(final_doc)
                export_text = f"{final_doc}\n\n---\n\n## 📝 Полная расшифровка исходных сообщений\n\n{full_transcript_text}"
            else:
                st.markdown(full_transcript_text)
                export_text = full_transcript_text

            st.download_button(
                label="💾 Скачать результат (.md)",
                data=export_text,
                file_name="ТЗ_из_голосовых.md",
                mime="text/markdown"
            )

        with tab2:
            for item in transcripts:
                with st.expander(f"🎵 {item['filename']}", expanded=True):
                    st.write(item['text'])

else:
    st.info("💡 Перетащите файлы голосовых сообщений в область загрузки выше, чтобы начать.")
