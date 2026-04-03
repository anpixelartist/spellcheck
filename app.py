import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st
import time

from spell_checker import BigramSpellChecker
import corpus_downloader


st.set_page_config(
    page_title="Context-Aware Spell Checker",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body, p, span, div, label, button, input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    h1, h2, h3 {
        letter-spacing: -0.01em;
    }

    .stTextArea textarea {
        font-size: 0.95rem;
        line-height: 1.75;
        border-radius: 10px;
        border: 1.5px solid #e2e4e9;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e8eaef;
        border-radius: 10px;
        padding: 0.85rem 0.75rem;
        text-align: center;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.3;
        word-break: break-all;
    }

    .stat-label {
        font-size: 0.65rem;
        font-weight: 500;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
        line-height: 1.2;
    }

    .correction-badge {
        display: inline-block;
        background: #eef0ff;
        color: #4F6DF5;
        padding: 1px 8px;
        border-radius: 5px;
        font-weight: 500;
        border-bottom: 2px solid #4F6DF5;
        cursor: default;
        position: relative;
        white-space: nowrap;
        vertical-align: baseline;
    }

    .correction-badge .badge-tooltip {
        display: none;
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #1a1a2e;
        color: #fff;
        padding: 5px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 400;
        white-space: nowrap;
        z-index: 9999;
        pointer-events: none;
    }

    .correction-badge .badge-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 5px solid transparent;
        border-top-color: #1a1a2e;
    }

    .correction-badge:hover .badge-tooltip {
        display: block;
    }

    .inline-text {
        font-size: 1rem;
        line-height: 1.85;
        color: #2d2d44;
        background: #fafbfd;
        border: 1px solid #e8eaef;
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        hyphens: auto;
    }

    .confidence-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
        flex-wrap: wrap;
    }

    .confidence-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #4a4a6a;
        min-width: 90px;
        flex-shrink: 0;
        white-space: nowrap;
    }

    .confidence-track {
        flex: 1;
        height: 7px;
        background: #eef0f5;
        border-radius: 4px;
        overflow: hidden;
        min-width: 60px;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
    }

    .confidence-val {
        font-size: 0.78rem;
        font-weight: 600;
        min-width: 45px;
        text-align: right;
        flex-shrink: 0;
        white-space: nowrap;
    }

    .summary-banner {
        background: linear-gradient(135deg, #f0f2ff 0%, #e8f5e9 100%);
        border: 1px solid #d4d9f5;
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1.25rem;
    }

    .summary-banner-text {
        font-size: 0.9rem;
        font-weight: 500;
        color: #2d2d44;
        line-height: 1.5;
    }

    .summary-banner-count {
        font-weight: 700;
        color: #4F6DF5;
    }

    .diff-box {
        font-size: 0.9rem;
        line-height: 1.8;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        border: 1px solid #e8eaef;
        background: #fafbfd;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .diff-original .correction-badge {
        background: #fff3e0;
        color: #e65100;
        border-bottom-color: #e65100;
    }

    .diff-corrected .correction-badge {
        background: #e8f5e9;
        color: #2e7d32;
        border-bottom-color: #2e7d32;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: #4F6DF5;
        border: none;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #3b57d9;
        box-shadow: 0 4px 12px rgba(79, 109, 245, 0.25);
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e8eaef;
        border-radius: 10px;
    }

    .correction-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
        table-layout: fixed;
    }

    .correction-table th {
        background: #f5f6fa;
        color: #4a4a6a;
        font-weight: 600;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 9px 12px;
        border-bottom: 1px solid #e8eaef;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .correction-table th:first-child { border-radius: 8px 0 0 0; width: 40px; }
    .correction-table th:nth-child(2) { width: 90px; }
    .correction-table th:nth-child(3) { width: 90px; }
    .correction-table th:nth-child(4) { width: 80px; }
    .correction-table th:nth-child(5) { width: 150px; }
    .correction-table th:last-child { border-radius: 0 8px 0 0; }

    .correction-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #f0f1f5;
        color: #2d2d44;
        word-wrap: break-word;
        overflow-wrap: break-word;
        vertical-align: middle;
    }

    .correction-table tr:last-child td { border-bottom: none; }
    .correction-table tr:hover td { background: #fafbfd; }

    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #8b8fa3;
    }

    .empty-state-icon {
        font-size: 2.25rem;
        margin-bottom: 0.65rem;
        opacity: 0.45;
        line-height: 1;
    }

    .empty-state-text {
        font-size: 0.88rem;
        font-weight: 400;
        line-height: 1.6;
    }

    .sidebar .block-container {
        padding-top: 1.25rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e8eaef;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


EXAMPLE_SENTENCES = [
    "I am going to the see tomorrow morning.",
    "She has a blue car and a blew dress.",
    "The weather is fare today for a walk in the park.",
    "He rode his bike down the rode very fast.",
    "I need to buy some meet for dinner tonight.",
    "It was a grate idea to visit the museum.",
    "The principle of the school gave a long speech.",
    "I can not brake the record without practice.",
    "Their going to the store to by some bread.",
    "The knight rode his horse through the night.",
]


def init_session():
    if "checker" not in st.session_state:
        st.session_state.checker = None
    if "corrections" not in st.session_state:
        st.session_state.corrections = None
    if "correction_history" not in st.session_state:
        st.session_state.correction_history = []
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False


init_session()


def render_sidebar():
    with st.sidebar:
        st.markdown("### ✦ Spell Checker")
        st.markdown("")

        corpus_stats = corpus_downloader.get_corpus_stats()

        st.markdown("### Model Stats")

        checker = st.session_state.checker
        trained = st.session_state.model_trained

        if checker and trained:
            vocab_display = f"{len(checker.vocab):,}"
            bigrams_display = f"{checker.total_bigrams:,}"
        else:
            vocab_display = "—"
            bigrams_display = "—"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{vocab_display}</div><div class="stat-label">Vocabulary</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{bigrams_display}</div><div class="stat-label">Bigrams</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown("### Corpus")

        if corpus_stats["exists"]:
            st.success(f"Ready ({corpus_stats['size_mb']} MB)")
        else:
            st.warning("No corpus downloaded")

        if st.button("Download Corpus", type="primary", use_container_width=True):
            with st.spinner("Downloading corpus from Project Gutenberg..."):
                try:
                    corpus_path = corpus_downloader.download_corpus(
                        lambda msg: st.toast(msg, icon="📥")
                    )
                    st.session_state.checker = BigramSpellChecker(corpus_path)
                    st.session_state.model_trained = True
                    st.success("Corpus loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        if corpus_stats["exists"]:
            if st.button("Delete Cached Corpus", use_container_width=True):
                corpus_downloader.delete_corpus()
                st.session_state.checker = None
                st.session_state.model_trained = False
                st.session_state.corrections = None
                st.success("Corpus deleted.")
                st.rerun()

        st.markdown("")
        st.markdown("### Settings")

        max_distance = st.slider(
            "Edit Distance",
            min_value=1,
            max_value=2,
            value=2,
            step=1,
            help="Maximum Levenshtein distance for candidate generation",
        )

        threshold = st.slider(
            "Probability Threshold",
            min_value=0.00001,
            max_value=0.01,
            value=0.0001,
            step=0.00001,
            format="%.5f",
            help="Bigram probability threshold below which a word is flagged",
        )

        st.session_state.max_distance = max_distance
        st.session_state.threshold = threshold

        if st.session_state.correction_history:
            st.markdown("")
            st.markdown("### Recent Corrections")
            for item in reversed(st.session_state.correction_history[-5:]):
                with st.expander(f"{item['time']} — {item['stats']['errors_found']} errors", expanded=False):
                    st.caption(item['input'][:80] + ("..." if len(item['input']) > 80 else ""))


def render_header():
    st.markdown("### ✦ Context-Aware Spell Checker")
    st.caption("Bigram N-gram Model  ·  Levenshtein Distance  ·  Context-Aware Corrections")
    st.markdown("")


def render_input():
    st.markdown("### Enter Text")

    input_text = st.text_area(
        label="Paste or type your text here",
        label_visibility="collapsed",
        placeholder="Type or paste a sentence, paragraph, or longer text here...\n\nExample: I am going to the see tomorrow. The weather is fare today.",
        height=180,
    )

    if input_text:
        words = len(input_text.split())
        paragraphs = len([p for p in input_text.split("\n\n") if p.strip()])
        st.caption(f"{words} words  ·  {paragraphs} paragraph{'s' if paragraphs != 1 else ''}")

    st.markdown("")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])

    with col_btn1:
        correct_clicked = st.button("✦ Correct", type="primary", use_container_width=True)

    with col_btn2:
        if st.button("Clear", use_container_width=True):
            st.session_state.corrections = None
            st.rerun()

    st.markdown("")

    st.markdown("##### Try an example")
    example_cols = st.columns(5)
    for i, col in enumerate(example_cols):
        with col:
            example = EXAMPLE_SENTENCES[i]
            short_label = example[:22] + "..." if len(example) > 22 else example
            if st.button(short_label, key=f"ex_{i}", use_container_width=True):
                st.session_state._example_text = example
                st.rerun()

    if correct_clicked and input_text:
        if not st.session_state.model_trained or not st.session_state.checker:
            st.error("Please download a corpus first in the sidebar.")
            return None

        with st.spinner("Analyzing text..."):
            result = st.session_state.checker.correct_text(
                input_text,
                threshold=st.session_state.threshold,
                max_distance=st.session_state.max_distance,
            )

            st.session_state.corrections = result
            st.session_state.correction_history.append({
                "time": time.strftime("%H:%M:%S"),
                "input": input_text[:100],
                "stats": result["stats"],
            })

            return result

    if hasattr(st.session_state, "_example_text"):
        st.session_state.corrections = None
        st.rerun()

    return None


def get_confidence_color(confidence):
    if confidence >= 80:
        return "#22C55E"
    elif confidence >= 50:
        return "#4F6DF5"
    elif confidence >= 30:
        return "#F59E0B"
    else:
        return "#EF4444"


def render_summary_banner(result):
    stats = result["stats"]
    errors = stats["errors_found"]

    if errors == 0:
        st.markdown(
            """
            <div class="summary-banner" style="background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); border-color: #c8e6c9;">
                <div class="summary-banner-text">
                    ✓ <span class="summary-banner-count" style="color: #2e7d32;">No errors detected</span> — your text looks great!
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="summary-banner">
                <div class="summary-banner-text">
                    ✦ <span class="summary-banner-count">{errors} correction{'s' if errors != 1 else ''}</span> found across {stats['total_sentences']} sentences in {stats['total_paragraphs']} paragraph{'s' if stats['total_paragraphs'] != 1 else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_inline_view(result):
    corrections = result["corrections"]
    original_text = result["original_text"]

    if not corrections:
        st.markdown("### Corrected Text")
        st.markdown(
            f'<div class="inline-text">{original_text}</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown("### Corrected Text")

    correction_map = {}
    for i, corr in enumerate(corrections):
        correction_map[corr["original"]] = {
            "correction": corr["correction"],
            "confidence": corr["confidence"],
            "prob_before": corr["prob_before"],
            "prob_after": corr["prob_after"],
        }

    paragraphs = result["corrected_text"].split("\n\n")
    html_parts = []

    for para in paragraphs:
        words = para.split()
        rendered_words = []

        for word in words:
            clean_word = word.lower().strip(".,!?;:'\"()[]{}")
            if clean_word in correction_map:
                info = correction_map[clean_word]
                color = get_confidence_color(info["confidence"])
                tooltip = (
                    f"was: {info['correction']} &middot; confidence: {info['confidence']}% &middot; "
                    f"prob: {info['prob_before']:.6f} → {info['prob_after']:.6f}"
                )
                rendered_words.append(
                    f'<span class="correction-badge" style="border-bottom-color:{color};color:{color};">'
                    f'{word}<span class="badge-tooltip">{tooltip}</span></span>'
                )
            else:
                rendered_words.append(word)

        html_parts.append(" ".join(rendered_words))

    full_html = "<br><br>".join(html_parts)
    st.markdown(f'<div class="inline-text">{full_html}</div>', unsafe_allow_html=True)


def render_corrections_table(result):
    corrections = result["corrections"]
    if not corrections:
        return

    st.markdown("### Correction Details")

    rows = ""
    for i, corr in enumerate(corrections, 1):
        color = get_confidence_color(corr["confidence"])
        rows += f"""
        <tr>
            <td>{i}</td>
            <td style="color:#8b8fa3;text-decoration:line-through;">{corr['original']}</td>
            <td style="font-weight:600;color:{color};">{corr['correction']}</td>
            <td>
                <span style="display:inline-block;background:{color}15;color:{color};padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.78rem;">
                    {corr['confidence']}%
                </span>
            </td>
            <td style="font-size:0.73rem;color:#8b8fa3;">
                {corr['prob_before']:.6f} → {corr['prob_after']:.6f}
            </td>
            <td style="font-size:0.73rem;color:#4a4a6a;word-wrap:break-word;overflow-wrap:break-word;">
                ...{corr['context']}...
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="correction-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Original</th>
                    <th>Correction</th>
                    <th>Confidence</th>
                    <th>Probability</th>
                    <th>Context</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_chart(result):
    corrections = result["corrections"]
    if not corrections:
        return

    st.markdown("### Confidence Scores")

    chart_html = ""
    for corr in corrections:
        confidence = corr["confidence"]
        color = get_confidence_color(confidence)
        width = min(confidence, 100)
        chart_html += f"""
        <div class="confidence-row">
            <span class="confidence-label">{corr['original']} → {corr['correction']}</span>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{width}%;background:{color};"></div>
            </div>
            <span class="confidence-val" style="color:{color};">{confidence}%</span>
        </div>
        """

    st.markdown(chart_html, unsafe_allow_html=True)


def render_diff_view(result):
    corrections = result["corrections"]
    if not corrections:
        return

    original_text = result["original_text"]
    corrected_text = result["corrected_text"]

    correction_set_original = {corr["original"] for corr in corrections}
    correction_set_corrected = {corr["correction"] for corr in corrections}

    def highlight_text(text, highlight_words):
        words = text.split()
        rendered = []
        for word in words:
            clean = word.lower().strip(".,!?;:'\"()[]{}")
            if clean in highlight_words:
                rendered.append(f'<span class="correction-badge">{word}</span>')
            else:
                rendered.append(word)
        return " ".join(rendered)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**")
        st.markdown(
            f'<div class="diff-box diff-original">{highlight_text(original_text, correction_set_original)}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Corrected**")
        st.markdown(
            f'<div class="diff-box diff-corrected">{highlight_text(corrected_text, correction_set_corrected)}</div>',
            unsafe_allow_html=True,
        )


def render_export(result):
    st.download_button(
        label="↓ Download Corrected Text",
        data=result["corrected_text"],
        file_name="corrected_text.txt",
        mime="text/plain",
    )


def main():
    render_sidebar()
    render_header()

    if not st.session_state.model_trained:
        st.info("👈 Download a corpus from the sidebar to get started.")
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">✦</div>
                <div class="empty-state-text">
                    Download a corpus to enable context-aware spell checking.<br>
                    The model uses bigram frequencies and Levenshtein distance<br>
                    to detect and correct real-word errors.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = render_input()

    if result:
        st.markdown("")
        st.divider()
        st.markdown("")

        render_summary_banner(result)
        st.markdown("")

        render_inline_view(result)
        st.markdown("")

        render_corrections_table(result)
        st.markdown("")

        render_confidence_chart(result)
        st.markdown("")

        with st.expander("📋 Side-by-Side Comparison", expanded=False):
            render_diff_view(result)

        st.markdown("")
        render_export(result)


if __name__ == "__main__":
    main()
