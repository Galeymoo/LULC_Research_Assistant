
import html
import re
import streamlit as st

from config import FINAL_TOP_K
from retrieval import ResearchRetriever
from generation import EvidenceAnswerGenerator


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="LULC Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CSS - CLEAN LIGHT THEME
# ==================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #FFFFFF !important;
    color: #202824 !important;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Typography */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

h1, h2, h3 {
    color: #202824 !important;
    font-weight: 650 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F0F3F1 !important;
    border-right: 1px solid #D8DFDA !important;
    opacity: 1 !important;
    z-index: 1000 !important;
}

section[data-testid="stSidebar"] > div {
    background-color: #F0F3F1 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-color: #F0F3F1 !important;
    opacity: 1 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #202824 !important;
}

/* Header */
.main-header {
    background-color: #F3F6F4;
    border: 1px solid #DCE3DE;
    border-radius: 12px;
    padding: 26px 28px;
    margin-bottom: 28px;
}

.main-header h1 {
    font-size: 29px;
    margin: 0 0 9px 0;
    color: #20362C !important;
}

.main-header p {
    color: #526159;
    font-size: 14px;
    line-height: 1.7;
    margin: 0;
}

/* Search input */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #202824 !important;
    border: 1px solid #CBD5CE !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 7px !important;
    font-weight: 600 !important;
    min-height: 42px;
}

/* Answer */
.answer-container {
    background-color: #F7F9F7;
    border: 1px solid #DCE3DE;
    border-radius: 10px;
    padding: 22px;
    margin-top: 12px;
    margin-bottom: 22px;
}

.answer-label {
    color: #3B6B5A;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Information box */
.info-box {
    background-color: #F3F6F4;
    border: 1px solid #DCE3DE;
    border-radius: 9px;
    padding: 18px;
    margin-top: 10px;
}

.info-title {
    color: #20362C;
    font-size: 15px;
    font-weight: 650;
    margin-bottom: 7px;
}

.info-description {
    color: #526159;
    font-size: 14px;
    line-height: 1.7;
}

/* Evidence */
div[data-testid="stExpander"] {
    border-radius: 8px;
    border: 1px solid #DCE3DE;
    background-color: #FFFFFF;
}

div[data-testid="stExpander"] summary {
    font-weight: 600;
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE3DE;
    border-radius: 8px;
    padding: 12px;
}

/* Responsive */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem 1rem;
    }

    .main-header {
        padding: 19px;
    }

    .main-header h1 {
        font-size: 23px;
    }

    .answer-container {
        padding: 16px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_passages" not in st.session_state:
    st.session_state.last_passages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "research_question" not in st.session_state:
    st.session_state.research_question = ""


# ==================================================
# CALLBACK FUNCTIONS
# ==================================================

def set_research_question(question):
    st.session_state["research_question"] = question


def clear_research_question():
    st.session_state["research_question"] = ""


def clear_current_answer():
    st.session_state["last_answer"] = None
    st.session_state["last_passages"] = []
    st.session_state["last_question"] = ""
    st.session_state["last_error"] = None
    st.session_state["research_question"] = ""


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_value(item, key, default=""):

    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


# ==================================================
# ABSTENTION DETECTION
# ==================================================

def is_abstention(answer):
    """
    Detect whether the LLM indicates that
    the supplied evidence is insufficient.
    """

    if not answer:
        return False

    text = answer.lower()

    abstention_phrases = [
        "insufficient evidence",
        "evidence is insufficient",
        "evidence does not contain",
        "does not contain any material relevant",
        "does not contain relevant information",
        "no relevant information was found",
        "cannot answer based on the provided evidence",
        "unable to answer based on the provided passages",
        "not enough information in the provided evidence",
        "the available evidence does not",
        "no factual response can be derived",
        "cannot provide a reliable answer",
        "the supplied research evidence does not",
        "the retrieved passages do not contain",
        "evidence does not sufficiently address",
        "the evidence is not sufficient",
        "none of the supplied passages address",
        "the evidence is insufficient to answer"
    ]

    return any(
        phrase in text
        for phrase in abstention_phrases
    )


# ==================================================
# REMOVE DUPLICATE ARTICLES
# ==================================================

def unique_passages_by_article(passages):
    """
    Keep the highest-ranked passage from each
    unique article.

    Assumes the retriever returns passages in
    descending reranking order.
    """

    unique = []
    seen_articles = set()

    for passage in passages:

        article_id = str(
            get_value(passage, "article_id", "")
        ).strip().upper()

        if not article_id:
            unique.append(passage)
            continue

        if article_id not in seen_articles:
            unique.append(passage)
            seen_articles.add(article_id)

    return unique


# ==================================================
# CITATION VALIDATION
# ==================================================

def citation_warnings(answer, passages):
    """
    Check whether formal article citations refer to
    articles present in the retrieved passages.

    Supports:
        [A21]
        [A21, p. 15]
        [A21, page 15]

    This checks article ID membership only.
    It does not verify claim-level support or page accuracy.
    """

    if not answer or is_abstention(answer):
        return None

    citation_pattern = (
        r"\[\s*(A\d+)"
        r"(?:\s*,\s*p(?:age)?\.?\s*\d+)?"
        r"\s*\]"
    )

    cited_ids = {
        article_id.upper()
        for article_id in re.findall(
            citation_pattern,
            answer,
            flags=re.IGNORECASE
        )
    }

    # No formal citations were found.
    if not cited_ids:
        return []

    retrieved_ids = {
        str(
            get_value(passage, "article_id", "")
        ).strip().upper()
        for passage in passages
        if get_value(passage, "article_id", "")
    }

    return sorted(cited_ids - retrieved_ids)


# ==================================================
# DISPLAY RESEARCH EVIDENCE
# ==================================================

def display_evidence(passages):

    if not passages:
        st.info("No supporting evidence was retrieved.")
        return

    for index, passage in enumerate(passages, start=1):

        article_id = str(
            get_value(passage, "article_id", "Unknown")
        )

        title = str(
            get_value(passage, "title", "Untitled article")
        )

        year = str(
            get_value(passage, "year", "Not specified")
        )

        page = str(
            get_value(passage, "page", "Not specified")
        )

        passage_text = str(
            get_value(passage, "text", "")
        )

        score = get_value(passage, "rerank_score", None)

        doi = str(
            get_value(passage, "doi", "") or ""
        )

        url = str(
            get_value(passage, "url", "") or ""
        )

        with st.expander(
            f"{index}. {article_id} — {title}",
            expanded=False
        ):

            st.markdown(
                f"**Article ID:** {html.escape(article_id)}  \n"
                f"**Publication year:** {html.escape(year)}  \n"
                f"**Page:** {html.escape(page)}"
            )

            if score is not None:
                st.caption(f"Reranking score: {score}")

            st.markdown("**Retrieved passage**")

            st.text(passage_text)

            if doi:

                doi_link = (
                    doi if doi.startswith("http")
                    else f"https://doi.org/{doi}"
                )

                st.markdown(f"[View DOI]({doi_link})")

            elif url:

                st.markdown(f"[View article]({url})")


# ==================================================
# LOAD SYSTEM
# ==================================================

@st.cache_resource(show_spinner="Loading research assistant...")
def load_system():

    retriever = ResearchRetriever()

    generator = EvidenceAnswerGenerator()

    return retriever, generator


try:

    retriever, generator = load_system()

except Exception as e:

    st.error("The research assistant could not be loaded.")

    with st.expander("Technical details"):
        st.exception(e)

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 📚 Research Assistant")

    st.caption("DAT401 Academic Project")

    st.divider()

    st.markdown("### About")

    st.write(
        "An evidence-grounded research assistant "
        "that answers academic questions using "
        "retrieved research literature."
    )

    st.divider()

    st.markdown("### Research collection")

    st.metric("Research papers", "189")

    st.metric("Indexed passages", "9,728")

    st.divider()

    st.markdown("### System architecture")

    st.caption("BM25 keyword retrieval")

    st.caption("SBERT dense retrieval")

    st.caption("Hybrid Reciprocal Rank Fusion")

    st.caption("Cross-Encoder reranking")

    st.caption("Evidence-grounded LLM")

    st.divider()

    st.markdown("### Research topics")

    st.write(
        "Land Use and Land Cover\n\n"
        "Remote Sensing\n\n"
        "Satellite Imagery\n\n"
        "Classification Algorithms"
    )

    st.divider()

    st.button(
        "Clear current answer",
        use_container_width=True,
        on_click=clear_current_answer
    )


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown("""
<div class="main-header">
    <h1>Research Assistant</h1>
    <p>
        Discover insights from academic literature on
        Land Use and Land Cover (LULC).
        Ask questions and explore answers supported
        by retrieved research evidence.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# SEARCH SECTION
# ==================================================

st.markdown("## Ask a research question")

st.caption(
    "Enter your question to retrieve relevant academic "
    "papers and generate an evidence-grounded answer."
)

question = st.text_area(
    "Research question",
    placeholder=(
        "Example: What are the advantages of Sentinel-2 "
        "imagery for LULC classification?"
    ),
    height=105,
    label_visibility="collapsed",
    key="research_question"
)


search_col, clear_col = st.columns([1, 1])

with search_col:

    search_clicked = st.button(
        "Search research papers",
        type="primary",
        use_container_width=True
    )

with clear_col:

    st.button(
        "Clear question",
        use_container_width=True,
        on_click=clear_research_question
    )


# ==================================================
# SUGGESTED QUESTIONS
# ==================================================

suggestions = [

    "What are the advantages of Sentinel-2 imagery for LULC classification?",

    "Which machine learning algorithms are commonly used for LULC classification?",

    "How does spatial resolution affect classification accuracy?",

    "What are the main challenges of remote sensing for land cover mapping?",

    "How can machine learning improve LULC classification?"

]

with st.expander("Suggested research questions"):

    st.caption("Select a question to place it in the search box.")

    for index, suggestion in enumerate(suggestions):

        st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
            on_click=set_research_question,
            args=(suggestion,)
        )


# ==================================================
# SEARCH AND GENERATE
# ==================================================

if search_clicked:

    current_question = (
        st.session_state.research_question.strip()
    )

    if not current_question:

        st.warning("Please enter a research question first.")

    else:

        st.session_state.last_answer = None

        st.session_state.last_passages = []

        st.session_state.last_question = current_question

        st.session_state.last_error = None

        try:

            # ------------------------------------------
            # RETRIEVAL
            # ------------------------------------------

            with st.spinner(
                "Retrieving relevant research papers..."
            ):

                passages = retriever.retrieve_hybrid_reranked(
                    current_question,
                    top_k=FINAL_TOP_K
                )

            # ------------------------------------------
            # REMOVE DUPLICATE ARTICLES
            # ------------------------------------------

            passages = unique_passages_by_article(passages)

            if not passages:

                st.warning(
                    "No research passages were retrieved. "
                    "Please try rephrasing your question."
                )

            else:

                # --------------------------------------
                # LLM GENERATION
                # --------------------------------------

                with st.spinner(
                    "Generating an evidence-grounded answer..."
                ):

                    answer = generator.generate_answer(
                        current_question,
                        passages,
                        use_evidence=True
                    )

                # --------------------------------------
                # SAVE RESULTS
                # --------------------------------------

                st.session_state.last_answer = answer

                st.session_state.last_passages = passages

        except Exception as e:

            st.session_state.last_error = str(e)


# ==================================================
# DISPLAY ERROR
# ==================================================

if st.session_state.last_error:

    st.error(
        "An error occurred while processing your question."
    )

    with st.expander("View technical details"):

        st.code(st.session_state.last_error)


# ==================================================
# DISPLAY ANSWER
# ==================================================

elif st.session_state.last_answer:

    st.divider()

    st.markdown("## Research answer")

    answer = st.session_state.last_answer

    # Detect abstention.
    abstained = is_abstention(answer)

    # ----------------------------------------------
    # GENERATED ANSWER
    # ----------------------------------------------

    st.markdown(
        '<div class="answer-container">'
        '<div class="answer-label">Generated answer</div>',
        unsafe_allow_html=True
    )

    st.markdown(answer)

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------
    # CITATION VALIDATION
    # ----------------------------------------------

    if abstained:

        st.info(
            "The assistant could not find sufficient "
            "supporting evidence in the research collection. "
            "Please try a question related to remote sensing, "
            "satellite imagery, or Land Use and Land Cover."
        )

    else:

        warnings = citation_warnings(
            answer,
            st.session_state.last_passages
        )

        if warnings:

            st.warning(
                "These cited article IDs were not found "
                "in the retrieved passages: "
                + ", ".join(warnings)
            )

        elif not re.search(
            r"\[\s*A\d+",
            answer,
            re.IGNORECASE
        ):

            st.warning(
                "No formal article citations were detected "
                "in the generated answer. Please verify its "
                "references before using it academically."
            )

        else:

            st.caption(
                "Citation metadata check: cited article IDs "
                "were found among retrieved passages. "
                "This does not independently establish "
                "claim-level citation support or page accuracy."
            )

    # ----------------------------------------------
    # SUPPORTING RESEARCH EVIDENCE
    # ----------------------------------------------

    st.divider()

    if abstained:

        st.markdown("## Research evidence")

        st.info(
            "Retrieved passages are not displayed as "
            "supporting evidence because the assistant "
            "abstained from answering."
        )

    else:

        st.markdown("## Supporting research evidence")

        st.caption(
            f"{len(st.session_state.last_passages)} "
            "retrieved passages from unique research papers."
        )

        display_evidence(
            st.session_state.last_passages
        )


# ==================================================
# INITIAL INFORMATION
# ==================================================

else:

    st.markdown("## Explore the research collection")

    st.markdown("""
    <div class="info-box">
        <div class="info-title">
            Evidence-based research support
        </div>
        <div class="info-description">
            Ask questions about LULC research and receive
            answers grounded in retrieved academic literature,
            with article references and supporting passages.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "DAT401 Academic Project | "
    "Evidence-Grounded Research Assistant"
)

st.caption(
    "Hybrid Retrieval · Cross-Encoder Reranking · LLM Generation"
)
