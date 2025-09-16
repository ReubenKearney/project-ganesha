import os
from typing import Optional

import streamlit as st

# Optional local .env support for development
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    # dotenv is optional at runtime (still in requirements for local dev)
    pass

# ---------- Config helpers ----------

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Resolve a config value from, in order:
    1) st.secrets (Streamlit Cloud)
    2) environment variables
    """
    # Streamlit Cloud secrets
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets.get(key, default)

    # Environment variables (local dev)
    return os.getenv(key, default)

# Example config keys (replace/extend per use case)
API_BASE_URL = get_secret("API_BASE_URL", "")
API_KEY = get_secret("API_KEY", "")

# ---------- Page state ----------

st.set_page_config(
    page_title="Streamlit App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar ----------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select view",
    options=[
        "Overview",
        "Use Case",
        "Diagnostics",
    ],
    index=0,
)

with st.sidebar.expander("Configuration", expanded=False):
    st.caption("Runtime config (read-only)")
    st.code(
        {
            "API_BASE_URL": API_BASE_URL or "<unset>",
            "API_KEY": "***" if API_KEY else "<unset>",
        },
        language="json",
    )

# ---------- Utility functions ----------

@st.cache_data(show_spinner=False)
def fetch_sample_data() -> dict:
    """
    Placeholder data loader for the upcoming use case.
    Replace with concrete API/file/database access as needed.
    """
    # Example structure; replace with real fetch using requests, pandas, etc.
    data = {
        "status": "ok",
        "items": [
            {"id": 1, "name": "Alpha", "value": 42},
            {"id": 2, "name": "Beta", "value": 17},
            {"id": 3, "name": "Gamma", "value": 29},
        ],
    }
    return data

def require_config(*keys: str) -> bool:
    """
    Validates that required config keys are present.
    Renders an inline warning if any are missing.
    Returns True if all keys are available; otherwise False.
    """
    missing = [k for k in keys if not get_secret(k)]
    if missing:
        st.warning(f"Missing required configuration: {', '.join(missing)}")
        return False
    return True

# ---------- Pages ----------

def render_overview():
    st.title("Overview")
    st.write(
        """
        This is a scaffold for a Streamlit application connected to a GitHub repo.
        The concrete functionality will be implemented in the **Use Case** page.
        """
    )

def render_use_case():
    st.title("Use Case")
    # Example: enforce required config before proceeding
    # Update required keys as the use case is defined.
    if not require_config("API_BASE_URL"):
        st.stop()

    st.subheader("Inputs")
    # Replace with actual inputs (text, file_uploader, selectbox, etc.)
    query = st.text_input("Query", placeholder="Enter a query or parameter")

    st.subheader("Actions")
    trigger = st.button("Run", type="primary", use_container_width=False)

    st.subheader("Outputs")
    if trigger:
        with st.spinner("Processing..."):
            data = fetch_sample_data()
        st.success("Completed")
        st.json(data)

def render_diagnostics():
    st.title("Diagnostics")
    st.write("Environment")
    st.code(
        {
            "python_version": os.sys.version.split()[0],
            "streamlit_version": st.__version__,
            "working_dir": os.getcwd(),
        },
        language="json",
    )

# ---------- Router ----------

if page == "Overview":
    render_overview()
elif page == "Use Case":
    render_use_case()
elif page == "Diagnostics":
    render_diagnostics()
else:
    st.error("Unknown page")
