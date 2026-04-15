"""ui/styles.py — Application CSS, injected once at startup."""

CSS_BLOCK: str = """
<style>
    .main-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(120deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0; margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.3rem; color: #666; text-align: center;
        margin-bottom: 2rem; font-weight: 400;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-none {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa; border-left: 4px solid #1e3a5f;
        padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;
    }
    .section-header {
        font-size: 1.5rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 3px solid #3d7ab5; padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.75rem 2rem;
        font-size: 1.1rem; font-weight: 600; border-radius: 10px;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { padding-left: 1rem; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 10px 10px 0 0;
        padding: 0.4rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
    }
</style>
"""
