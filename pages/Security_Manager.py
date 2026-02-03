"""
Security Manager for Digital Witness.

A dedicated page for store managers/security to review and manage alerts.
Accessible as a separate Streamlit page.
"""
import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts import AlertManager, AlertStatus, ManagedAlert, NotificationService
from src.analysis import Severity
from src.config import ALERTS_STORAGE_DIR


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Security Manager - Digital Witness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'


# ============================================================================
# Theme-aware CSS
# ============================================================================

def get_theme_css():
    """Generate CSS based on current theme."""
    is_dark = st.session_state.theme == 'dark'

    if is_dark:
        bg_primary = '#0e1117'
        bg_secondary = '#1a1d24'
        bg_card = '#262730'
        text_primary = '#fafafa'
        text_secondary = '#b0b0b0'
        border_color = '#3d4149'
    else:
        bg_primary = '#ffffff'
        bg_secondary = '#f8f9fa'
        bg_card = '#ffffff'
        text_primary = '#1e3a5f'
        text_secondary = '#666666'
        border_color = '#dee2e6'

    return f"""
    <style>
        /* Theme variables */
        :root {{
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-card: {bg_card};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --border-color: {border_color};
        }}

        /* Page navigation */
        .page-nav {{
            display: flex;
            justify-content: center;
            gap: 10px;
            padding: 15px;
            background: {bg_secondary};
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .page-nav-btn {{
            padding: 10px 25px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .page-nav-btn-active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }}
        .page-nav-btn-inactive {{
            background: {bg_card};
            color: {text_secondary};
            border: 1px solid {border_color};
        }}
        .page-nav-btn-inactive:hover {{
            background: {border_color};
        }}

        /* Alert cards */
        .alert-card {{
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid;
            background: {bg_card};
        }}
        .alert-critical {{ border-color: #dc3545; background: {'#3d2429' if is_dark else '#f8d7da'}; }}
        .alert-high {{ border-color: #fd7e14; background: {'#3d3124' if is_dark else '#fff3cd'}; }}
        .alert-medium {{ border-color: #ffc107; background: {'#3d3924' if is_dark else '#fff9e6'}; }}
        .alert-low {{ border-color: #17a2b8; background: {'#243d3d' if is_dark else '#d1ecf1'}; }}

        /* Status badges */
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-pending {{ background: #ffc107; color: #000; }}
        .status-acknowledged {{ background: #28a745; color: #fff; }}
        .status-dismissed {{ background: #6c757d; color: #fff; }}
        .status-escalated {{ background: #dc3545; color: #fff; }}

        /* Metric cards */
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            color: white;
            text-align: center;
        }}
        .metric-value {{ font-size: 36px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; opacity: 0.9; }}

        /* Header */
        .dashboard-header {{
            background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }}

        /* Text colors */
        .text-secondary {{
            color: {text_secondary};
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border-radius: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }}
    </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)


# ============================================================================
# Initialize Session State
# ============================================================================

def init_session_state():
    """Initialize session state for manager dashboard."""
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager(ALERTS_STORAGE_DIR)

    if 'notification_service' not in st.session_state:
        st.session_state.notification_service = NotificationService()

    if 'selected_alert_id' not in st.session_state:
        st.session_state.selected_alert_id = None

    if 'manager_name' not in st.session_state:
        st.session_state.manager_name = "Manager"


init_session_state()


# ============================================================================
# Helper Functions
# ============================================================================

def get_severity_color(severity: Severity) -> str:
    """Get color for severity level."""
    colors = {
        Severity.CRITICAL: "#dc3545",
        Severity.HIGH: "#fd7e14",
        Severity.MEDIUM: "#ffc107",
        Severity.LOW: "#17a2b8",
        Severity.NONE: "#28a745"
    }
    return colors.get(severity, "#6c757d")


def get_status_class(status: AlertStatus) -> str:
    """Get CSS class for status badge."""
    return f"status-{status.value}"


def format_time_ago(dt: datetime) -> str:
    """Format datetime as relative time."""
    delta = datetime.now() - dt
    if delta.days > 0:
        return f"{delta.days}d ago"
    elif delta.seconds >= 3600:
        return f"{delta.seconds // 3600}h ago"
    elif delta.seconds >= 60:
        return f"{delta.seconds // 60}m ago"
    else:
        return "Just now"


# ============================================================================
# Theme Toggle
# ============================================================================

def render_theme_toggle():
    """Render theme toggle at the top of the page."""
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        is_dark = st.toggle(
            "Dark Mode",
            value=st.session_state.theme == 'dark',
            key="theme_toggle_security"
        )
        new_theme = 'dark' if is_dark else 'light'
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render sidebar with filters and settings."""
    st.sidebar.markdown("## Security Manager")

    # Navigation
    st.sidebar.markdown("### Navigation")
    st.sidebar.page_link("app.py", label="Main Dashboard", icon="📊")
    st.sidebar.page_link("pages/Security_Manager.py", label="Security Manager", icon="🛡️")

    st.sidebar.markdown("---")

    # Manager identification
    st.session_state.manager_name = st.sidebar.text_input(
        "Your Name",
        value=st.session_state.manager_name,
        help="Used for audit trail when acknowledging alerts"
    )

    st.sidebar.markdown("---")

    # Filters
    st.sidebar.markdown("### Filters")

    status_filter = st.sidebar.multiselect(
        "Status",
        options=[s.value for s in AlertStatus],
        default=["pending"],
        help="Filter alerts by status"
    )

    severity_filter = st.sidebar.multiselect(
        "Severity",
        options=[s.value for s in Severity if s != Severity.NONE],
        default=[],
        help="Filter by severity (empty = all)"
    )

    st.sidebar.markdown("---")

    # Actions
    st.sidebar.markdown("### Actions")

    if st.sidebar.button("Refresh Alerts", use_container_width=True):
        st.session_state.alert_manager._load_alerts()
        st.rerun()

    if st.sidebar.button("Clear Old Alerts", use_container_width=True):
        removed = st.session_state.alert_manager.clear_old_alerts(days=30)
        st.sidebar.success(f"Removed {removed} old alerts")

    st.sidebar.markdown("---")

    # Statistics
    stats = st.session_state.alert_manager.get_statistics()
    st.sidebar.markdown("### Statistics")
    st.sidebar.metric("Total Alerts", stats["total_alerts"])
    st.sidebar.metric("Pending", stats["pending_count"])
    st.sidebar.metric("Notifications Sent", stats["notifications_sent"])

    return status_filter, severity_filter


# ============================================================================
# Main Content
# ============================================================================

def render_header():
    """Render dashboard header."""
    render_theme_toggle()
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin: 0;">Security Manager</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.8;">
            Digital Witness - Real-Time Security Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    """Render metrics row."""
    stats = st.session_state.alert_manager.get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pending = stats["by_status"].get("pending", 0)
        st.metric(
            "Pending Alerts",
            pending,
            delta=None,
            delta_color="inverse" if pending > 0 else "normal"
        )

    with col2:
        critical = stats["by_severity"].get("CRITICAL", 0)
        high = stats["by_severity"].get("HIGH", 0)
        st.metric("Critical/High", critical + high)

    with col3:
        ack = stats["by_status"].get("acknowledged", 0)
        st.metric("Acknowledged", ack)

    with col4:
        st.metric("Notifications", stats["notifications_sent"])


def render_alert_card(alert: ManagedAlert):
    """Render a single alert card."""
    severity_color = get_severity_color(alert.severity)
    text_color = '#b0b0b0' if st.session_state.theme == 'dark' else '#666'

    # Card container
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"""
            <div style="border-left: 4px solid {severity_color}; padding-left: 15px;">
                <h4 style="margin: 0; color: {severity_color};">
                    {alert.severity.value} Risk Alert
                </h4>
                <p style="margin: 5px 0; color: {text_color};">
                    {alert.alert_id} - {format_time_ago(alert.created_at)}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: {severity_color};">
                    {alert.intent_score:.0%}
                </div>
                <div style="font-size: 12px; color: {text_color};">Risk Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            status_colors = {
                "pending": "#ffc107",
                "acknowledged": "#28a745",
                "dismissed": "#6c757d",
                "escalated": "#dc3545"
            }
            status_color = status_colors.get(alert.status.value, "#6c757d")
            st.markdown(f"""
            <div style="text-align: center;">
                <span style="background: {status_color}; color: white; padding: 4px 12px;
                       border-radius: 20px; font-size: 12px;">
                    {alert.status.value.upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Expandable details
        with st.expander("View Details & Actions"):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Discrepancy Summary:**")
                st.text(alert.alert.discrepancy_summary)

            with detail_col2:
                st.markdown("**Behavior Summary:**")
                st.text(alert.alert.behavior_summary)

            st.markdown("**Full Explanation:**")
            st.text(alert.alert.explanation[:500] + "..." if len(alert.alert.explanation) > 500 else alert.alert.explanation)

            # Forensic Evidence Clips (XAI)
            if alert.alert.evidence_clips:
                st.markdown("---")
                st.markdown("**Forensic Evidence Clips:**")
                st.caption("Video segments showing detected suspicious behavior for review")

                clip_cols = st.columns(min(len(alert.alert.evidence_clips), 3))
                for i, clip in enumerate(alert.alert.evidence_clips):
                    with clip_cols[i % 3]:
                        clip_path = clip.path
                        if clip_path.exists():
                            st.video(str(clip_path))
                            st.caption(f"{clip.event_type} ({clip.start_time:.1f}s - {clip.end_time:.1f}s)")
                        else:
                            st.warning(f"Clip not found: {clip_path.name}")
                            st.caption(f"{clip.event_type} ({clip.start_time:.1f}s - {clip.end_time:.1f}s)")

            # Notes
            if alert.notes:
                st.markdown("**Notes:**")
                for note in alert.notes[-3:]:  # Show last 3 notes
                    st.markdown(f"- *{note.timestamp.strftime('%H:%M')}* ({note.user}): {note.note}")

            st.markdown("---")

            # Action buttons
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)

            with action_col1:
                if alert.status == AlertStatus.PENDING:
                    if st.button("Acknowledge", key=f"ack_{alert.alert_id}", use_container_width=True):
                        st.session_state.alert_manager.acknowledge_alert(
                            alert.alert_id,
                            st.session_state.manager_name
                        )
                        st.rerun()

            with action_col2:
                if alert.status in [AlertStatus.PENDING, AlertStatus.ACKNOWLEDGED]:
                    if st.button("Dismiss", key=f"dismiss_{alert.alert_id}", use_container_width=True):
                        st.session_state.selected_alert_id = alert.alert_id
                        st.session_state.dismiss_mode = True

            with action_col3:
                if alert.status in [AlertStatus.PENDING, AlertStatus.ACKNOWLEDGED]:
                    if st.button("Escalate", key=f"escalate_{alert.alert_id}", use_container_width=True):
                        st.session_state.selected_alert_id = alert.alert_id
                        st.session_state.escalate_mode = True

            with action_col4:
                if st.button("Add Note", key=f"note_{alert.alert_id}", use_container_width=True):
                    st.session_state.selected_alert_id = alert.alert_id
                    st.session_state.note_mode = True

        st.markdown("---")


def render_alert_list(status_filter: list, severity_filter: list):
    """Render filtered alert list."""
    alerts = st.session_state.alert_manager.get_all_alerts()

    # Apply filters
    if status_filter:
        alerts = [a for a in alerts if a.status.value in status_filter]

    if severity_filter:
        alerts = [a for a in alerts if a.severity.value in severity_filter]

    # Sort by creation time (newest first)
    alerts.sort(key=lambda a: a.created_at, reverse=True)

    if not alerts:
        st.info("No alerts match the current filters.")
        return

    st.markdown(f"### Showing {len(alerts)} Alert(s)")

    for alert in alerts:
        render_alert_card(alert)


def render_modals():
    """Render modal dialogs for actions."""
    # Dismiss modal
    if st.session_state.get('dismiss_mode') and st.session_state.selected_alert_id:
        with st.form("dismiss_form"):
            st.markdown("### Dismiss Alert")
            reason = st.text_area("Reason for dismissal", placeholder="e.g., False positive, customer was staff member")

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Confirm Dismiss", type="primary"):
                    if reason:
                        st.session_state.alert_manager.dismiss_alert(
                            st.session_state.selected_alert_id,
                            st.session_state.manager_name,
                            reason
                        )
                        st.session_state.dismiss_mode = False
                        st.session_state.selected_alert_id = None
                        st.rerun()
                    else:
                        st.error("Please provide a reason")

            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.dismiss_mode = False
                    st.session_state.selected_alert_id = None
                    st.rerun()

    # Escalate modal
    if st.session_state.get('escalate_mode') and st.session_state.selected_alert_id:
        with st.form("escalate_form"):
            st.markdown("### Escalate Alert")
            reason = st.text_area("Reason for escalation", placeholder="e.g., Requires immediate manager attention")

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Confirm Escalate", type="primary"):
                    if reason:
                        st.session_state.alert_manager.escalate_alert(
                            st.session_state.selected_alert_id,
                            st.session_state.manager_name,
                            reason
                        )
                        st.session_state.escalate_mode = False
                        st.session_state.selected_alert_id = None
                        st.rerun()
                    else:
                        st.error("Please provide a reason")

            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.escalate_mode = False
                    st.session_state.selected_alert_id = None
                    st.rerun()

    # Add note modal
    if st.session_state.get('note_mode') and st.session_state.selected_alert_id:
        with st.form("note_form"):
            st.markdown("### Add Note")
            note = st.text_area("Note", placeholder="Add your observations or comments")

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Add Note", type="primary"):
                    if note:
                        st.session_state.alert_manager.add_note(
                            st.session_state.selected_alert_id,
                            st.session_state.manager_name,
                            note
                        )
                        st.session_state.note_mode = False
                        st.session_state.selected_alert_id = None
                        st.rerun()
                    else:
                        st.error("Please enter a note")

            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.note_mode = False
                    st.session_state.selected_alert_id = None
                    st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main dashboard function."""
    # Sidebar
    status_filter, severity_filter = render_sidebar()

    # Main content
    render_header()
    render_metrics()
    st.markdown("---")

    # Modals (if any action is pending)
    render_modals()

    # Alert list
    render_alert_list(status_filter, severity_filter)

    # Auto-refresh hint
    text_color = '#b0b0b0' if st.session_state.theme == 'dark' else '#666'
    st.markdown(f"""
    <div style="text-align: center; color: {text_color}; font-size: 12px; margin-top: 20px;">
        Click "Refresh Alerts" in the sidebar to check for new alerts
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
