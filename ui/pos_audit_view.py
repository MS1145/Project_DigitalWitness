"""
ui/pos_audit_view.py - Mock POS terminal and transaction-vs-video mismatch audit.
"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import PRODUCT_CATALOG
from core.result_types import PipelineResult


class PosAuditView:
    """
    Renders the POS Audit tab.

    Left  - Manual POS entry: user picks items + quantities.
    Right - Video evidence: behavioural signals from the pipeline.
    Bottom - Mismatch table: flags items detected but not rung up.
    """

    def render(self, result: PipelineResult | None) -> None:
        st.markdown('<div class="section-header">Mock POS Terminal & Audit</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Simulate a POS transaction then compare it against what the camera
            detected. Mismatches between rung-up items and detected product
            interactions are highlighted as potential theft indicators.
        </div>""", unsafe_allow_html=True)

        col_pos, col_vid = st.columns(2)
        with col_pos:
            self._render_pos_terminal()
        with col_vid:
            self._render_video_evidence(result)

        st.markdown("---")
        self._render_mismatch_audit(result)

    #  Private section renderers ─
    def _render_pos_terminal(self) -> None:
        st.markdown("### Mock POS Terminal")
        st.caption("Enter items as they are being rung up at the checkout.")

        cat_names  = [p["name"] for p in PRODUCT_CATALOG]
        cat_by_name = {p["name"]: p for p in PRODUCT_CATALOG}

        with st.form("pos_form", clear_on_submit=False):
            sel_name = st.selectbox("Select product", cat_names)
            sel_qty  = st.number_input("Quantity", min_value=1, max_value=20, value=1)
            add_btn  = st.form_submit_button("Add to Transaction")

        if add_btn:
            item       = dict(cat_by_name[sel_name])
            item["qty"] = int(sel_qty)
            st.session_state.pos_items.append(item)

        if st.button("Clear Transaction", key="clear_pos"):
            st.session_state.pos_items = []

        if st.session_state.pos_items:
            st.markdown("#### Transaction Ledger")
            rows  = []
            total = 0.0
            for it in st.session_state.pos_items:
                subtotal = it["price"] * it["qty"]
                total   += subtotal
                rows.append({
                    "SKU"     : it["sku"],
                    "Product" : it["name"],
                    "Qty"     : it["qty"],
                    "Unit £"  : f"£{it['price']:.2f}",
                    "Subtotal": f"£{subtotal:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(f"**Total: £{total:.2f}**")
        else:
            st.info("No items added yet.")

    def _render_video_evidence(self, result: PipelineResult | None) -> None:
        st.markdown("### Behavioural Video Evidence")
        st.caption(
            "The behaviour model classifies person actions, not specific products. "
            "Evidence is based on detected interaction events."
        )
        if not result:
            st.info("Run video analysis first.")
            return

        det           = result.detection
        verdict       = result.lstm_verdict
        events        = result.behaviour_events
        pickups       = det.product_pickups if det else {}
        picking_count = pickups.get("Picking-Holding", 0)
        shop_windows  = sum(1 for e in events if e.behavior_type == "shoplifting")
        susp_windows  = sum(1 for e in events
                            if e.behavior_type in {"shoplifting", "Looking around"})

        vid_rows = [
            {"Behavioural Signal": "Persons tracked",
             "Count": det.persons_tracked if det else 0,
             "Risk" : "-"},
            {"Behavioural Signal": "Item interaction (Picking-Holding)",
             "Count": picking_count,
             "Risk" : "Possible" if picking_count > 0 else "-"},
            {"Behavioural Signal": "Suspicious behaviour windows (LSTM)",
             "Count": susp_windows,
             "Risk" : "High" if susp_windows > 0 else "None"},
            {"Behavioural Signal": "Shoplifting classification windows",
             "Count": shop_windows,
             "Risk" : "High" if shop_windows > 0 else "None"},
        ]
        st.dataframe(pd.DataFrame(vid_rows), use_container_width=True, hide_index=True)

        if verdict and verdict.is_shoplifting:
            st.error(
                f"LSTM verdict: **SHOPLIFTING** ({verdict.confidence:.0%}) - "
                "transaction flagged for review."
            )
        elif verdict:
            st.success(
                f"LSTM verdict: **NORMAL** ({verdict.confidence:.0%}) - "
                "no shoplifting pattern detected."
            )

    def _render_mismatch_audit(self, result: PipelineResult | None) -> None:
        st.markdown("### Transaction Risk Audit")
        st.caption(
            "Since the behaviour model does not identify specific products, the audit "
            "compares the number of items rung up at POS against the number of "
            "Picking-Holding interactions and suspicious behaviour windows detected."
        )
        if not result:
            st.info("Run video analysis first.")
            return
        if not st.session_state.pos_items:
            st.info("Add items to the POS transaction to compare.")
            return

        det    = result.detection
        verdict = result.lstm_verdict
        events = result.behaviour_events

        pos_total_qty = sum(it["qty"] for it in st.session_state.pos_items)
        pos_total_val = sum(it["price"] * it["qty"] for it in st.session_state.pos_items)
        picking_count = (det.product_pickups.get("Picking-Holding", 0) if det else 0)
        shop_windows  = sum(1 for e in events if e.behavior_type == "shoplifting")
        is_shop       = verdict.is_shoplifting if verdict else False

        audit_rows = [
            {"Metric": "POS items scanned",
             "Value" : pos_total_qty,
             "Status": "OK"},
            {"Metric": "POS transaction value",
             "Value" : f"£{pos_total_val:.2f}",
             "Status": "OK"},
            {"Metric": "Item interactions detected (Picking-Holding)",
             "Value" : picking_count,
             "Status": ("More handled than scanned"
                        if picking_count > pos_total_qty else "OK")},
            {"Metric": "Shoplifting behaviour windows (LSTM)",
             "Value" : shop_windows,
             "Status": "Suspicious" if shop_windows > 0 else "None"},
        ]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        flags = []
        if is_shop:
            flags.append(
                f"LSTM classified video as **SHOPLIFTING** "
                f"({verdict.confidence:.0%}) while {pos_total_qty} item(s) were rung up. "
                "Verify transaction matches items handled."
            )
        if picking_count > pos_total_qty:
            flags.append(
                f"**{picking_count} item interaction(s)** detected but only "
                f"**{pos_total_qty} item(s)** scanned at POS - potential scan avoidance."
            )
        if is_shop and pos_total_qty == 0:
            flags.append(
                "Shoplifting behaviour detected but **no purchase recorded** at POS."
            )

        if flags:
            st.markdown("#### Potential Theft Indicators")
            for f in flags:
                st.error(f)
            st.warning("**Human review required** before any action is taken.")
        else:
            st.success("No mismatches found - POS transaction consistent with video evidence.")

        fig = go.Figure()
        fig.add_bar(name="POS Items Scanned",
                    x=["Transaction"], y=[pos_total_qty], marker_color="#38ef7d")
        fig.add_bar(name="Item Interactions Detected",
                    x=["Transaction"], y=[picking_count],  marker_color="#667eea")
        fig.add_bar(name="Shoplifting Windows",
                    x=["Transaction"], y=[shop_windows],   marker_color="#dc3545")
        fig.update_layout(barmode="group", title="POS vs Behavioural Evidence",
                          height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
