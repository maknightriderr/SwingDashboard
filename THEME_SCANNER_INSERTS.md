# 🎯 Theme Scanner — 3 edits to app.py

You're adding **one new file** (`theme_scanner.py`) and making **3 tiny edits** to `app.py`.
Your `signals.py` is NOT touched at all.

---

## EDIT 1 — Import the module (near the top, with the other feature imports)

**FIND** this block (around line 88–94):

```python
# ── Mutual Fund & ETF module (fully separate from stock/signals logic) ─────────
try:
    import funds as _funds
    _FUNDS_AVAILABLE = True
except Exception:
    _funds = None
    _FUNDS_AVAILABLE = False
```

**ADD directly AFTER it:**

```python
# ── Market Theme Scanner module (curated NSE theme baskets) ────────────────────
try:
    import theme_scanner as _themes
    _THEMES_AVAILABLE = True
except Exception:
    _themes = None
    _THEMES_AVAILABLE = False
```

---

## EDIT 2 — Add the nav link (inside NAV_GROUPS)

**FIND** the "Market Intelligence" group (around line 1240):

```python
        "🔄 Market Intelligence": [
            ("🔄 Sector Rotation",   "sector"),
            ("💪 RS Leaders",        "rs"),
```

**ADD** the Theme Scanner line right after "Sector Rotation":

```python
        "🔄 Market Intelligence": [
            ("🔄 Sector Rotation",   "sector"),
            ("🎯 Theme Scanner",     "themes"),
            ("💪 RS Leaders",        "rs"),
```

(Just insert the one `("🎯 Theme Scanner", "themes"),` line.)

---

## EDIT 3 — Add the page block

**FIND** the Sector Rotation page block. It starts with:

```python
# ── Sector Rotation ──────────────────────────────────────────────────────────
elif _page == 'sector':
```

Scroll to the END of that block — it finishes with the picks render:

```python
    if st.session_state.picks_cache is not None:
        st.markdown(
            '<div class="sec" style="margin-top:2rem">🎯 Algorithmic Entry Setups</div>',
            unsafe_allow_html=True)
        render_picks(st.session_state.picks_cache, theme_t)
```

**ADD the entire block below directly AFTER those lines** (and before the next
`# ── Universe Scanner ──` / `elif _page == 'scanner':`):

```python
# ── Theme Scanner ────────────────────────────────────────────────────────────
elif _page == 'themes':
    st.markdown('<div class="sec">🎯 Market Theme Scanner</div>',
                unsafe_allow_html=True)
    st.caption("Curated NSE market themes (Defence, Railways, EV, PSU Banks, "
               "Green Energy, Capex…) ranked by aggregate strength — momentum, "
               "breadth above key EMAs, and relative strength vs Nifty. See where "
               "capital is rotating, then drill into each theme's leaders.")

    if not _THEMES_AVAILABLE:
        st.warning("🎯 Theme Scanner needs `theme_scanner.py` in your repo root "
                   "(same folder as app.py and signals.py). Upload it, then reboot.",
                   icon="⚠️")
    else:
        tc1, tc2 = st.columns([3, 1])
        with tc1:
            st.markdown(
                '<div style="font-size:.8rem;color:var(--muted);font-weight:600">'
                'Each theme is a curated basket drawn from your existing universe. '
                'Edit the lists in <code>theme_scanner.py</code> any time.</div>',
                unsafe_allow_html=True)
        with tc2:
            run_theme = st.button("🎯 Scan Themes", width="stretch")

        with st.expander("🔍 Theme basket coverage", expanded=False):
            try:
                st.code(_themes.theme_coverage(), language=None)
            except Exception as _e:
                st.caption(f"Coverage unavailable: {_e}")

        if run_theme:
            with st.spinner("Scoring NSE themes by aggregate strength…"):
                st.session_state.theme_scan_cache = _themes.scan_themes(
                    min_constituents=3)

        tdata = st.session_state.get("theme_scan_cache")
        if tdata is None:
            st.info("💡 Click **🎯 Scan Themes** to rank NSE themes hottest → coldest.")
        elif not tdata.get("themes"):
            st.warning("No themes had enough scored constituents. "
                       + tdata.get("note", "Try again — Yahoo may be rate-limited."))
        else:
            themes = tdata["themes"]
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);margin-bottom:1rem">'
                f'Ranked {len(themes)} themes · scanned {tdata["scanned"]} stocks · '
                f'{tdata["timestamp"]}</div>',
                unsafe_allow_html=True)

            # ── Hottest-themes summary cards ──────────────────────────────────
            cards = ""
            for t in themes:
                rank = t["rank"]
                score = t["score"]
                medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else
                         "🥉" if rank == 3 else f"#{rank}")
                # Hot/warm/cold colour
                if score >= 55:
                    clr = theme_t["green"]; bdr = theme_t["accent"]
                elif score >= 30:
                    clr = theme_t["yellow"]; bdr = theme_t["border"]
                else:
                    clr = theme_t["muted"]; bdr = theme_t["border"]
                rs = t["avg_rs"]
                rs_clr = theme_t["green"] if rs >= 1.0 else theme_t["red"]
                cards += (
                    f'<div style="background:var(--card);border:1px solid {bdr};'
                    f'border-radius:12px;padding:1rem 1.1rem;min-width:200px;flex:1">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:.3rem">'
                    f'<span style="font-size:.82rem;font-weight:800;color:var(--text)">'
                    f'{medal} {t["theme"]}</span></div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:{clr};'
                    f'margin:.1rem 0">{score:.0f}'
                    f'<span style="font-size:.6rem;color:var(--muted)"> /100</span></div>'
                    f'<div style="height:5px;background:var(--input);border-radius:3px;'
                    f'margin:.3rem 0 .5rem"><div style="height:5px;border-radius:3px;'
                    f'background:{clr};width:{min(score,100):.0f}%"></div></div>'
                    f'<div style="font-size:.68rem;color:var(--muted);line-height:1.5">'
                    f'1M <b style="color:var(--text)">{t["avg_ret_1m"]:+.1f}%</b> · '
                    f'3M <b style="color:var(--text)">{t["avg_ret_3m"]:+.1f}%</b><br>'
                    f'{t["pct_above_50ema"]:.0f}% &gt;50EMA · '
                    f'RS <b style="color:{rs_clr}">{rs:.2f}</b> · '
                    f'{t["n_scored"]} stks</div></div>'
                )
            st.markdown(
                f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;'
                f'margin-bottom:1.5rem">{cards}</div>',
                unsafe_allow_html=True)

            # ── Per-theme drill-down (expandable constituents) ────────────────
            st.markdown('<div class="sec">🔬 Theme Constituents</div>',
                        unsafe_allow_html=True)
            for t in themes:
                medal = ("🥇" if t["rank"] == 1 else "🥈" if t["rank"] == 2 else
                         "🥉" if t["rank"] == 3 else f"#{t['rank']}")
                with st.expander(
                        f"{medal} {t['theme']} — score {t['score']:.0f} · "
                        f"1M {t['avg_ret_1m']:+.1f}% · breadth {t['breadth_up']:.0f}% up · "
                        f"{t['n_scored']} stocks", expanded=(t["rank"] == 1)):
                    rows = []
                    for l in t["leaders"]:
                        rows.append({
                            "Stock": l["stock"],
                            "CMP": l["cmp"],
                            "1M %": l["ret_1m"],
                            "3M %": l["ret_3m"],
                            "RS": l["rs_ratio"],
                            ">50EMA": "✅" if l["above_50ema"] else "—",
                            ">200EMA": "✅" if l["above_200ema"] else "—",
                            "Trend": l["trend"],
                        })
                    import pandas as _pd_t
                    tdf = _pd_t.DataFrame(rows)
                    _h = min(max(len(tdf) * 36 + 40, 150), 480)
                    st.dataframe(
                        tdf, hide_index=True, height=_h, use_container_width=True,
                        column_config={
                            "Stock": st.column_config.TextColumn("Stock", width="small", pinned=True),
                            "CMP":   st.column_config.NumberColumn("CMP", format="₹%.2f"),
                            "1M %":  st.column_config.NumberColumn("1M %", format="%.1f"),
                            "3M %":  st.column_config.NumberColumn("3M %", format="%.1f"),
                            "RS":    st.column_config.NumberColumn("RS", format="%.2f"),
                        })
                    st.download_button(
                        f"⬇️ Export {t['theme']} CSV",
                        tdf.to_csv(index=False).encode("utf-8"),
                        file_name=f"theme_{t['theme'].replace(' ','_').replace('/','-')}_"
                                  f"{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv", key=f"theme_dl_{t['rank']}")
            st.caption("💡 Hot themes (green, score 55+) show where capital is rotating. "
                       "Inside each, stocks are ranked by relative strength — the leaders "
                       "of the leading themes are your highest-conviction shortlist. "
                       "Always confirm your own entry setup before trading.")
```

---

## EDIT 4 (one line) — register the session cache key

**FIND** this line in the big session-state init list (around line 305):

```python
             ("smc_scan_cache", None), ("vcp_scan_cache", None), ("rs_scan_cache", None),
```

**CHANGE it to** (just add `("theme_scan_cache", None),`):

```python
             ("smc_scan_cache", None), ("vcp_scan_cache", None), ("rs_scan_cache", None),
             ("theme_scan_cache", None),
```

---

## Deploy

1. Upload **`theme_scanner.py`** to your repo root (drag via *Add file → Upload files*).
2. Apply the 4 edits above to **`app.py`**, commit.
3. Reboot the app (*Manage app → Reboot*).
4. Open **🎯 Theme Scanner** in the sidebar → click **Scan Themes**.
