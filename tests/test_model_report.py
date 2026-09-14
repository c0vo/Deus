"""
Tests for pipeline.model_report — the per-horizon skill table.

The same table goes to Telegram (HTML, parse_mode="HTML") after the weekly
retrain and to the console of the training script, so both renderers are
checked against model_metrics rows shaped as Database.get_latest_model_metrics
returns them. Pure string assertions: no database, no network.
"""

from __future__ import annotations

import re

from pipeline.model_report import COLUMNS, LEGEND, render_metrics_table_html, render_metrics_table_text


def _row(horizon: int, **overrides) -> dict:
    row = {
        "id": horizon, "run_id": "weekly-test", "created_at": "2026-09-13 13:00:00",
        "scope": "universal", "horizon_days": horizon, "schema_version": 4,
        "config_name": "run2", "status": "model", "n_rows": 98431, "n_dates": 6100,
        "n_tickers": 21, "train_end": "2026-08-14",
        "auc_mean": 0.5412, "auc_std": 0.012, "auc_ci_low": 0.5213, "auc_ci_high": 0.5634,
        "logloss_mean": 0.689, "brier_mean": 0.2471, "brier_skill_mean": 0.0042,
        "acc_mean": 0.551, "acc_majority_mean": 0.54, "hi_conf_acc": 0.561, "hi_conf_n": 1204,
        "decile_spread_mean": 0.004, "prior_up_rate": 0.543,
        "config_json": {"name": "run2", "universe": "core"}, "folds_json": [], "importance_json": {},
    }
    row.update(overrides)
    return row


MODEL = _row(5)
PRIOR = _row(21, status="prior", auc_mean=0.5081, auc_ci_low=0.4893, auc_ci_high=0.5270,
             brier_skill_mean=-0.0011, hi_conf_acc=0.52, hi_conf_n=88, prior_up_rate=0.571)
NOT_MEASURABLE = _row(252, status="prior", auc_mean=None, auc_std=None, auc_ci_low=None,
                      auc_ci_high=None, brier_skill_mean=None, hi_conf_acc=None, hi_conf_n=None,
                      decile_spread_mean=None, prior_up_rate=0.724, n_rows=31000,
                      config_json={"name": "run2", "note": "walkforward_folds(h=252): 1 usable fold(s), need >= 2"})


def _table_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if " | " in line]


def test_text_table_has_the_contract_columns_and_one_line_per_horizon():
    text = render_metrics_table_text([PRIOR, NOT_MEASURABLE, MODEL])
    lines = _table_lines(text)
    assert [c.strip() for c in lines[0].split("|")] == list(COLUMNS)
    assert list(COLUMNS) == ["Horizon", "Status", "AUC [90% CI]", "Brier skill", "Hi-conf acc (n)",
                             "Base up", "Tickers/Rows"]
    body = lines[1:]
    assert [line.split("|")[0].strip() for line in body] == ["5d", "1m", "1y"]   # sorted by horizon
    for legend in LEGEND:
        assert legend in text


def test_model_row_renders_auc_with_its_interval_and_skill():
    line = _table_lines(render_metrics_table_text([MODEL]))[1]
    cells = [c.strip() for c in line.split("|")]
    assert cells == ["5d", "model", "0.541 [0.52-0.56]", "+0.0042", "56.1% (1,204)", "54.3%", "21/98,431"]


def test_prior_row_keeps_the_metrics_that_explain_it():
    line = _table_lines(render_metrics_table_text([PRIOR]))[1]
    cells = [c.strip() for c in line.split("|")]
    assert cells[:4] == ["1m", "prior", "0.508 [0.49-0.53]", "-0.0011"]
    assert cells[5] == "57.1%"


def test_not_measurable_row_says_so_and_gives_the_reason():
    text = render_metrics_table_text([NOT_MEASURABLE])
    cells = [c.strip() for c in _table_lines(text)[1].split("|")]
    assert cells[:4] == ["1y", "prior", "not measurable", "n/a"]
    assert cells[4] == "n/a (0)"
    assert "1y not measurable: walkforward_folds(h=252)" in text


def test_prior_row_lists_the_ship_conditions_it_failed():
    reasons = ["AUC 0.508 < 0.53", "AUC CI low 0.489 <= 0.50", "Brier skill -0.0011 <= 0"]
    prior = {**PRIOR, "config_json": {"name": "run2", "measurable": True, "ship_reasons": reasons,
                                      "note": "; ".join(reasons)}}
    text = render_metrics_table_text([MODEL, prior])
    cells = [c.strip() for c in _table_lines(text)[2].split("|")]
    assert cells[:3] == ["1m", "prior", "0.508 [0.49-0.53]"]            # a measurement: shown
    assert "1m prior: AUC 0.508 < 0.53; AUC CI low 0.489 <= 0.50; Brier skill -0.0011 <= 0" in text
    assert "5d model:" not in text and "5d prior:" not in text          # a shipped row has no note


def test_confirm_folds_with_too_few_windows_read_not_measurable_even_with_an_auc():
    note = ("n_eff 4.1 < 10: too few independent 252-session windows to measure; "
            "AUC CI not available (the confirm folds hold too few bootstrap blocks)")
    thin = _row(252, status="prior", auc_mean=0.61, auc_ci_low=None, auc_ci_high=None,
                config_json={"name": "run2", "measurable": False, "note": note})
    text = render_metrics_table_text([thin])
    cells = [c.strip() for c in _table_lines(text)[1].split("|")]
    assert cells[:3] == ["1y", "prior", "not measurable"]
    assert f"1y not measurable: {note}" in text


def test_html_wraps_the_table_in_pre_and_escapes_everything():
    hostile = _row(63, status="prior", auc_mean=None, auc_ci_low=None, auc_ci_high=None,
                   config_json={"note": "<script>alert(1)</script> & more"})
    html = render_metrics_table_html([MODEL, hostile], total_seconds=432.4,
                                     failures=[("1y", "ValueError: <b>boom</b> & bust")])

    assert html.count("<pre>") == 1 and html.count("</pre>") == 1
    pre = re.search(r"<pre>(.*)</pre>", html, flags=re.S).group(1)
    assert "Horizon | Status | AUC [90% CI]" in pre
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>boom</b>" not in html and "&lt;b&gt;boom&lt;/b&gt; &amp; bust" in html
    assert "Brier skill &gt; 0 = probabilities beat the base rate" in html
    assert "training took 7m 12s" in html
    assert "<b>Failed:</b>" in html
    # Only the tags Telegram's HTML mode accepts.
    assert set(re.findall(r"</?([a-z]+)", html)) <= {"b", "i", "pre"}


def test_empty_rows_render_a_placeholder():
    assert render_metrics_table_text([]) == "No model metrics recorded yet."
    html = render_metrics_table_html([], total_seconds=5.0, failures=[("panel", "no bars")])
    assert "No model metrics recorded yet." in html
    assert "panel: no bars" in html
