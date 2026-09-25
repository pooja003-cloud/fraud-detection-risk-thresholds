"""Operational simulation of the alert workflow (added after the independent review).

The simple cost model in costs.py assumes every alerted fraud is stopped. In
the triage workflow only the AUTO-HOLD band stops a transaction at
authorisation. Review-band alerts are worked AFTER authorisation, so the
alerted transaction itself is lost. What a review can prevent is the REST of
the fraud burst: once an analyst confirms fraud, the card is blocked.

Rules of the simulation (per card, in time order):
  * hold   = score >= t_hold               -> transaction held; if fraud, customer denies -> card blocked at once
  * review = t_review <= score < t_hold    -> transaction goes through; if fraud, card blocked after `delay_hours`
  * a confirmed fraud blocks the compromised card: further fraud on that card within `episode_days`
    is prevented, and genuine spending is declined for `reissue_days` until the replacement card arrives
    (a later, separate compromise of the same customer is treated as a new episode)
  * analysts and customers are assumed to disposition correctly (genuine alerts never block a card)

Costs:
  missed-fraud loss (amount x (1 - recovery)) + reviews x review_cost + holds x hold_contact_cost
  + wrongly held genuine transactions x wrong_hold_cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

HOLD_CONTACT_COST = 1.0      # automated confirmation (SMS/app) plus occasional agent handling, $ per hold
WRONG_HOLD_COST = 10.0       # friction per genuine transaction held (goodwill, abandonment, complaint risk), $
REVIEW_DELAY_HOURS = 12.0    # average time from alert to analyst disposition (mix of same-day and 24h SLAs)
EPISODE_DAYS = 7.0           # a block stops further fraud on the compromised card for this long
REISSUE_DAYS = 3.0           # genuine transactions are declined until the replacement card arrives


def _last_event_before(d: pd.DataFrame, fraud, hold, review, delay_hours):
    """For each transaction, the time of the most recent blocking event on the same card strictly before it."""
    event_time = np.where(hold, d["ts"].to_numpy(),
                          (d["ts"] + pd.Timedelta(hours=delay_hours)).to_numpy()).astype("datetime64[ns]")
    ev = pd.DataFrame({"cc_num": d["cc_num"].to_numpy(), "t": event_time})[fraud & (hold | review)].sort_values("t")
    left = pd.DataFrame({"cc_num": d["cc_num"].to_numpy(), "ts": d["ts"].to_numpy().astype("datetime64[ns]"),
                         "i": np.arange(len(d))}).sort_values("ts")
    m = pd.merge_asof(left, ev.rename(columns={"t": "ts_ev"}).assign(ts=lambda x: x.ts_ev).sort_values("ts"),
                      on="ts", by="cc_num", direction="backward", allow_exact_matches=False)
    last = np.empty(len(d), dtype="datetime64[ns]"); last[:] = np.datetime64("NaT")
    last[m["i"].to_numpy()] = m["ts_ev"].to_numpy()
    return last, ev


def simulate(df: pd.DataFrame, score_col: str, t_hold: float, t_review: float,
             delay_hours: float = REVIEW_DELAY_HOURS, episode_days: float = EPISODE_DAYS, reissue_days: float = REISSUE_DAYS,
             review_cost: float = config.REVIEW_COST,
             hold_contact_cost: float = HOLD_CONTACT_COST, wrong_hold_cost: float = WRONG_HOLD_COST,
             recovery_rate: float = config.RECOVERY_RATE, per_card: bool = False):
    """Return a dict of outcomes (and optionally a per-card frame for bootstrapping)."""
    d = df[["cc_num", "ts", "amt", "is_fraud", score_col]]
    s = d[score_col].to_numpy()
    fraud = d["is_fraud"].to_numpy() == 1
    hold = s >= t_hold
    review = (s >= t_review) & ~hold
    last, ev = _last_event_before(d, fraud, hold, review, delay_hours)
    ts = d["ts"].to_numpy().astype("datetime64[ns]")
    since = ts - last
    has = ~np.isnat(last)
    declined = has & np.where(fraud, since <= np.timedelta64(int(episode_days * 86400), "s"),
                              since <= np.timedelta64(int(reissue_days * 86400), "s"))
    held = hold & ~declined
    reviewed = review & ~declined
    prevented = fraud & (held | declined)
    lost = fraud & ~prevented
    wrong_hold = held & ~fraud
    genuine_declined = declined & ~fraud
    amt = d["amt"].to_numpy()
    loss = amt * lost * (1 - recovery_rate)
    cost = loss + reviewed * review_cost + held * hold_contact_cost + wrong_hold * wrong_hold_cost
    n_days = d["ts"].dt.normalize().nunique()
    out = {
        "t_hold": t_hold, "t_review": t_review, "delay_hours": delay_hours,
        "holds_per_day": held.sum() / n_days, "reviews_per_day": reviewed.sum() / n_days,
        "wrong_holds": int(wrong_hold.sum()), "genuine_declined_after_block": int(genuine_declined.sum()),
        "frauds": int(fraud.sum()), "frauds_prevented": int(prevented.sum()),
        "fraud_prevented_pct": prevented.sum() / max(fraud.sum(), 1),
        "fraud_usd": float(amt[fraud].sum()), "fraud_usd_lost": float(amt[lost].sum()),
        "fraud_usd_prevented_pct": float(amt[prevented].sum() / max(amt[fraud].sum(), 1e-9)),
        "card_blocks": int((ev.groupby("cc_num")["t"].diff().isna() | (ev.groupby("cc_num")["t"].diff() > pd.Timedelta(days=episode_days))).sum()),
        "review_precision": float((reviewed & fraud).sum() / max(reviewed.sum(), 1)),
        "loss_usd": float(loss.sum()), "review_cost_usd": float(reviewed.sum() * review_cost),
        "hold_cost_usd": float(held.sum() * hold_contact_cost + wrong_hold.sum() * wrong_hold_cost),
        "total_cost_usd": float(cost.sum()),
    }
    if per_card:
        pc = pd.DataFrame({"cc_num": d["cc_num"].to_numpy(), "cost": cost, "loss": loss}).groupby("cc_num").sum()
        return out, pc
    return out


def daily_workload(df: pd.DataFrame, score_col: str, t_hold: float, t_review: float,
                   delay_hours: float = REVIEW_DELAY_HOURS, episode_days: float = EPISODE_DAYS,
                   reissue_days: float = REISSUE_DAYS) -> pd.DataFrame:
    """Per-day counts of holds, reviews and fraud confirmations (for peak-day staffing)."""
    d = df[["cc_num", "ts", "amt", "is_fraud", score_col]].reset_index(drop=True)
    s = d[score_col].to_numpy()
    fraud = d["is_fraud"].to_numpy() == 1
    hold = s >= t_hold
    review = (s >= t_review) & ~hold
    last, ev = _last_event_before(d, fraud, hold, review, delay_hours)
    ts = d["ts"].to_numpy().astype("datetime64[ns]")
    since = ts - last
    has = ~np.isnat(last)
    declined = has & np.where(fraud, since <= np.timedelta64(int(episode_days * 86400), "s"),
                              since <= np.timedelta64(int(reissue_days * 86400), "s"))
    out = pd.DataFrame({"day": d["ts"].dt.normalize(), "hour": d["ts"].dt.hour,
                        "held": hold & ~declined, "reviewed": review & ~declined})
    daily = out.groupby("day").agg(holds=("held", "sum"), reviews=("reviewed", "sum"))
    new_block = ev.groupby("cc_num")["t"].diff().isna() | (ev.groupby("cc_num")["t"].diff() > pd.Timedelta(days=episode_days))
    conf = pd.Series(ev.loc[new_block.values, "t"]).dt.normalize().value_counts().rename("fraud_confirmations")
    daily = daily.join(conf, how="left").fillna(0)
    hourly_holds = out[out.held].groupby("hour").size().reindex(range(24), fill_value=0)
    return daily, hourly_holds


def optimise_review_threshold(df: pd.DataFrame, score_col: str, t_hold: float, grid, **kw) -> pd.DataFrame:
    rows = [simulate(df, score_col, t_hold, t, **kw) for t in grid]
    return pd.DataFrame(rows)


def card_bootstrap_saving(pc_a: pd.DataFrame, pc_b: pd.DataFrame, n_boot: int = 1000, seed: int = config.SEED):
    """95% interval for total cost(a) - total cost(b), resampling cards (fraud bursts stay together)."""
    both = pc_a[["cost"]].join(pc_b[["cost"]], lsuffix="_a", rsuffix="_b", how="outer").fillna(0)
    diff = (both["cost_a"] - both["cost_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    sims = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(n_boot)])
    return float(diff.sum()), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))
