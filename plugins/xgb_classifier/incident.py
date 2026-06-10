# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
# Licensed under the NASA Open Source Agreement version 1.3
"""Incident aggregation for the OnAIR attack classifier (NOS3-201).

Collapses a hysteresis-confirmed run of anomalous frames into a single
*incident*: start/end frame, duration, dominant attack cluster + sub-technique,
and accumulated confidence. Operators care about incidents ("an anomaly began
at T, lasted N s, looked like a propulsion-command-class attack"), not a stream
of independent per-frame flags.

Pure-Python and framework-free on purpose: the live OnAIR plugin feeds it the
same per-frame stream it already computes, and the offline corpus re-score
(NOS3-203) drives the *identical* logic so live and offline incident metrics
agree by construction.

State machine mirrors the Isolation-Forest plugin's alert/clear hysteresis:
  - a run of >= `alert_hysteresis` consecutive anomalous frames *confirms* an
    incident (its start back-dates to the first frame of that run);
  - the incident stays open across short nominal gaps; it *closes* only after
    >= `clear_hysteresis` consecutive nominal frames;
  - a building run that dies before confirmation is discarded (no incident).

The winning cluster is the one with the greatest summed top-1 probability over
the incident's classified frames; `agreement` is the fraction of those frames
that voted for it (the real "how sure across the whole event" signal — a long
incident with 95 % agreement is far more trustworthy than any single 0.86
frame).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Incident:
    frame_start: int          # first frame of the confirmed anomaly run
    frame_end: int            # last anomalous frame in the incident
    n_frames: int             # frame_end - frame_start + 1
    n_anomaly_frames: int     # classified (anomalous) frames inside the span
    mode: str                 # ADCS mode at incident start
    cluster: str              # winning cluster (or sub-technique if unclustered)
    sub_technique: str        # winning sub-technique within the cluster
    confidence: float         # mean top-1 prob of winning-cluster frames
    agreement: float          # fraction of anomaly frames voting winning cluster
    peak_confidence: float    # max top-1 prob seen in the incident

    def as_row(self) -> list:
        return [
            self.frame_start, self.frame_end, self.n_frames,
            self.n_anomaly_frames, self.mode, self.cluster, self.sub_technique,
            f"{self.confidence:.6f}", f"{self.agreement:.6f}",
            f"{self.peak_confidence:.6f}",
        ]

    @staticmethod
    def header() -> list:
        return [
            "frame_start", "frame_end", "n_frames", "n_anomaly_frames",
            "mode", "cluster", "sub_technique", "confidence", "agreement",
            "peak_confidence",
        ]

    def to_dict(self) -> dict:
        return asdict(self)


class IncidentAggregator:
    """Fold a per-frame (is_anomaly, cluster, sub_technique, confidence) stream
    into discrete incidents. Call `update()` every frame; it returns an
    `Incident` on the frame that closes one, else `None`. Call `flush()` at
    end-of-stream to emit any still-open incident."""

    def __init__(self, alert_hysteresis: int = 3, clear_hysteresis: int = 5,
                 min_anomaly_frames: int = 1):
        self.alert_hyst = max(1, int(alert_hysteresis))
        self.clear_hyst = max(1, int(clear_hysteresis))
        self.min_anomaly_frames = max(1, int(min_anomaly_frames))
        self._consec_anom = 0
        self._consec_nom = 0
        self._confirmed = False
        self._reset_accum()

    def _reset_accum(self) -> None:
        self._f_start: int | None = None
        self._f_last_anom: int | None = None
        self._mode_at_start: str = ""
        self._n_anom = 0
        self._peak = 0.0
        self._votes: dict[str, float] = defaultdict(float)       # cluster -> Σ prob
        self._vote_n: dict[str, int] = defaultdict(int)          # cluster -> #frames
        self._sub_votes: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))                          # cluster -> sub -> Σ prob

    def _accumulate(self, frame_idx: int, mode: str, cluster: str,
                    sub_technique: str, confidence: float) -> None:
        if self._f_start is None:
            self._f_start = frame_idx
            self._mode_at_start = mode
        self._f_last_anom = frame_idx
        self._n_anom += 1
        key = cluster or sub_technique or "unknown"
        self._votes[key] += float(confidence)
        self._vote_n[key] += 1
        self._sub_votes[key][sub_technique or key] += float(confidence)
        if confidence > self._peak:
            self._peak = float(confidence)

    def update(self, frame_idx: int, is_anomaly: bool, mode: str = "",
               cluster: str = "", sub_technique: str = "",
               confidence: float = 0.0) -> Incident | None:
        closed: Incident | None = None
        if is_anomaly:
            self._consec_anom += 1
            self._consec_nom = 0
            # A brand-new building run resets pre-confirmation accumulation.
            if self._consec_anom == 1 and not self._confirmed:
                self._reset_accum()
            self._accumulate(frame_idx, mode, cluster, sub_technique, confidence)
            if not self._confirmed and self._consec_anom >= self.alert_hyst:
                self._confirmed = True
        else:
            self._consec_nom += 1
            self._consec_anom = 0
            if self._confirmed and self._consec_nom >= self.clear_hyst:
                closed = self._close()
            elif not self._confirmed:
                # Building run died before confirmation — discard.
                self._reset_accum()
        return closed

    def flush(self) -> Incident | None:
        """Close any open, confirmed incident at end-of-stream."""
        if self._confirmed:
            return self._close()
        return None

    def _close(self) -> Incident | None:
        confirmed = self._confirmed
        f_start, f_last, n_anom = self._f_start, self._f_last_anom, self._n_anom
        votes, vote_n, sub_votes = self._votes, self._vote_n, self._sub_votes
        mode = self._mode_at_start
        peak = self._peak
        # Reset state for the next incident regardless of outcome.
        self._confirmed = False
        self._consec_anom = 0
        self._consec_nom = 0
        self._reset_accum()

        if not confirmed or f_start is None or n_anom < self.min_anomaly_frames:
            return None
        # Winning cluster = greatest summed top-1 probability.
        win = max(votes, key=lambda c: votes[c])
        confidence = votes[win] / vote_n[win] if vote_n[win] else 0.0
        agreement = vote_n[win] / n_anom if n_anom else 0.0
        sub = max(sub_votes[win], key=lambda s: sub_votes[win][s]) if sub_votes[win] else win
        return Incident(
            frame_start=int(f_start),
            frame_end=int(f_last),
            n_frames=int(f_last - f_start + 1),
            n_anomaly_frames=int(n_anom),
            mode=mode,
            cluster=win,
            sub_technique=sub,
            confidence=float(confidence),
            agreement=float(agreement),
            peak_confidence=float(peak),
        )


def cluster_map_from_taxonomy(taxonomy: dict) -> dict[str, str]:
    """Build a {sub_technique: cluster_label} map from a cluster_taxonomy.json
    dict. Members of a multi-member cluster map to its representative label;
    every other class maps to itself."""
    out: dict[str, str] = {}
    for rep, members in (taxonomy.get("clusters") or {}).items():
        for m in members:
            out[m] = rep
    return out
