"""
CaseLine — Access Log Correlation Module.

Cross-references physical access control events (badge swipes, door entries,
card readers) with video-based person tracking to:

  1. Correlate badge events with detected persons near entry/exit points
  2. Identify unauthorized access (person detected but no badge swipe)
  3. Identify ghost swipes (badge swipe but no person detected)
  4. Build a unified timeline merging access and video evidence
"""

import os
import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class AccessEvent:
    """Represents a single access control event."""
    timestamp: float          # Seconds from video start (or Unix epoch)
    badge_id: str             # Badge/card identifier
    access_point: str         # Door/gate/reader name
    event_type: str           # ENTRY, EXIT, DENIED, TAILGATE
    granted: bool = True      # Whether access was granted
    holder_name: str = ""     # Optional: badge holder name


@dataclass
class CorrelationResult:
    """Result of correlating an access event with video evidence."""
    access_event: AccessEvent
    camera_id: str
    matched_track_id: Optional[int] = None
    match_confidence: float = 0.0
    time_delta: float = 0.0          # Seconds between badge swipe and person detection
    correlation_type: str = "MATCH"  # MATCH, UNMATCHED_BADGE, UNMATCHED_PERSON, GHOST_SWIPE
    notes: str = ""


class AccessLogParser:
    """
    Parses access control CSV logs into structured AccessEvent objects.

    Supports flexible CSV formats with automatic column detection.
    """

    # Common column name variants to detect
    TIMESTAMP_COLS = ["timestamp", "time", "datetime", "event_time", "swipe_time", "access_time"]
    BADGE_COLS = ["badge_id", "card_id", "badge", "card", "employee_id", "user_id", "id"]
    POINT_COLS = ["access_point", "door", "reader", "gate", "location", "entry_point"]
    TYPE_COLS = ["event_type", "type", "action", "access_type", "direction"]
    GRANTED_COLS = ["granted", "allowed", "authorized", "status"]
    NAME_COLS = ["holder_name", "name", "employee_name", "user_name", "person"]

    def __init__(self, video_start_time: Optional[str] = None):
        """
        Parameters
        ----------
        video_start_time : str or None
            ISO format string of when the video recording started.
            Used to convert absolute timestamps to video-relative seconds.
            If None, timestamps are assumed to already be in seconds.
        """
        self.video_start_time = None
        if video_start_time:
            try:
                self.video_start_time = datetime.fromisoformat(video_start_time)
            except (ValueError, TypeError):
                pass

    def _find_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """Find a matching column name from candidates."""
        cols_lower = {c.lower().strip(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        return None

    def parse(self, csv_path: str) -> List[AccessEvent]:
        """
        Parse an access log CSV file into AccessEvent objects.

        Automatically detects column names from common variants.
        """
        if not os.path.exists(csv_path):
            print(f"[AccessLogParser] File not found: {csv_path}")
            return []

        df = pd.read_csv(csv_path)
        if df.empty:
            return []

        columns = df.columns.tolist()

        # Auto-detect columns
        ts_col = self._find_column(columns, self.TIMESTAMP_COLS)
        badge_col = self._find_column(columns, self.BADGE_COLS)
        point_col = self._find_column(columns, self.POINT_COLS)
        type_col = self._find_column(columns, self.TYPE_COLS)
        granted_col = self._find_column(columns, self.GRANTED_COLS)
        name_col = self._find_column(columns, self.NAME_COLS)

        if not ts_col:
            print(f"[AccessLogParser] No timestamp column found in: {columns}")
            return []

        events = []
        for _, row in df.iterrows():
            # Parse timestamp
            raw_ts = row[ts_col]
            timestamp = self._parse_timestamp(raw_ts)

            badge_id = str(row[badge_col]) if badge_col else "UNKNOWN"
            access_point = str(row[point_col]) if point_col else "UNKNOWN"
            event_type = str(row[type_col]).upper() if type_col else "ENTRY"

            granted = True
            if granted_col:
                val = row[granted_col]
                if isinstance(val, bool):
                    granted = val
                elif isinstance(val, str):
                    granted = val.lower() in ("true", "yes", "granted", "1", "allowed")
                elif isinstance(val, (int, float)):
                    granted = bool(val)

            holder_name = str(row[name_col]) if name_col and pd.notna(row.get(name_col)) else ""

            events.append(AccessEvent(
                timestamp=timestamp,
                badge_id=badge_id,
                access_point=access_point,
                event_type=event_type,
                granted=granted,
                holder_name=holder_name,
            ))

        events.sort(key=lambda e: e.timestamp)
        print(f"[AccessLogParser] Parsed {len(events)} access events from {csv_path}")
        return events

    def _parse_timestamp(self, raw_ts) -> float:
        """Convert a raw timestamp value to seconds (video-relative if possible)."""
        # If already numeric, treat as seconds
        if isinstance(raw_ts, (int, float)):
            return float(raw_ts)

        raw_str = str(raw_ts).strip()

        # Try ISO format parsing
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%H:%M:%S",
            "%M:%S",
        ]:
            try:
                dt = datetime.strptime(raw_str, fmt)
                if self.video_start_time and dt.year > 2000:
                    return (dt - self.video_start_time).total_seconds()
                # If only time (no date), return seconds from midnight
                if dt.year == 1900:  # strptime default year for time-only
                    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
                return dt.timestamp()
            except ValueError:
                continue

        # Last resort: try float conversion
        try:
            return float(raw_str)
        except ValueError:
            return 0.0


class AccessLogCorrelator:
    """
    Cross-references access control events with video person tracking data
    to produce a unified, correlated evidence timeline.
    """

    def __init__(self, time_window: float = 15.0):
        """
        Parameters
        ----------
        time_window : float
            Maximum seconds between a badge swipe and person detection
            to consider them correlated.
        """
        self.time_window = time_window

    def correlate(
        self,
        access_events: List[AccessEvent],
        tracking_csv: str,
        camera_id: str,
        camera_access_points: Optional[Dict[str, str]] = None,
    ) -> List[CorrelationResult]:
        """
        Correlate access events with person tracking data from a camera.

        Parameters
        ----------
        access_events : list of AccessEvent
            Parsed access control events.
        tracking_csv : str
            Path to the person tracking CSV for a camera.
        camera_id : str
            Camera identifier.
        camera_access_points : dict or None
            Mapping of camera_id -> access_point name.
            If provided, only events matching this camera's access point
            are considered.

        Returns
        -------
        list of CorrelationResult
        """
        if not os.path.exists(tracking_csv):
            return []

        trk = pd.read_csv(tracking_csv)
        if trk.empty:
            return []

        # Filter events relevant to this camera's access point
        relevant_events = access_events
        if camera_access_points and camera_id in camera_access_points:
            target_point = camera_access_points[camera_id]
            relevant_events = [
                e for e in access_events
                if e.access_point.lower() == target_point.lower()
            ]

        # Build a lookup of person appearances by timestamp
        track_appearances = {}
        for track_id in trk["track_id"].unique():
            track_data = trk[trk["track_id"] == track_id]
            track_appearances[int(track_id)] = {
                "first_seen": track_data["timestamp"].min(),
                "last_seen": track_data["timestamp"].max(),
                "observations": len(track_data),
            }

        results = []
        matched_tracks = set()

        # Match each access event to nearest person detection
        for event in relevant_events:
            best_track = None
            best_delta = float("inf")

            for track_id, appearance in track_appearances.items():
                if track_id in matched_tracks:
                    continue

                # Check if person appeared within the time window of the badge swipe
                delta_first = abs(event.timestamp - appearance["first_seen"])
                delta_last = abs(event.timestamp - appearance["last_seen"])
                min_delta = min(delta_first, delta_last)

                # Also check if the event falls within the track's time span
                if appearance["first_seen"] <= event.timestamp <= appearance["last_seen"]:
                    min_delta = 0.0

                if min_delta <= self.time_window and min_delta < best_delta:
                    best_delta = min_delta
                    best_track = track_id

            if best_track is not None:
                matched_tracks.add(best_track)
                confidence = max(0.0, 1.0 - (best_delta / self.time_window))
                results.append(CorrelationResult(
                    access_event=event,
                    camera_id=camera_id,
                    matched_track_id=best_track,
                    match_confidence=round(confidence, 3),
                    time_delta=round(best_delta, 2),
                    correlation_type="MATCH",
                    notes=f"Badge {event.badge_id} correlated with Track {best_track} "
                          f"(Δ{best_delta:.1f}s)",
                ))
            else:
                # Ghost swipe: badge event but no person detected
                results.append(CorrelationResult(
                    access_event=event,
                    camera_id=camera_id,
                    matched_track_id=None,
                    match_confidence=0.0,
                    time_delta=0.0,
                    correlation_type="GHOST_SWIPE",
                    notes=f"Badge {event.badge_id} swiped at {event.access_point} "
                          f"but no person detected on {camera_id}",
                ))

        # Check for unmatched persons (detected but no badge swipe)
        for track_id, appearance in track_appearances.items():
            if track_id not in matched_tracks:
                results.append(CorrelationResult(
                    access_event=AccessEvent(
                        timestamp=appearance["first_seen"],
                        badge_id="NONE",
                        access_point="UNKNOWN",
                        event_type="UNAUTHORIZED",
                        granted=False,
                    ),
                    camera_id=camera_id,
                    matched_track_id=track_id,
                    match_confidence=0.0,
                    time_delta=0.0,
                    correlation_type="UNMATCHED_PERSON",
                    notes=f"Track {track_id} detected on {camera_id} with no corresponding "
                          f"badge swipe ({appearance['first_seen']:.1f}s - {appearance['last_seen']:.1f}s)",
                ))

        results.sort(key=lambda r: r.access_event.timestamp)
        return results

    def save_correlation_report(self, results: List[CorrelationResult], output_csv: str):
        """Save correlation results to a CSV file."""
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        rows = []
        for r in results:
            rows.append({
                "timestamp": r.access_event.timestamp,
                "camera_id": r.camera_id,
                "badge_id": r.access_event.badge_id,
                "holder_name": r.access_event.holder_name,
                "access_point": r.access_event.access_point,
                "event_type": r.access_event.event_type,
                "granted": r.access_event.granted,
                "matched_track_id": r.matched_track_id if r.matched_track_id else "",
                "match_confidence": r.match_confidence,
                "time_delta_sec": r.time_delta,
                "correlation_type": r.correlation_type,
                "notes": r.notes,
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"[AccessLogCorrelator] Saved {len(rows)} correlation records → {output_csv}")
        return df

    def generate_summary(self, results: List[CorrelationResult]) -> Dict:
        """
        Generate a high-level summary of access log correlation.

        Returns
        -------
        dict with keys:
            total_events, matched, ghost_swipes, unmatched_persons,
            denied_entries, avg_match_confidence, flagged_events
        """
        matched = [r for r in results if r.correlation_type == "MATCH"]
        ghosts = [r for r in results if r.correlation_type == "GHOST_SWIPE"]
        unmatched = [r for r in results if r.correlation_type == "UNMATCHED_PERSON"]
        denied = [r for r in results if not r.access_event.granted]

        flagged = ghosts + unmatched + denied

        avg_conf = (
            sum(r.match_confidence for r in matched) / len(matched)
            if matched else 0.0
        )

        return {
            "total_events": len(results),
            "matched": len(matched),
            "ghost_swipes": len(ghosts),
            "unmatched_persons": len(unmatched),
            "denied_entries": len(denied),
            "avg_match_confidence": round(avg_conf, 3),
            "flagged_events": len(flagged),
            "flags": [
                {"type": r.correlation_type, "notes": r.notes}
                for r in flagged
            ],
        }
