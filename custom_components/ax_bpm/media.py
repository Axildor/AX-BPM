"""media_player identity helpers shared by the BPM and mood sensors.

Kept in a NEUTRAL module (not sensor.py) because both entities need them and
sensor.py already imports mood_sensor.py — importing the other way round
would be circular.

Why this module exists (stale-BPM fix)
--------------------------------------
Track-change detection previously compared only
``(media_artist, media_title, media_duration)``. A player that reports the
same metadata for consecutive tracks froze that tuple, so:
  - the debounce gate never fired (``if track_id == self._last_track:
    return``), and
  - the "did the track change while resolving?" guard
    (``self._resolving_track != self._current_track_id()``) compared equal
    and let a PREVIOUS track's result publish under the NEW track.

Both defects share one root cause: the identity was inferred from
free-text metadata that a player is free to keep constant. This module
adds the two signals that actually work when metadata does not move.
"""

from __future__ import annotations

#: Position (seconds) at or below which a playing item counts as "just
#: started". A rewind to this point while the metadata is unchanged is
#: treated as the next track starting, not as a user seek.
RESTART_POSITION_MAX = 2.0

#: A position above this must precede a restart for the restart to count.
#: Prevents a small backwards seek near the start of a track from bumping
#: the generation and re-triggering a full resolution.
RESTART_POSITION_MIN = 5.0


def track_identity(state) -> tuple:
    """Stable identity for the currently playing item.

    Both available signals are included, so a change in EITHER counts as a
    change:

    - ``media_content_id`` is the player's real per-item identity. Some
      players expose it and keep it accurate even when they report no
      title; others reuse one id per album/queue.
    - ``(media_artist, media_title, media_duration)`` is the fallback the
      integration used exclusively before this fix.

    Combining them is strictly safer than preferring one: a player that
    recycles content ids still triggers on a title change, and a player
    that freezes metadata still triggers on a content-id change.

    Returns a tuple ending in a ``"meta"`` marker so an identity built
    with a content id can never collide with one built without it.
    """
    if state is None:
        return ("none",)

    attrs = state.attributes
    parts: list[tuple] = []

    content_id = attrs.get("media_content_id")
    if content_id:
        parts.append(("id", str(content_id)))

    parts.append(
        (
            "meta",
            attrs.get("media_artist"),
            attrs.get("media_title"),
            attrs.get("media_duration"),
        )
    )
    return tuple(parts)


def media_position(state) -> float | None:
    """Playback position in seconds, or None when the player omits it."""
    if state is None:
        return None
    position = state.attributes.get("media_position")
    if isinstance(position, (int, float)):
        return float(position)
    return None


def looks_restarted(previous: float | None, current: float | None) -> bool:
    """True when playback clearly started over on the same metadata.

    This is the case ``track_identity`` cannot see: a player that keeps
    artist/title/duration AND content id constant across tracks still
    resets its position to ~0 when the next item begins. Requiring the
    previous position to be well past the start keeps an ordinary
    backwards seek from looking like a track change.
    """
    if previous is None or current is None:
        return False
    return current <= RESTART_POSITION_MAX and previous > RESTART_POSITION_MIN


def media_duration(state) -> float | None:
    """Track duration in seconds, or None when the player omits it.

    The stale watchdog is armed ONLY when this returns a positive number:
    a live stream has no expected end, so "the track should have finished"
    is unknowable there and holding the BPM is the correct behaviour.
    """
    if state is None:
        return None
    duration = state.attributes.get("media_duration")
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration)
    return None
