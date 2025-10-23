"""Utility to fetch Polymarket price history for the newest active event.

This module implements the workflow described in the user instructions:

1. Find a slug for the newest active event when `--slug` is not supplied.
2. Enumerate the markets under the event and collect their `clobTokenIds`.
3. Fetch the historical price series for each token and print the first
   few samples, showing both the raw price (`p`) and the normalised
   probability (`p / 10000`).

Example usage
-------------

    # Auto-select the newest event
    python -m src.polymarket_fetch_simple

    # Specify the slug explicitly
    python -m src.polymarket_fetch_simple --slug fed-decision-in-october

The script is intentionally minimal: HTTP failures raise directly via
``requests.Response.raise_for_status`` and only basic error handling is
performed when the expected data cannot be found.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, List

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def _create_session() -> requests.Session:
    """Return a requests session that ignores proxy env vars.

    The execution environment may inject proxy settings that block
    outbound traffic. Setting ``trust_env`` to ``False`` disables those
    automatically so the script can connect directly to the Polymarket
    endpoints.
    """

    session = requests.Session()
    session.trust_env = False
    return session


SESSION = _create_session()


@dataclass
class PricePoint:
    """Single entry in the price history response."""

    ts: int
    price: float
    probability: float


def choose_slug_auto() -> str:
    """Return the slug of the newest active event.

    The API call retrieves a single event ordered by `id` descending with
    `closed=false` to ensure only active events are considered.
    """

    params = {
        "closed": "false",
        "order": "id",
        "ascending": "false",
        "limit": 1,
    }
    response = SESSION.get(f"{GAMMA}/events", params=params, timeout=20)
    response.raise_for_status()
    items = response.json()
    if not items:
        raise RuntimeError("アクティブなイベントが見つかりませんでした")

    slug = items[0].get("slug")
    if not slug:
        raise RuntimeError("イベントのslugが見つかりませんでした")
    return slug


def get_event(slug: str) -> dict:
    """Fetch event metadata for the given slug."""

    response = SESSION.get(f"{GAMMA}/events/slug/{slug}", timeout=20)
    response.raise_for_status()
    return response.json()


def get_market_by_slug(slug: str) -> dict:
    """Fetch market metadata for the given market slug."""

    response = SESSION.get(f"{GAMMA}/markets/slug/{slug}", timeout=20)
    response.raise_for_status()
    return response.json()


def parse_token_ids(raw) -> List[str]:
    """Normalise the clobTokenIds field to a list of strings."""

    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def get_price_history(token_id: str, *, interval: str = "max", fidelity: int = 60) -> List[PricePoint]:
    """Retrieve the price history for a token id."""

    params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    response = SESSION.get(f"{CLOB}/prices-history", params=params, timeout=30)
    response.raise_for_status()

    history = response.json().get("history", [])
    points = []
    for entry in history:
        price = float(entry["p"])
        probability = price / 10000.0
        points.append(
            PricePoint(
                ts=int(entry["t"]),
                price=price,
                probability=probability,
            )
        )
    return points


def iter_token_ids(markets: Iterable[dict]) -> Iterable[tuple[dict, List[str]]]:
    """Yield markets paired with any token ids found for them."""

    for market in markets:
        token_ids = parse_token_ids(market.get("clobTokenIds"))
        if not token_ids and market.get("slug"):
            market_detail = get_market_by_slug(market["slug"])
            token_ids = parse_token_ids(market_detail.get("clobTokenIds"))
        if token_ids:
            yield market, token_ids


def format_price_point(point: PricePoint) -> str:
    """Return a formatted string for a price point."""

    return f"    ts={point.ts} p={point.price:.2f} prob={point.probability:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Polymarket price history")
    parser.add_argument("--slug", help="Event slug (auto-detected when omitted)")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows from each token history to display",
    )
    args = parser.parse_args()

    slug = args.slug or choose_slug_auto()
    print(f"[info] using slug: {slug}")

    event = get_event(slug)
    markets = event.get("markets") or []
    if not markets:
        raise RuntimeError("イベント配下のマーケットが見つかりませんでした")

    total_tokens = 0
    for market, token_ids in iter_token_ids(markets):
        question = market.get("question") or market.get("slug") or "(unknown market)"
        print(f"\n[market] {question} :: token_ids={token_ids}")

        for token_id in token_ids:
            history = get_price_history(token_id)
            total_tokens += 1
            print(f"  [token] {token_id}  points={len(history)}")
            for point in history[: args.limit]:
                print(format_price_point(point))

    if total_tokens == 0:
        raise RuntimeError("token_id（clobTokenIds）が見つかりませんでした")


if __name__ == "__main__":
    main()
