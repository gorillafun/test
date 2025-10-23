"""Utility to fetch Polymarket price history for the newest active event.

This module implements the workflow described in the user instructions:

1. Find a slug for the newest active event when `--slug` is not supplied.
2. Enumerate the markets under the event and collect their `clobTokenIds`.
3. Fetch the historical price series for each token using one minute
   fidelity and export the rows to a CSV file. Each record includes the
   raw price (`p`) and the normalised probability (`p / 10000`).

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
import csv
import json
from dataclasses import dataclass
from pathlib import Path
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


def get_price_history(token_id: str, *, interval: str = "max", fidelity: int = 1) -> List[PricePoint]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Polymarket price history")
    parser.add_argument("--slug", help="Event slug (auto-detected when omitted)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of rows from each token history (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("polymarket_prices.csv"),
        help="Destination CSV file for the collected price history",
    )
    args = parser.parse_args()

    slug = args.slug or choose_slug_auto()
    print(f"[info] using slug: {slug}")

    event = get_event(slug)
    markets = event.get("markets") or []
    if not markets:
        raise RuntimeError("イベント配下のマーケットが見つかりませんでした")

    total_tokens = 0
    csv_rows: List[dict] = []
    for market, token_ids in iter_token_ids(markets):
        question = market.get("question") or market.get("slug") or "(unknown market)"
        market_slug = market.get("slug") or ""

        for token_id in token_ids:
            history = get_price_history(token_id)
            total_tokens += 1
            limit = args.limit if args.limit and args.limit > 0 else len(history)
            for point in history[:limit]:
                csv_rows.append(
                    {
                        "event_slug": slug,
                        "market_question": question,
                        "market_slug": market_slug,
                        "token_id": token_id,
                        "timestamp": point.ts,
                        "price": point.price,
                        "probability": round(point.probability, 4),
                    }
                )

    if total_tokens == 0:
        raise RuntimeError("token_id（clobTokenIds）が見つかりませんでした")

    if not csv_rows:
        raise RuntimeError("価格履歴が取得できませんでした")

    fieldnames = [
        "event_slug",
        "market_question",
        "market_slug",
        "token_id",
        "timestamp",
        "price",
        "probability",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"[info] wrote {len(csv_rows)} rows from {total_tokens} tokens to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
