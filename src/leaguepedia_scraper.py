"""
Leaguepedia Roster Scraper
==========================
Fetches current team rosters from the Leaguepedia Cargo API and writes
them to data/roster_overrides.json, which the predictor loads before
falling back to the last-game CSV roster.

Usage:
    python src/leaguepedia_scraper.py              # Fetch LEC (default)
    python src/leaguepedia_scraper.py --leagues LEC PRM LFL
    python src/leaguepedia_scraper.py --test       # Single API probe, no file write

API docs: https://lol.fandom.com/wiki/Help:Leaguepedia_SemQuery
"""

import requests
import json
import time
import os
import sys
import re
import unicodedata
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CARGO_URL = "https://lol.fandom.com/api.php"
RATE_LIMIT_SECONDS = 2.0   # polite delay between requests
MAX_RETRIES = 4

HEADERS = {
    "User-Agent": (
        "LOL5-RosterScraper/1.0 (private research tool; "
        "contact: roster-bot@localhost)"
    ),
    "Accept": "application/json",
}

# Map league code (as used in the model) → Leaguepedia tournament page prefix
# These are the "Tournamentpage" values in TournamentRosters that we filter on.
LEAGUE_TOURNAMENT_MAP = {
    "LEC": ["LEC/2026 Season/Spring Season", "LEC/2026 Season/Spring Playoffs"],
    "PRM": ["Prime League/2026 Spring Season", "Prime League/2026 Spring Playoffs"],
    "LFL": ["LFL/2026 Spring Season", "LFL Division 2/2026 Spring Season"],
    "NLC": ["NLC/2026 Spring Season"],
}

DEFAULT_LEAGUES = ["LEC"]

# Output path (relative to the script's parent directory)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "roster_overrides.json"
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _cargo_request(params: dict, retries: int = MAX_RETRIES) -> dict | None:
    """Make a Cargo API request with retry/backoff."""
    params.setdefault("action", "cargoquery")
    params.setdefault("format", "json")
    params.setdefault("limit", "50")

    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            resp = requests.get(CARGO_URL, params=params, headers=HEADERS, timeout=15)
            data = resp.json()
            if "error" in data:
                code = data["error"].get("code", "")
                if code == "ratelimited":
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s before retry {attempt+1}…")
                    time.sleep(wait)
                    continue
                print(f"  API error: {data['error'].get('info', data['error'])}")
                return None
            return data
        except Exception as exc:
            print(f"  Request error (attempt {attempt+1}): {exc}")
            time.sleep(2 * (attempt + 1))

    print("  All retries exhausted.")
    return None


# ---------------------------------------------------------------------------
# Leaguepedia queries
# ---------------------------------------------------------------------------

def fetch_tournament_rosters(tournament_page: str) -> list[dict]:
    """
    Query TournamentRosters for a specific tournament page.
    Returns a list of dicts with keys: team, top, jng, mid, bot, sup.
    """
    params = {
        "tables": "TournamentRosters",
        "fields": "Team, Top, Jungle, Mid, Bot, Support, OverviewPage",
        "where": f'OverviewPage = "{tournament_page}"',
        "limit": "50",
    }
    data = _cargo_request(params)
    if not data:
        return []

    results = []
    for row in data.get("cargoquery", []):
        r = row.get("title", {})
        team = r.get("Team", "").strip()
        if not team:
            continue
        results.append({
            "team": team,
            "top": (r.get("Top") or "").strip(),
            "jng": (r.get("Jungle") or "").strip(),
            "mid": (r.get("Mid") or "").strip(),
            "bot": (r.get("Bot") or "").strip(),
            "sup": (r.get("Support") or "").strip(),
            "source_tournament": r.get("OverviewPage", tournament_page),
        })

    return results


def fetch_scoreboard_players(tournament_page: str) -> list[dict]:
    """
    Fallback: derive rosters from ScoreboardPlayers when TournamentRosters
    has no entries yet. Uses the most recent game's lineup for each team.
    
    Returns same format as fetch_tournament_rosters.
    """
    params = {
        "tables": "ScoreboardPlayers",
        "fields": "Team, Name, Role, OverviewPage",
        "where": f'OverviewPage = "{tournament_page}"',
        "order_by": "ScoreboardPlayers._pageName",
        "limit": "500",
    }
    data = _cargo_request(params)
    if not data:
        return []

    # Collect all entries per team → keep last appearance per role
    teams: dict[str, dict] = {}
    role_map = {
        "Top": "top", "Jungle": "jng", "Mid": "mid", "Bot": "bot", "Support": "sup",
        "top": "top", "jng": "jng", "mid": "mid", "bot": "bot", "sup": "sup",
    }
    for row in data.get("cargoquery", []):
        r = row.get("title", {})
        team = (r.get("Team") or "").strip()
        name = (r.get("Name") or "").strip()
        role_raw = (r.get("Role") or "").strip()
        role = role_map.get(role_raw, role_raw.lower())
        if team and name and role in {"top", "jng", "mid", "bot", "sup"}:
            if team not in teams:
                teams[team] = {"team": team, "top": "", "jng": "", "mid": "", "bot": "", "sup": ""}
            teams[team][role] = name

    results = []
    for t in teams.values():
        if any(t[p] for p in ("top", "jng", "mid", "bot", "sup")):
            t["source_tournament"] = tournament_page
            results.append(t)
    return results


def fetch_league_rosters(league_code: str) -> list[dict]:
    """
    Fetch all team rosters for a given league code.
    Tries TournamentRosters first for each configured tournament page;
    falls back to ScoreboardPlayers if that returns nothing.
    """
    tournament_pages = LEAGUE_TOURNAMENT_MAP.get(league_code, [])
    if not tournament_pages:
        print(f"  No tournament page configured for league '{league_code}'")
        return []

    all_rosters: dict[str, dict] = {}  # team_name → roster dict

    for tournament_page in tournament_pages:
        print(f"  Querying TournamentRosters for: {tournament_page}")
        rosters = fetch_tournament_rosters(tournament_page)

        if not rosters:
            print(f"    No TournamentRosters entries — trying ScoreboardPlayers fallback…")
            rosters = fetch_scoreboard_players(tournament_page)

        if rosters:
            print(f"    Found {len(rosters)} team(s).")
            for r in rosters:
                team = r["team"]
                # Later tournament pages (e.g. playoffs) override regular season
                all_rosters[team] = r
        else:
            print(f"    No data found for this tournament page.")

    return list(all_rosters.values())


# ---------------------------------------------------------------------------
# Name normalization (for matching Leaguepedia names to Elixir team names)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, strip diacritics, remove non-alphanumeric for fuzzy matching."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Known name differences between Leaguepedia and Elixir team names.
# Add entries here if a team isn't being matched.
TEAM_NAME_ALIASES = {
    "Fnatic": "Fnatic",
    "G2 Esports": "G2 Esports",
    "Team Vitality": "Team Vitality",
    "Team BDS": "Team BDS",
    "SK Gaming": "SK Gaming",
    "Karmine Corp": "Karmine Corp",
    "Movistar KOI": "Movistar KOI",
    "FlyQuest EMEA": "FlyQuest EMEA",
    "Rogue": "Rogue",
}


# ---------------------------------------------------------------------------
# Override file management
# ---------------------------------------------------------------------------

def load_existing_overrides() -> dict:
    """Load existing roster_overrides.json if it exists."""
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_overrides(overrides: dict) -> None:
    """Write roster_overrides.json, creating directories if needed."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    print(f"\nWrote roster overrides → {OUTPUT_PATH}")


def print_diff(old: dict, new: dict) -> None:
    """Print a human-readable diff of roster changes."""
    positions = ["top", "jng", "mid", "bot", "sup"]
    all_teams = sorted(set(list(old.keys()) + list(new.keys())) - {"_meta"})
    changes = 0

    for team in all_teams:
        old_r = old.get(team, {})
        new_r = new.get(team, {})
        if old_r == new_r:
            continue
        print(f"\n  {team}:")
        for pos in positions:
            o = old_r.get(pos, "—")
            n = new_r.get(pos, "—")
            if o != n:
                print(f"    {pos.upper()}: {o:20s} → {n}")
                changes += 1

    if changes == 0:
        print("  No changes detected vs. previous override file.")
    else:
        print(f"\n  {changes} roster change(s) detected.")


# ---------------------------------------------------------------------------
# Main fetch routine
# ---------------------------------------------------------------------------

def fetch_and_save(leagues: list[str], dry_run: bool = False) -> dict:
    """
    Fetch rosters for the given leagues and update the override file.
    
    Args:
        leagues:  List of league codes, e.g. ['LEC']
        dry_run:  If True, print results but don't write to disk
    
    Returns:
        The updated overrides dict
    """
    existing = load_existing_overrides()
    updated = {k: v for k, v in existing.items() if k != "_meta"}

    for league in leagues:
        print(f"\n{'='*55}")
        print(f"  Fetching {league} rosters from Leaguepedia…")
        print(f"{'='*55}")
        rosters = fetch_league_rosters(league)

        if not rosters:
            print(f"  ⚠  No rosters returned for {league}. "
                  f"Check LEAGUE_TOURNAMENT_MAP config.")
            continue

        for r in rosters:
            team_name = r["team"]
            updated[team_name] = {
                "top": r["top"],
                "jng": r["jng"],
                "mid": r["mid"],
                "bot": r["bot"],
                "sup": r["sup"],
            }
            print(f"  {team_name}")
            for pos in ("top", "jng", "mid", "bot", "sup"):
                print(f"    {pos.upper()}: {r[pos] or '—'}")

    updated["_meta"] = {
        "fetched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "Leaguepedia Cargo API",
        "leagues": leagues,
    }

    print(f"\n{'='*55}")
    print("  ROSTER CHANGES vs. previous override file:")
    print_diff(existing, updated)

    if not dry_run:
        save_overrides(updated)
    else:
        print("\n  [DRY RUN — file not written]")

    return updated


# ---------------------------------------------------------------------------
# Test mode — single probe to verify API is reachable
# ---------------------------------------------------------------------------

def test_api() -> None:
    """Quick probe to verify API connectivity and field names."""
    print("Testing Leaguepedia Cargo API…")
    # Try a very simple query first
    params = {
        "tables": "TournamentRosters",
        "fields": "Team, Top, Jungle, Mid, Bot, Support, OverviewPage",
        "where": 'OverviewPage = "LEC/2026 Season/Spring Season"',
        "limit": "3",
    }
    data = _cargo_request(params)
    if data is None:
        print("✗ API call failed — check connectivity or rate limits.")
        return

    rows = data.get("cargoquery", [])
    print(f"✓ API reachable. Got {len(rows)} row(s) from TournamentRosters.")
    for row in rows:
        r = row.get("title", {})
        print(f"  Team: {r.get('Team')} | Top: {r.get('Top')} | "
              f"Jungle: {r.get('Jungle')} | Mid: {r.get('Mid')} | "
              f"Bot: {r.get('Bot')} | Support: {r.get('Support')}")

    if not rows:
        print("  No rows returned — tournament page name may need updating.")
        print("  Trying ScoreboardPlayers fallback…")
        params2 = {
            "tables": "ScoreboardPlayers",
            "fields": "Team, Name, Role, OverviewPage",
            "where": 'OverviewPage = "LEC/2026 Season/Spring Season"',
            "limit": "10",
        }
        data2 = _cargo_request(params2)
        if data2:
            rows2 = data2.get("cargoquery", [])
            print(f"  ScoreboardPlayers returned {len(rows2)} row(s).")
            for row2 in rows2[:5]:
                r2 = row2.get("title", {})
                print(f"    {r2.get('Team')} | {r2.get('Name')} | {r2.get('Role')}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        test_api()
        sys.exit(0)

    # Parse --leagues flag
    leagues = DEFAULT_LEAGUES
    if "--leagues" in args:
        idx = args.index("--leagues")
        leagues = []
        for a in args[idx + 1:]:
            if a.startswith("--"):
                break
            leagues.append(a)

    dry_run = "--dry-run" in args

    fetch_and_save(leagues, dry_run=dry_run)
