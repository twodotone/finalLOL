"""
gol.gg LPL Scraper — Backfills golddiffat15 into Oracle's Elixir data.

Strategy:
1. Scrape the matchlist page for each LPL tournament to get game IDs
2. For each game, scrape the fullstats page to get per-player GD@15
3. Match to Oracle's Elixir data by date + team name + player name
4. Write the backfilled golddiffat15 into the CSV files

Usage:
    python src/golgg_scraper.py              # Scrape and backfill
    python src/golgg_scraper.py --test       # Test on a single game
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
import json
import sys
import numpy as np
from datetime import datetime

BASE_URL = "https://gol.gg"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

# LPL tournaments to scrape (ordered chronologically)
LPL_TOURNAMENTS = [
    # 2026
    "LPL 2026 Split 1",
    "LPL 2026 Split 1 Playoffs",
]

RATE_LIMIT_SECONDS = 1.5  # Be respectful, don't hammer the site


def fetch_page(url, retries=3):
    """Fetch a page with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, 'html.parser')
            elif resp.status_code == 404:
                print(f"  404: {url}")
                return None
            else:
                print(f"  HTTP {resp.status_code} on attempt {attempt+1}: {url}")
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
        time.sleep(2 * (attempt + 1))
    return None


def get_game_ids_from_matchlist(tournament_name):
    """Scrape the matchlist page for a tournament and return game IDs."""
    url = f"{BASE_URL}/tournament/tournament-matchlist/{requests.utils.quote(tournament_name)}/"
    print(f"  Fetching matchlist: {url}")
    soup = fetch_page(url)
    if soup is None:
        return []
    
    series_ids = []
    # The matchlist typically links to page-summary for the series
    for link in soup.find_all('a', href=True):
        href = link['href']
        match = re.search(r'/game/stats/(\d+)/', href)
        if match:
            gid = int(match.group(1))
            if gid not in series_ids:
                series_ids.append(gid)
    
    print(f"  Found {len(series_ids)} series")
    
    all_game_ids = []
    for i, sid in enumerate(series_ids):
        # Fetch summary page to find all games in the series
        print(f"    Series {i+1}/{len(series_ids)} (ID: {sid})...", end="", flush=True)
        summary_url = f"{BASE_URL}/game/stats/{sid}/page-summary/"
        s_soup = fetch_page(summary_url)
        found_in_series = 0
        if s_soup:
            for link in s_soup.find_all('a', href=True):
                href = link['href']
                if 'page-game' in href or 'page-summary' in href:
                    match = re.search(r'/game/stats/(\d+)/', href)
                    if match:
                        gid = int(match.group(1))
                        if gid not in all_game_ids:
                            all_game_ids.append(gid)
                            found_in_series += 1
        print(f" found {found_in_series} games")
                            
    print(f"  Found {len(all_game_ids)} total games")
    return all_game_ids


def scrape_game_fullstats(game_id):
    """Scrape the fullstats page for a single game. Returns player-level data."""
    url = f"{BASE_URL}/game/stats/{game_id}/page-fullstats/"
    soup = fetch_page(url)
    if soup is None:
        return None
    
    result = {
        'game_id': game_id,
        'players': [],
    }
    
    # Try to get the date and teams from the page header
    title = soup.find('title')
    if title:
        result['title'] = title.text.strip()
    
    h1 = soup.find('h1')
    if h1:
        teams = h1.text.strip().split(' vs ')
        if len(teams) >= 2:
            result['team_blue'] = teams[0].strip()
            result['team_red'] = teams[1].strip()

    # Look for date info - gol.gg typically has it in the page
    date_elem = soup.find('div', class_='col-12 col-sm-5 text-right')
    if date_elem:
        result['date_str'] = date_elem.text.strip()
    
    # Find the stats table
    table = soup.find('table', class_='completestats')
    if not table:
        return result
        
    player_row = None
    gd15_row = None
    csd15_row = None
    xpd15_row = None
    
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if not cells:
            continue
        row_header = cells[0].text.strip().upper()
        if row_header == 'PLAYER' or row_header == 'NAME':
            player_row = cells
        elif 'GD' in row_header and '15' in row_header:
            gd15_row = cells
        elif 'CSD' in row_header and '15' in row_header:
            csd15_row = cells
        elif 'XPD' in row_header and '15' in row_header:
            xpd15_row = cells
            
    if player_row and gd15_row:
        # Transposed table: first cell is label, following are players (1-5 blue, 6-10 red)
        for i in range(1, len(player_row)):
            player_name = player_row[i].text.strip()
            if not player_name:
                continue
                
            def parse_val(r_cells, idx):
                if r_cells and idx < len(r_cells):
                    text = r_cells[idx].text.strip().replace(',', '').replace('+', '')
                    try: return int(text)
                    except: pass
                return None
                
            gd15_val = parse_val(gd15_row, i)
            csd15_val = parse_val(csd15_row, i)
            xpd15_val = parse_val(xpd15_row, i)
                
            team_name = result.get('team_blue') if i <= 5 else result.get('team_red')
            
            if gd15_val is not None:
                result['players'].append({
                    'player': player_name,
                    'team': team_name,
                    'gd15': gd15_val,
                    'csd15': csd15_val,
                    'xpd15': xpd15_val,
                })
    
    return result


def scrape_game_summary(game_id):
    """Scrape the summary page for date/team info if fullstats didn't have it."""
    url = f"{BASE_URL}/game/stats/{game_id}/page-summary/"
    soup = fetch_page(url)
    if soup is None:
        return {}
    
    info = {}
    
    # Try to extract date
    for div in soup.find_all('div'):
        text = div.text.strip()
        # Look for date patterns like "2025-01-15" or "15/01/2025" etc.
        date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', text)
        if date_match:
            info['date'] = date_match.group(1)
            break
    
    # Try to get team names
    team_links = soup.find_all('a', href=re.compile(r'/team/team-stats/'))
    teams = []
    for tl in team_links:
        name = tl.text.strip()
        if name and name not in teams:
            teams.append(name)
    if len(teams) >= 2:
        info['team_blue'] = teams[0]
        info['team_red'] = teams[1]
    
    return info


def test_single_game():
    """Test scraping a single known LPL game."""
    print("Testing single game scrape...")
    # Try a recent LPL game ID (from the browser exploration)
    test_ids = [75280, 75281, 75282]
    
    for gid in test_ids:
        print(f"\n--- Game {gid} ---")
        result = scrape_game_fullstats(gid)
        if result:
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  Teams: {result.get('team_blue', '?')} vs {result.get('team_red', '?')}")
            print(f"  Players found: {len(result['players'])}")
            for p in result['players']:
                print(f"    {p['player']:20s} ({p.get('team', '?'):20s}) GD@15: {p['gd15']:+d}")
            return result
        else:
            print("  Failed to scrape")
    return None


def scrape_all_lpl():
    """Scrape all LPL tournaments and save to a master CSV."""
    all_games = []
    
    for tournament in LPL_TOURNAMENTS:
        print(f"\n{'='*60}")
        print(f"Tournament: {tournament}")
        print(f"{'='*60}")
        
        game_ids = get_game_ids_from_matchlist(tournament)
        
        for i, gid in enumerate(game_ids):
            print(f"  [{i+1}/{len(game_ids)}] Game {gid}...", end="", flush=True)
            result = scrape_game_fullstats(gid)
            
            if result and result['players']:
                # Get summary for date if needed
                summary = scrape_game_summary(gid)
                result.update({k: v for k, v in summary.items() if k not in result})
                
                for p in result['players']:
                    all_games.append({
                        'golgg_game_id': gid,
                        'tournament': tournament,
                        'date': result.get('date', ''),
                        'team_blue': result.get('team_blue', ''),
                        'team_red': result.get('team_red', ''),
                        'player': p['player'],
                        'team': p.get('team', ''),
                        'golddiffat15': p['gd15'],
                        'csdiffat15': p.get('csd15'),
                        'xpdiffat15': p.get('xpd15'),
                    })
                print(f" OK ({len(result['players'])} players)")
            else:
                print(" SKIP (no data)")
    
    # Save
    df = pd.DataFrame(all_games)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'golgg_lpl_gd15.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} player-game records to {out_path}")
    return df


def backfill_oracle_csvs(golgg_df):
    """Merge gol.gg GD@15 data into Oracle's Elixir CSV files."""
    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')
    print("\nBackfilling Oracle's Elixir CSVs...")
    
    def normalize_name(name):
        if pd.isna(name): return ""
        return re.sub(r'[^a-z0-9]', '', str(name).lower())
    
    golgg_df['norm_player'] = golgg_df['player'].apply(normalize_name)
    golgg_df['date_dt'] = pd.to_datetime(golgg_df['date'], errors='coerce', utc=True)
    
    valid_gol = golgg_df.dropna(subset=['date_dt', 'golddiffat15']).copy()
    valid_gol['date_only'] = valid_gol['date_dt'].dt.date
    valid_gol = valid_gol.sort_values('golgg_game_id')
    
    for year in ['2026']:
        filepath = os.path.join(csv_dir, f"{year}_LoL_esports_match_data_from_OraclesElixir.csv")
        if not os.path.exists(filepath):
            continue
            
        print(f"Processing {year}...")
        df = pd.read_csv(filepath, low_memory=False)
        
        lpl_mask = (df['league'] == 'LPL') & (df['position'] != 'team')
        if not lpl_mask.any():
            print(f"  No LPL data in {year}, skipping.")
            continue
            
        df_lpl = df[lpl_mask].copy()
        df_lpl['norm_player'] = df_lpl['playername'].apply(normalize_name)
        df_lpl['date_dt'] = pd.to_datetime(df_lpl['date'], errors='coerce', utc=True)
        df_lpl['date_only'] = df_lpl['date_dt'].dt.date
        
        # Clear existing backfilled data for LPL in this year to prevent duplicates
        df.loc[lpl_mask, 'golddiffat15'] = np.nan
        df.loc[lpl_mask, 'csdiffat15'] = np.nan
        df.loc[lpl_mask, 'xpdiffat15'] = np.nan
        
        filled_count = 0
        
        for (p, d), oe_group in df_lpl.groupby(['norm_player', 'date_only']):
            if pd.isna(d) or not p: continue
            
            # Find golgg games for this player on this date
            gol_group = valid_gol[(valid_gol['norm_player'] == p) & (valid_gol['date_only'] == d)]
            
            # If not found, check +/- 1 day
            if len(gol_group) == 0:
                gol_group = valid_gol[(valid_gol['norm_player'] == p) & (abs(valid_gol['date_only'] - d) <= pd.Timedelta(days=1))]
                
            if len(gol_group) > 0:
                oe_group = oe_group.sort_values(['date_dt', 'gameid', 'game'])
                gol_group = gol_group.sort_values('golgg_game_id')
                
                # Match sequentially
                for i in range(min(len(oe_group), len(gol_group))):
                    idx = oe_group.index[i]
                    gol_row = gol_group.iloc[i]
                    
                    df.at[idx, 'golddiffat15'] = gol_row['golddiffat15']
                    if pd.notna(gol_row.get('csdiffat15')):
                        df.at[idx, 'csdiffat15'] = gol_row['csdiffat15']
                    if pd.notna(gol_row.get('xpdiffat15')):
                        df.at[idx, 'xpdiffat15'] = gol_row['xpdiffat15']
                    filled_count += 1
                        
        # Now fix team rows
        team_rows_filled = 0
        for (gameid, teamid), group in df[df['league'] == 'LPL'].groupby(['gameid', 'teamid']):
            player_rows = group[group['position'] != 'team']
            team_rows = group[group['position'] == 'team']
            if len(team_rows) == 1 and len(player_rows) > 0:
                if not player_rows['golddiffat15'].isna().all():
                    df.loc[team_rows.index, 'golddiffat15'] = player_rows['golddiffat15'].sum()
                    df.loc[team_rows.index, 'csdiffat15'] = player_rows['csdiffat15'].sum()
                    df.loc[team_rows.index, 'xpdiffat15'] = player_rows['xpdiffat15'].sum()
                    team_rows_filled += 1
                        
        print(f"  Filled {filled_count} player values and {team_rows_filled} team values in {year} data")
        df.to_csv(filepath, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_single_game()
    elif len(sys.argv) > 1 and sys.argv[1] == '--backfill':
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'golgg_lpl_gd15.csv'))
        backfill_oracle_csvs(df)
    else:
        df = scrape_all_lpl()
        backfill_oracle_csvs(df)
