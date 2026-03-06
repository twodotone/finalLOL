import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PlayerEloSystem:
    def __init__(self, base_k=20, placement_k=40, placement_matches=10,
                 intl_k_multiplier=2.0, daily_decay_pct=0.001, max_decay_elo=50):
        self.base_k = base_k
        self.placement_k = placement_k
        self.placement_matches = placement_matches
        self.intl_k_multiplier = intl_k_multiplier
        self.daily_decay_pct = daily_decay_pct
        self.max_decay_elo = max_decay_elo

        # Player storage: {playerid: {'elo', 'games_played', 'last_played', 'home_league'}}
        self.players = {}

        # --- Tournament codes (neutral ground — no transfer tax, no baseline) ---
        self.tournament_leagues = {'MSI', 'WCC', 'EWC', 'WLDs', 'FST', 'DCup', 'ASI', 'Asia Master', 'CCWS'}

        # --- Regional Tiers — only domestic leagues ---
        self.league_tiers = {
            'Tier_1': {'leagues': ['LPL', 'LCK'], 'base_elo': 1600},
            'Tier_2': {'leagues': ['LEC', 'LCS', 'LTA', 'LTA N', 'LTA S', 'LCP'], 'base_elo': 1500},
            'Tier_3': {'leagues': ['PCS', 'VCS', 'CBLOL', 'LLA'], 'base_elo': 1450},
            'Tier_4': {'leagues': ['LFL', 'NACL', 'AL', 'LCKC', 'LDL', 'LJL', 'TCL', 'LVP SL', 'PRM'], 'base_elo': 1400},
            'Tier_5': {'base_elo': 1350}
        }

        # Flatten for fast lookup
        self.league_base_elo = {}
        for tier, data in self.league_tiers.items():
            if 'leagues' in data:
                for league in data['leagues']:
                    self.league_base_elo[league] = data['base_elo']

        # --- Dynamic Regional Gravity ---
        # Tracks accumulated baseline shifts from international results
        # {league_code: float shift}  e.g. {'LCP': +12.5, 'LPL': -3.0}
        self.regional_baseline_shifts = {}
        # Cap per-event shift to prevent single-tournament distortion
        self.max_event_shift = 30

    # ------------------------------------------------------------------
    # Regional Baseline (with dynamic gravity)
    # ------------------------------------------------------------------
    def get_league_base_elo(self, league):
        """Returns the current regional baseline including any dynamic shifts."""
        if league in self.tournament_leagues:
            return 1500  # tournaments don't have their own baseline
        base = self.league_base_elo.get(league, self.league_tiers['Tier_5']['base_elo'])
        shift = self.regional_baseline_shifts.get(league, 0.0)
        return base + shift

    def recalculate_league_baselines(self, event_player_deltas):
        """
        After an international event, update regional baselines.
        
        event_player_deltas: dict {player_id: elo_delta} for all players
                             who participated in the event.
        
        Groups deltas by each player's home_league, averages,
        and shifts the regional baseline (capped at ±max_event_shift).
        """
        league_deltas = {}  # {league: [delta1, delta2, ...]}
        for pid, delta in event_player_deltas.items():
            if pid not in self.players:
                continue
            home = self.players[pid].get('home_league')
            if home and home not in self.tournament_leagues:
                league_deltas.setdefault(home, []).append(delta)

        for league, deltas in league_deltas.items():
            avg_delta = sum(deltas) / len(deltas)
            capped = max(-self.max_event_shift, min(self.max_event_shift, avg_delta))
            current = self.regional_baseline_shifts.get(league, 0.0)
            # Cap the TOTAL accumulated shift to ±max_total_shift
            new_total = current + capped
            max_total = self.max_event_shift * 3  # ±90 ELO max total regional shift
            new_total = max(-max_total, min(max_total, new_total))
            self.regional_baseline_shifts[league] = new_total

    def is_tournament(self, league):
        return league in self.tournament_leagues

    # ------------------------------------------------------------------
    # Player Initialization & Decay
    # ------------------------------------------------------------------
    def _initialize_player(self, player_id, date, league):
        home = league if not self.is_tournament(league) else None
        base_elo = self.get_league_base_elo(league) if home else 1500
        self.players[player_id] = {
            'elo': base_elo,
            'games_played': 0,
            'last_played': date,
            'home_league': home
        }

    def _apply_decay_and_transfers(self, player_id, current_date, new_league):
        """Handles softened time-decay and domestic-only transfer tax."""
        player = self.players[player_id]
        effective_league = new_league

        # If the match is at a tournament, keep home league — no transfer
        if new_league and self.is_tournament(new_league):
            effective_league = player['home_league'] or new_league

        # 1. Softened Time Decay (capped)
        days_since_last = (current_date - player['last_played']).days
        if days_since_last > 0:
            home = player['home_league'] or effective_league
            regional_mean = self.get_league_base_elo(home) if home else 1500

            diff = player['elo'] - regional_mean
            new_diff = diff * ((1 - self.daily_decay_pct) ** days_since_last)

            # Cap total decay at max_decay_elo
            actual_decay = abs(diff - new_diff)
            if actual_decay > self.max_decay_elo:
                sign = 1 if diff > 0 else -1
                new_diff = diff - sign * self.max_decay_elo

            player['elo'] = regional_mean + new_diff

        # 2. Transfer Tax — only between domestic leagues (not tournaments)
        old_home = player['home_league']
        if (effective_league and old_home
                and effective_league != old_home
                and not self.is_tournament(effective_league)):
            old_base = self.get_league_base_elo(old_home)
            new_base = self.get_league_base_elo(effective_league)

            if new_base > old_base:
                surplus = player['elo'] - old_base
                if surplus > 0:
                    player['elo'] -= surplus * 0.25

            player['home_league'] = effective_league

        player['last_played'] = current_date

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_player_elo(self, player_id, current_date, league=None):
        """Retrieves ELO, applying decay and transfer tax automatically."""
        if player_id not in self.players:
            self._initialize_player(player_id, current_date, league)
        else:
            self._apply_decay_and_transfers(player_id, current_date, league)
        return self.players[player_id]['elo']

    def update_player_elo(self, player_id, new_elo, current_date):
        self.players[player_id]['elo'] = new_elo
        self.players[player_id]['games_played'] += 1
        self.players[player_id]['last_played'] = current_date

    def get_k_factor(self, player_id, is_cross_region=False):
        """
        K-factor with placement boost and international multiplier.
        - Placement players (< placement_matches games): placement_k
        - Cross-regional international matches: base_k * intl_k_multiplier
        - Standard domestic matches: base_k
        """
        if self.players[player_id]['games_played'] < self.placement_matches:
            k = self.placement_k
        else:
            k = self.base_k
        if is_cross_region:
            k *= self.intl_k_multiplier
        return k

    def calculate_expected_score(self, team_a_elo, team_b_elo):
        """Standard ELO formula."""
        return 1 / (1 + 10 ** ((team_b_elo - team_a_elo) / 400.0))

    def _get_home_league(self, player_id):
        """Return a player's home (domestic) league."""
        if player_id in self.players:
            return self.players[player_id].get('home_league')
        return None

    def process_match(self, match_date, league, team_a_players, team_b_players, team_a_won):
        """
        Processes a single match with cross-regional K-factor awareness.
        
        - If the match league is a tournament AND the two teams come from
          different home leagues → is_cross_region = True → elevated K.
        - Tournament matches never trigger transfer tax.
        """

        # Get Current ELOs (tournament matches keep home_league)
        team_a_elos = {pid: self.get_player_elo(pid, match_date, league)
                       for pid in team_a_players}
        team_b_elos = {pid: self.get_player_elo(pid, match_date, league)
                       for pid in team_b_players}

        # Determine if this is a cross-regional clash
        is_cross_region = False
        if self.is_tournament(league):
            homes_a = {self._get_home_league(pid) for pid in team_a_players} - {None}
            homes_b = {self._get_home_league(pid) for pid in team_b_players} - {None}
            if homes_a and homes_b and homes_a.isdisjoint(homes_b):
                is_cross_region = True

        # Team averages
        avg_a_elo = sum(team_a_elos.values()) / len(team_a_elos)
        avg_b_elo = sum(team_b_elos.values()) / len(team_b_elos)

        # Expected scores
        expected_a = self.calculate_expected_score(avg_a_elo, avg_b_elo)
        expected_b = 1 - expected_a

        # Actual scores
        actual_a = 1 if team_a_won else 0
        actual_b = 0 if team_a_won else 1

        # Update players
        updated_a = {}
        for pid, elo in team_a_elos.items():
            k = self.get_k_factor(pid, is_cross_region=is_cross_region)
            new_elo = elo + k * (actual_a - expected_a)
            self.update_player_elo(pid, new_elo, match_date)
            updated_a[pid] = new_elo

        updated_b = {}
        for pid, elo in team_b_elos.items():
            k = self.get_k_factor(pid, is_cross_region=is_cross_region)
            new_elo = elo + k * (actual_b - expected_b)
            self.update_player_elo(pid, new_elo, match_date)
            updated_b[pid] = new_elo

        return {
            'expected_a': expected_a,
            'expected_b': expected_b,
            'avg_a_elo_pre': avg_a_elo,
            'avg_b_elo_pre': avg_b_elo,
            'team_a_updated_elos': updated_a,
            'team_b_updated_elos': updated_b,
            'is_cross_region': is_cross_region
        }
