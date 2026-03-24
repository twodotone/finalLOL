"""
Lane Performance ELO System
============================
A parallel ELO system that tracks individual player performance by lane,
independent of team win/loss. Players earn/lose lane ELO based on their
actual lane stats (golddiff@15, csdiff@15, xpdiff@15) vs their opponent.

This layers on top of the existing team-level PlayerEloSystem without
modifying it. The lane ELOs become additional features for XGBoost.
"""
import numpy as np
from collections import defaultdict


class LaneEloSystem:
    """
    Tracks per-player ELO based on individual lane performance.

    Unlike the team ELO (which gives everyone +K or -K based on W/L),
    lane ELO rewards/punishes players for their actual lane stats:
      - A top laner who goes +500 golddiff@15 in a loss still gains ELO
      - A mid laner who goes -300 golddiff@15 in a win still loses ELO

    The performance score is a composite of lane diffs, normalised to
    roughly [-1, +1] using a tanh squash.  This prevents single blowout
    games from warping ratings.
    """

    POSITIONS = ['top', 'jng', 'mid', 'bot', 'sup']

    def __init__(self, base_elo=1500, k_factor=32,
                 placement_k=48, placement_games=10,
                 daily_decay_pct=0.0005, max_decay_elo=40):
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.placement_k = placement_k
        self.placement_games = placement_games
        self.daily_decay_pct = daily_decay_pct
        self.max_decay_elo = max_decay_elo

        # {playerid: {'elo': float, 'games': int, 'last_played': datetime}}
        self.players = {}

        # Normalisation constants for the tanh squash
        # Loosened to allow more signal through before squashing
        self._gold_scale = 800.0
        self._cs_scale = 10.0
        self._xp_scale = 600.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_player(self, pid, date, base_elo=None):
        if pid not in self.players:
            initial_elo = base_elo if base_elo is not None else self.base_elo
            self.players[pid] = {
                'elo': initial_elo,
                'games': 0,
                'last_played': date,
                'home_base': initial_elo
            }

    def _apply_decay(self, pid, current_date):
        p = self.players[pid]
        days = (current_date - p['last_played']).days
        if days <= 0:
            return
        home_base = p.get('home_base', self.base_elo)
        diff = p['elo'] - home_base
        new_diff = diff * ((1 - self.daily_decay_pct) ** days)
        actual = abs(diff - new_diff)
        if actual > self.max_decay_elo:
            sign = 1 if diff > 0 else -1
            new_diff = diff - sign * self.max_decay_elo
        p['elo'] = home_base + new_diff
        p['last_played'] = current_date

    def _get_k(self, pid):
        if self.players[pid]['games'] < self.placement_games:
            return self.placement_k
        return self.k_factor

    # ------------------------------------------------------------------
    # Performance scoring
    # ------------------------------------------------------------------
    def compute_performance_score(self, golddiff15, csdiff15, xpdiff15,
                                  kills, deaths, assists, dpm, damageshare):
        """
        Composite lane performance score squashed to roughly [-1, +1].

        Primary signal (70%): lane phase diffs (gold/cs/xp @ 15 min)
        Secondary signal (30%): full-game efficiency (KDA-based + damage)

        Using tanh to prevent outlier games from dominating.
        """
        # Lane phase composite (normalised)
        lane_raw = (golddiff15 / self._gold_scale
                    + csdiff15 / self._cs_scale
                    + xpdiff15 / self._xp_scale) / 3.0

        # Full-game efficiency: KDA ratio mapped through tanh
        deaths_safe = max(deaths, 1)
        kda = (kills + 0.5 * assists) / deaths_safe
        # kda of ~3 is average, ~6+ is excellent.  Map to [0, 1] range.
        kda_score = np.tanh((kda - 3.0) / 3.0)

        # Damage share: 0.20 is average, >0.28 is carry-level
        # Map so average → 0, high → positive
        dmg_score = (damageshare - 0.20) * 5.0  # 0.20→0, 0.30→0.5

        # Blend: 70% lane, 30% game
        game_score = 0.6 * kda_score + 0.4 * dmg_score
        
        # Check for missing lane data
        if golddiff15 == 0 and csdiff15 == 0 and xpdiff15 == 0:
            raw = game_score
        else:
            raw = 0.7 * lane_raw + 0.3 * game_score

        return np.tanh(raw)  # squash to (-1, +1)

    # ------------------------------------------------------------------
    # Expected lane performance (from ELO diff)
    # ------------------------------------------------------------------
    def expected_performance(self, player_elo, opponent_elo):
        """
        Expected performance score given ELO ratings.
        Returns value in [0, 1] — 0.5 means equal.
        Standard logistic/ELO formula.
        """
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_lane_elo(self, pid, current_date, base_elo=None):
        """Get a player's lane ELO, initialising or decaying as needed."""
        self._init_player(pid, current_date, base_elo=base_elo)
        self._apply_decay(pid, current_date)
        return self.players[pid]['elo']

    def process_lane_matchup(self, date,
                             player_id, opponent_id,
                             golddiff15, csdiff15, xpdiff15,
                             kills, deaths, assists,
                             dpm, damageshare):
        """
        Process a single lane matchup from one game.

        Updates both players' lane ELOs based on actual vs expected
        performance.
        """
        # Ensure both exist & apply decay
        p_elo = self.get_lane_elo(player_id, date)
        o_elo = self.get_lane_elo(opponent_id, date)

        expected = self.expected_performance(p_elo, o_elo)

        # Actual performance: map tanh score from [-1,1] to [0,1]
        perf = self.compute_performance_score(
            golddiff15, csdiff15, xpdiff15,
            kills, deaths, assists, dpm, damageshare
        )
        actual = (perf + 1.0) / 2.0  # map [-1,1] → [0,1]

        # ELO update
        k_p = self._get_k(player_id)
        k_o = self._get_k(opponent_id)

        # Regional Scaling: Apply a multiplier based on the league's quality
        # to prevent Lane ELO inflation from stomping in minor leagues.
        # Calibrated to the league_tiers in elo.py
        if base_elo is not None:
             if base_elo >= 1600: quality_mult = 1.0   # Tier S
             elif base_elo >= 1550: quality_mult = 0.95 # Tier 1
             elif base_elo >= 1475: quality_mult = 0.85 # Tier 2/2b
             elif base_elo >= 1400: quality_mult = 0.75 # Tier 3
             else: quality_mult = 0.65                  # Minor
             k_p *= quality_mult
             k_o *= quality_mult

        new_p = p_elo + k_p * (actual - expected)
        new_o = o_elo + k_o * ((1 - actual) - (1 - expected))

        self.players[player_id]['elo'] = new_p
        self.players[player_id]['games'] += 1
        self.players[player_id]['last_played'] = date

        self.players[opponent_id]['elo'] = new_o
        self.players[opponent_id]['games'] += 1
        self.players[opponent_id]['last_played'] = date

        return {
            'player_pre': p_elo,
            'opponent_pre': o_elo,
            'performance_score': perf,
            'player_post': new_p,
            'opponent_post': new_o,
        }
