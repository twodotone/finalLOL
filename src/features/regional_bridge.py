"""
Regional Bridge — time-weighted cross-regional performance tracker.

Maintains a recency-weighted record of how each region performs against
every other region *relative to ELO expectations* (residual-based).
Outputs a single adjustment value for cross-regional matchup predictions.

Design principles:
- Exponential decay: recent results matter more (half-life ~1 year)
- Bayesian shrinkage: with few games, the adjustment stays close to 0
- Capped at ±max_adj to prevent overcorrection
- Blends pair-level data with overall regional performance when pair data is sparse
"""

import numpy as np
from collections import defaultdict
from math import log


class RegionalBridge:
    def __init__(self, half_life_days=365, max_adj=0.15, min_effective_games=5.0,
                 pair_blend_threshold=10.0):
        """
        Args:
            half_life_days: exponential decay half-life for weighting games
            max_adj: cap on the bridge adjustment (±6% default)
            min_effective_games: minimum time-weighted game count before signal kicks in
            pair_blend_threshold: above this many effective pair games, use pair-only;
                                  below, blend with overall regional performance
        """
        self.half_life_days = half_life_days
        self.decay_lambda = log(2) / half_life_days
        self.max_adj = max_adj
        self.min_effective_games = min_effective_games
        self.pair_blend_threshold = pair_blend_threshold

        # {(region_a, region_b): [(date, residual_for_a), ...]}
        # residual = actual_result_a - expected_result_a (from ELO)
        self.pair_records = defaultdict(list)

        # {region: [(date, residual), ...]}  — overall international performance
        self.region_records = defaultdict(list)

        # League code -> major region mapping
        self.league_to_region = {
            'LCK': 'LCK',
            'LPL': 'LPL',
            'LEC': 'LEC',
            'LCS': 'LCS',
            'LTA': 'LTA', 'LTA S': 'LTA', 'LTA N': 'LTA',
            'CBLOL': 'CBLOL',
            'LLA': 'LLA', 'LAS': 'LLA',
            'PCS': 'PCS',
            'VCS': 'VCS',
            'LJL': 'LJL',
            'LCO': 'LCO',
            'TCL': 'TCL',
            'LCP': 'LCP',
        }

        self.tournament_leagues = {
            'MSI', 'WCC', 'EWC', 'WLDs', 'FST', 'DCup', 'ASI',
            'Asia Master', 'CCWS', 'AC', 'Americas Cup',
            'IC', 'KeSPA', 'LES',
            'EM', 'EUM',  # EMEA Masters / EU Masters (cross-sub-league)
        }

    def get_region(self, league):
        """Map a league code to its major region. Returns None for unknown/tournament."""
        if league in self.tournament_leagues:
            return None
        return self.league_to_region.get(league, league)

    def record_game(self, date, region_a, region_b, actual_a, expected_a):
        """
        Record a cross-regional game result.

        Args:
            date: game date (datetime)
            region_a: major region of team A
            region_b: major region of team B
            actual_a: 1 if team A won, 0 otherwise
            expected_a: ELO-based expected win probability for team A
        """
        if region_a == region_b or region_a is None or region_b is None:
            return

        residual_a = actual_a - expected_a

        # Pair-level: store from both perspectives
        self.pair_records[(region_a, region_b)].append((date, residual_a))
        self.pair_records[(region_b, region_a)].append((date, -residual_a))

        # Overall: each region's international performance
        self.region_records[region_a].append((date, residual_a))
        self.region_records[region_b].append((date, -residual_a))

    def _weighted_avg(self, records, as_of_date):
        """Compute time-weighted average residual and effective game count."""
        if not records:
            return 0.0, 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for game_date, residual in records:
            days_ago = (as_of_date - game_date).days
            if days_ago < 0:
                continue
            w = np.exp(-self.decay_lambda * days_ago)
            weighted_sum += w * residual
            total_weight += w

        if total_weight < 1e-9:
            return 0.0, 0.0

        return weighted_sum / total_weight, total_weight

    def get_bridge_adj(self, date, region_a, region_b):
        """
        Get the bridge adjustment for region_a vs region_b.

        Returns a value in [-max_adj, +max_adj]. Positive means region_a
        is expected to outperform ELO against region_b.

        Uses pair-level data when sufficient, blending with overall regional
        performance when pair data is sparse.
        """
        if region_a == region_b or region_a is None or region_b is None:
            return 0.0

        # Pair-level signal
        pair_avg, pair_weight = self._weighted_avg(
            self.pair_records.get((region_a, region_b), []), date
        )

        # Overall regional signal: region_a's overall intl performance minus region_b's
        overall_a_avg, overall_a_weight = self._weighted_avg(
            self.region_records.get(region_a, []), date
        )
        overall_b_avg, overall_b_weight = self._weighted_avg(
            self.region_records.get(region_b, []), date
        )
        overall_avg = overall_a_avg - overall_b_avg
        overall_weight = min(overall_a_weight, overall_b_weight)

        # Blend: pair data dominates when sufficient, overall fills in when sparse
        if pair_weight >= self.pair_blend_threshold:
            raw_adj = pair_avg
            effective_weight = pair_weight
        elif pair_weight > 0 and overall_weight > 0:
            # Linear blend: pair_fraction increases as pair_weight approaches threshold
            pair_fraction = pair_weight / self.pair_blend_threshold
            raw_adj = pair_fraction * pair_avg + (1 - pair_fraction) * overall_avg
            effective_weight = pair_weight + (1 - pair_fraction) * overall_weight
        elif overall_weight > 0:
            raw_adj = overall_avg
            effective_weight = overall_weight
        else:
            return 0.0

        # Bayesian shrinkage: scale down when effective data is low
        if effective_weight < self.min_effective_games:
            raw_adj *= effective_weight / self.min_effective_games

        # Cap
        return max(-self.max_adj, min(self.max_adj, raw_adj))
