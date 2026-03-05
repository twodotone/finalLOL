import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PlayerEloSystem:
    def __init__(self, base_k=20, placement_k=40, placement_matches=10, decay_days=30, decay_rate=25):
        self.base_k = base_k
        self.placement_k = placement_k
        self.placement_matches = placement_matches
        self.decay_days = decay_days
        self.decay_rate = decay_rate
        
        # Player storage: {playerid: {'elo': int, 'games_played': int, 'last_played': datetime, 'league': str}}
        self.players = {}
        
        # Regional Tiers - baseline ELOs for new players depending on the league they debut in
        self.league_tiers = {
            'Tier_1': {'leagues': ['LPL', 'LCK', 'MSI', 'WCC', 'EWC'], 'base_elo': 1600},
            'Tier_2': {'leagues': ['LEC', 'LCS', 'LTA', 'LCP'], 'base_elo': 1500},
            'Tier_3': {'leagues': ['PCS', 'VCS', 'CBLOL', 'LLA'], 'base_elo': 1450},
            'Tier_4': {'leagues': ['LFL', 'NACL', 'AL', 'LCKC', 'LDL', 'LJL', 'TCL', 'LVP SL', 'PRM'], 'base_elo': 1400},
            # Tier 5 default catch-all
            'Tier_5': {'base_elo': 1350}
        }
        
        # Flatten league tiers for easy lookup
        self.league_base_elo = {}
        for tier, data in self.league_tiers.items():
            if 'leagues' in data:
                for league in data['leagues']:
                    self.league_base_elo[league] = data['base_elo']
                    
    def get_league_base_elo(self, league):
        return self.league_base_elo.get(league, self.league_tiers['Tier_5']['base_elo'])
    
    def _initialize_player(self, player_id, date, league):
        base_elo = self.get_league_base_elo(league)
        self.players[player_id] = {
            'elo': base_elo,
            'games_played': 0,
            'last_played': date,
            'league': league
        }

    def _apply_decay_and_transfers(self, player_id, current_date, new_league):
        """ Handles time-decay and league transfer skepticism (tax). """
        player = self.players[player_id]
        
        # 1. Time Decay (Daily 0.3% compound decay towards regional mean)
        days_since_last = (current_date - player['last_played']).days
        if days_since_last > 0:
            regional_mean = self.get_league_base_elo(player['league'])
            
            # Pull ELO toward regional mean by 0.3% for each day of inactivity
            daily_decay_pct = 0.003
            
            diff = player['elo'] - regional_mean
            new_diff = diff * ((1 - daily_decay_pct) ** days_since_last)
            player['elo'] = regional_mean + new_diff
                
        # 2. Transfer Skepticism (Tax)
        old_league = player['league']
        if old_league != new_league and new_league is not None:
            old_base = self.get_league_base_elo(old_league)
            new_base = self.get_league_base_elo(new_league)
            
            # If moving to a stronger league (transferring UP), tax the "farmed" ELO
            if new_base > old_base:
                # ELO above the old regional mean
                surplus = player['elo'] - old_base
                if surplus > 0:
                    # 25% tax on the surplus
                    tax = surplus * 0.25
                    player['elo'] -= tax
            
            # Update their active league
            player['league'] = new_league
            
        player['last_played'] = current_date

    def get_player_elo(self, player_id, current_date, league=None):
        """ Retrieves ELO, applying decay and transfer tax automatically. """
        if player_id not in self.players:
            self._initialize_player(player_id, current_date, league)
        else:
            self._apply_decay_and_transfers(player_id, current_date, league)
            
        return self.players[player_id]['elo']

    def update_player_elo(self, player_id, new_elo, current_date):
        self.players[player_id]['elo'] = new_elo
        self.players[player_id]['games_played'] += 1
        self.players[player_id]['last_played'] = current_date

    def get_k_factor(self, player_id):
        """ High K-factor for new players to calibrate quickly. """
        if self.players[player_id]['games_played'] < self.placement_matches:
            return self.placement_k
        return self.base_k

    def calculate_expected_score(self, team_a_elo, team_b_elo):
        """ Standard ELO formula """
        return 1 / (1 + 10 ** ((team_b_elo - team_a_elo) / 400.0))

    def process_match(self, match_date, league, team_a_players, team_b_players, team_a_won):
        """
        Processes a single match.
        team_a_players / team_b_players should be lists of player_ids.
        team_a_won is boolean.
        Returns the expected probabilities and the updated ELOs for tracking.
        """
        
        # Get Current ELOs
        team_a_elos = {pid: self.get_player_elo(pid, match_date, league) for pid in team_a_players}
        team_b_elos = {pid: self.get_player_elo(pid, match_date, league) for pid in team_b_players}
        
        # Calculate Team Averages (aggregate strength)
        avg_a_elo = sum(team_a_elos.values()) / len(team_a_elos)
        avg_b_elo = sum(team_b_elos.values()) / len(team_b_elos)
        
        # Calculate Expected Scores
        expected_a = self.calculate_expected_score(avg_a_elo, avg_b_elo)
        expected_b = 1 - expected_a
        
        # Actual Scores
        actual_a = 1 if team_a_won else 0
        actual_b = 0 if team_a_won else 1
        
        # Update Players
        updated_a = {}
        for pid, elo in team_a_elos.items():
            k = self.get_k_factor(pid)
            new_elo = elo + k * (actual_a - expected_a)
            self.update_player_elo(pid, new_elo, match_date)
            updated_a[pid] = new_elo
            
        updated_b = {}
        for pid, elo in team_b_elos.items():
            k = self.get_k_factor(pid)
            new_elo = elo + k * (actual_b - expected_b)
            self.update_player_elo(pid, new_elo, match_date)
            updated_b[pid] = new_elo
            
        return {
            'expected_a': expected_a,
            'expected_b': expected_b,
            'avg_a_elo_pre': avg_a_elo,
            'avg_b_elo_pre': avg_b_elo,
            'team_a_updated_elos': updated_a,
            'team_b_updated_elos': updated_b
        }
