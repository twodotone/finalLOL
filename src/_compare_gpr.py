"""Compare our ELO rankings vs Riot GPR — side by side analysis."""

# Riot GPR from screenshots (rank, team, league, points)
riot = [
    (1, 'Gen.G Esports', 'LCK', 1586),
    (2, 'T1', 'LCK', 1513),
    (3, 'Hanwha Life Esports', 'LCK', 1466),
    (4, 'Bilibili Gaming', 'LPL', 1465),
    (5, 'KT Rolster', 'LCK', 1445),
    (6, "Anyone's Legend", 'LPL', 1436),
    (7, 'G2 Esports', 'LEC', 1405),
    (8, 'CTBC Flying Oyster', 'LCP', 1382),
    (9, 'JD Gaming', 'LPL', 1381),
    (10, 'Top Esports', 'LPL', 1374),
    (11, 'FlyQuest', 'LCS', 1360),
    (12, 'Weibo Gaming', 'LPL', 1351),
    (13, 'BNK FEARX', 'LCK', 1340),
    (13, 'Dplus Kia', 'LCK', 1340),
    (15, 'Karmine Corp', 'LEC', 1319),
    (16, 'Team Secret Whales', 'LCP', 1307),
    (17, 'Invictus Gaming', 'LPL', 1291),
    (17, 'Deep Cross Gaming', 'LCP', 1291),
    (19, 'Cloud9', 'LCS', 1290),
    (20, 'Movistar KOI', 'LEC', 1281),
    (21, 'GAM Esports', 'LCP', 1274),
    (22, 'Team Liquid', 'LCS', 1259),
    (23, 'NIP', 'LPL', 1251),
    (24, 'MVK Esports', 'LCP', 1250),
    (25, 'LNG Esports', 'LPL', 1247),
    (26, 'Nongshim', 'LCK', 1244),
    (27, 'LYON', 'LCS', 1220),
    (28, 'Team WE', 'LPL', 1217),
    (29, 'Edward Gaming', 'LPL', 1213),
    (30, 'Team Vitality', 'LEC', 1206),
    (30, 'DN SOOPers', 'LCK', 1206),
    (32, 'Fnatic', 'LEC', 1205),
    (33, 'Vivo Keyd Stars', 'CBLOL', 1199),
    (34, 'DRX', 'LCK', 1191),
    (35, 'SoftBank HAWKS', 'LCP', 1189),
    (36, 'RED Canids', 'CBLOL', 1183),
    (36, 'GIANTX', 'LEC', 1183),
    (38, 'LGD Gaming', 'LPL', 1182),
    (38, 'FURIA', 'CBLOL', 1182),
    (40, 'BRION', 'LCK', 1181),
    (41, 'Sentinels', 'LCS', 1172),
    (42, 'LOUD', 'CBLOL', 1171),
]

# Our ELO rankings (from active 2026 teams analysis)
ours = [
    (1, 'Gen.G', 'LCK', 1827),
    (2, 'Bilibili Gaming', 'LPL', 1747),
    (3, "Anyone's Legend", 'LPL', 1713),
    (4, 'G2 Esports', 'LEC', 1708),
    (5, 'T1', 'LCK', 1678),
    (6, 'JD Gaming', 'LPL', 1673),
    (7, 'FURIA', 'CBLOL', 1657),
    (8, 'Karmine Corp', 'LEC', 1649),
    (9, 'BNK FEARX', 'LCK', 1645),
    (10, 'KT Rolster', 'LCK', 1629),
    (11, 'Weibo Gaming', 'LPL', 1624),
    (12, 'Team Secret Whales', 'LCP', 1621),
    (13, 'Hanwha Life Esports', 'LCK', 1617),
    (14, 'LNG Esports', 'LPL', 1610),
    (15, 'Dplus Kia', 'LCK', 1609),
    (16, 'Top Esports', 'LPL', 1603),
    (17, 'Movistar KOI', 'LEC', 1584),
    (18, 'LYON', 'LCS', 1574),
    (19, 'DN SOOPers', 'LCK', 1570),
    (20, 'Edward Gaming', 'LPL', 1559),
]

# Match names between systems
team_map = {
    'Gen.G Esports': 'Gen.G',
    "Anyone's Legend": "Anyone's Legend",
    'Hanwha Life Esports': 'Hanwha Life Esports',
    'Bilibili Gaming': 'Bilibili Gaming',
    'KT Rolster': 'KT Rolster',
    'G2 Esports': 'G2 Esports',
    'JD Gaming': 'JD Gaming',
    'Top Esports': 'Top Esports',
    'Weibo Gaming': 'Weibo Gaming',
    'BNK FEARX': 'BNK FEARX',
    'Dplus Kia': 'Dplus Kia',
    'Karmine Corp': 'Karmine Corp',
    'Team Secret Whales': 'Team Secret Whales',
    'Movistar KOI': 'Movistar KOI',
    'DN SOOPers': 'DN SOOPers',
    'Edward Gaming': 'Edward Gaming',
    'T1': 'T1',
    'FURIA': 'FURIA',
    'LYON': 'LYON',
    'LNG Esports': 'LNG Esports',
}

our_lookup = {name: rank for rank, name, _, _ in ours}

print("=" * 85)
print(f"{'TEAM':<25} {'RIOT':>5} {'OURS':>5} {'DIFF':>6}  NOTES")
print("=" * 85)

big_over = []
big_under = []
missing_from_ours = []

for riot_rank, riot_name, league, pts in riot:
    our_name = team_map.get(riot_name)
    our_rank = our_lookup.get(our_name) if our_name else None
    if our_rank:
        diff = our_rank - riot_rank
        sign = '+' if diff > 0 else ''
        note = ''
        if diff <= -5:
            note = ' ** WE OVERRATE'
            big_over.append((riot_name, league, riot_rank, our_rank, diff))
        elif diff >= 5:
            note = ' ** WE UNDERRATE'
            big_under.append((riot_name, league, riot_rank, our_rank, diff))
        print(f"  {riot_name:<25} {riot_rank:>4}  {our_rank:>4}  {sign}{diff:>4}  {league}{note}")
    else:
        missing_from_ours.append((riot_name, league, riot_rank))
        print(f"  {riot_name:<25} {riot_rank:>4}    --    --  {league} (not in our top 20)")

print()
print("=" * 85)
print("BIGGEST OVERRATES (we rank much higher than Riot):")
print("=" * 85)
for name, league, rr, our, diff in sorted(big_over, key=lambda x: x[4]):
    print(f"  {name:<25} Riot #{rr:>2} vs Ours #{our:>2}  ({diff:+d})  {league}")

print()
print("=" * 85)
print("BIGGEST UNDERRATES (we rank much lower than Riot):")
print("=" * 85)
for name, league, rr, our, diff in sorted(big_under, key=lambda x: -x[4]):
    print(f"  {name:<25} Riot #{rr:>2} vs Ours #{our:>2}  ({diff:+d})  {league}")

print()
print("=" * 85)
print("MISSING FROM OUR TOP 20 (Riot has them, we don't):")
print("=" * 85)
for name, league, rr in missing_from_ours:
    print(f"  {name:<25} Riot #{rr:>2}  {league}")

# Regional analysis
print()
print("=" * 85)
print("REGIONAL BIAS ANALYSIS")
print("=" * 85)
print()
print("RIOT Regional Strength Scores:")
print("  LCK: 1586  |  LPL: 1353  |  LEC: 1169  |  LCP: 1156  |  LCS: 1084  |  CBLOL: 842")
print()
print("OUR Static League Base ELO:")
print("  Tier 1: LCK, LPL = 1600  (treats them as EQUAL)")
print("  Tier 2: LEC, LCS, LTA, LCP = 1500")
print("  Tier 3: CBLOL, PCS, VCS = 1450")
print()

# Compute avg rank diff by league
import collections
league_diffs = collections.defaultdict(list)
for riot_rank, riot_name, league, pts in riot:
    our_name = team_map.get(riot_name)
    our_rank = our_lookup.get(our_name) if our_name else None
    if our_rank:
        league_diffs[league].append(our_rank - riot_rank)

print("AVG RANK DIFFERENCE BY LEAGUE (negative = we overrate):")
for league in ['LCK', 'LPL', 'LEC', 'LCP', 'LCS', 'CBLOL']:
    diffs = league_diffs.get(league, [])
    if diffs:
        avg = sum(diffs) / len(diffs)
        sign = '+' if avg > 0 else ''
        print(f"  {league:>6}: {sign}{avg:.1f} avg rank diff  (n={len(diffs)} teams)")
    else:
        print(f"  {league:>6}: no matched teams")

print()
print("=" * 85)
print("DIAGNOSIS: WHY DO OUR RANKINGS DIVERGE?")
print("=" * 85)
print("""
1. FURIA (#7 us vs #38 Riot): +31 rank gap
   Root cause: intl_k_multiplier=2.0 on Americas Cup. 8-0 vs C9/SEN with K=40
   gave +138 ELO. Riot's CBLOL regional score is 842 (lowest!) — they heavily
   discount CBLOL results. We don't discount enough.

2. LCK UNDERVALUATION (T1 +3, HLE +10, KT +5):
   Our ELO system treats LCK = LPL (both tier 1, base 1600). Riot has LCK at
   1586 vs LPL at 1353 — a 233-point gap! This means:
   - LCK mid-table teams (HLE 2-3, KT 2-4) still get high Riot ranks because
     "losing in LCK" is worth more than "winning in LPL"
   - Our system penalizes HLE/KT for losses without accounting for LCK's
     overall dominance

3. LPL OVERVALUATION (BLG +2, AL +3, JDG +3):
   Mirror of #2. We treat LPL wins at full value; Riot discounts them relative
   to LCK. Anyone's Legend 10-4 in LPL = Riot #6; we have them #3.

4. MISSING TEAMS (FlyQuest #11, CTBC #8, Cloud9 #19):
   Likely name/ID mismatches in our data OR these teams don't appear in our
   active-2026 filter. FlyQuest at #11 is notable — they may have rebranded or
   our data doesn't have their 2026 games yet.

5. KARMINE CORP OVERRATE (#8 us vs #15 Riot):
   KC is 12-5 in LEC — strong record. But Riot discounts LEC more heavily
   (regional score 1169, below LCP at 1156). Our system gives full credit
   to LEC wins.

6. CONTEXT OF PLAY weighting:
   Riot explicitly weights playoffs > regular season. Our ELO system treats
   every game equally (same K regardless of stage). A team that performs at
   Worlds playoffs gets more Riot credit than regular season wins.
""")
