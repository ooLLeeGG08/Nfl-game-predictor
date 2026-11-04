"""Current NFL teams: display names, divisions and conferences."""

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SEA", "SF"],
}

NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

DIVISION_OF = {team: div for div, teams in DIVISIONS.items() for team in teams}
CONFERENCE_OF = {team: div.split()[0] for team, div in DIVISION_OF.items()}


def same_division(a: str, b: str) -> bool:
    return DIVISION_OF[a] == DIVISION_OF[b]


# Brand-ish colors, used only for team abbreviations and probability bars.
# Where a team's primary is too dark for a black page, a secondary brand color
# is used instead; anything still below WCAG AA is lightened in readable_color().
BRAND_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#FF3C00",
    "DAL": "#869397", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#FFB612",
    "HOU": "#A71930", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LA": "#FFA300", "LAC": "#0080C6", "LV": "#A5ACAF", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#C60C30", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#FFB612",
}

CARD_BG = "#161616"


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def readable_color(hex_color: str, bg: str = CARD_BG, target: float = 4.5) -> str:
    """Mix toward white in small steps until text in this color meets WCAG AA on ``bg``."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    color = hex_color
    for step in range(21):
        t = step / 20
        color = "#" + "".join(f"{round(c + (255 - c) * t):02X}" for c in rgb)
        if contrast(color, bg) >= target:
            break
    return color


TEAM_COLORS = {team: readable_color(c) for team, c in BRAND_COLORS.items()}
