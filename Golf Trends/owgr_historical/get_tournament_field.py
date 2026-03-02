"""
Scrape ESPN Golf leaderboard to get list of players in a tournament.
"""

import json
import os
import re

import requests
from bs4 import BeautifulSoup


def get_tournament_players(tournament_id: str) -> list[dict]:
    """
    Scrape ESPN leaderboard to get all players who participated in a tournament.
    
    Args:
        tournament_id: ESPN tournament ID (e.g., '401465526')
        
    Returns:
        List of dicts with player info: name, espn_id, position
    """
    url = f"https://www.espn.com/golf/leaderboard?tournamentId={tournament_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    players = []
    
    # Find all leaderboard table rows
    # Each row contains position and player info
    rows = soup.find_all('tr', class_=re.compile(r'PlayerRow'))
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        
        # Position is in the second cell (index 1) - first cell is caret/expand icon
        pos_text = cells[1].get_text(strip=True)
        # Handle tied positions (e.g., "T3" -> 3) and CUT players (show as "-")
        pos_clean = pos_text.replace('T', '').replace('-', '').strip()
        try:
            position = int(pos_clean) if pos_clean else 999
        except ValueError:
            position = 999
        
        # Find player link (has class "leaderboard_player_name")
        player_link = row.find('a', class_='leaderboard_player_name')
        if not player_link:
            # Fallback to any player link
            player_link = row.find('a', href=re.compile(r'/golf/player/_/id/\d+'))
        if not player_link:
            continue
            
        href = player_link.get('href', '')
        name = player_link.get_text(strip=True)
        
        # Skip empty names or non-player links
        if not name or name in ['Player Stats', 'Course Stats']:
            continue
            
        # Extract ESPN player ID from URL
        match = re.search(r'/golf/player/_/id/(\d+)', href)
        espn_id = match.group(1) if match else None
        
        # Avoid duplicates
        if not any(p['espn_id'] == espn_id for p in players):
            players.append({
                'name': name,
                'espn_id': espn_id,
                'position': position,
            })
    
    return players


def get_player_names(tournament_id: str) -> list[str]:
    """
    Get just the player names from a tournament.
    
    Args:
        tournament_id: ESPN tournament ID
        
    Returns:
        List of player names
    """
    players = get_tournament_players(tournament_id)
    return [p['name'] for p in players]


def get_player_results(tournament_id: str) -> dict[str, dict]:
    """
    Get player results with finishing positions.
    
    Args:
        tournament_id: ESPN tournament ID
        
    Returns:
        Dict of {PlayerName: {"Finishing Position": position}}
    """
    players = get_tournament_players(tournament_id)
    return {p['name']: {'Finishing Position': p['position']} for p in players}


if __name__ == '__main__':
    # Example: 2023 THE PLAYERS Championship
    tournament_id = '401811934'
    tournament_name = 'Cognizant2026'
    
    players = get_tournament_players(tournament_id)
    
    # Path to the JSON file (same directory as this script)
    json_path = os.path.join(os.path.dirname(__file__), 'tournament_fields.json')
    
    # Load existing data or create empty dict
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            tournament_fields = json.load(f)
    else:
        tournament_fields = {}
    
    # Add this tournament's players with finishing positions
    tournament_fields[tournament_name] = {
        p['name']: {'Finishing Position': p['position']} for p in players
    }
    
    # Save back to JSON
    with open(json_path, 'w') as f:
        json.dump(tournament_fields, f, indent=2)
    
    print(f"Saved {len(players)} players for '{tournament_name}' to {json_path}")
