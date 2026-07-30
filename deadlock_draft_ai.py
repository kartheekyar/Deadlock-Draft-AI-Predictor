import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# 1. FETCH LIVE ACTIVE HEROES FROM THE API
def fetch_deadlock_heroes():
    """Fetches the live playable characters from the public Deadlock API."""
    url = "https://deadlock-api.com"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {hero['id']: hero['name'] for hero in response.json()}
    except Exception:
        print("API offline or rate-limited. Bootstrapping with fallback meta character dictionary...")

    # Static fallback dictionary matching internal game data if network fails
    return {1: "Abrams", 2: "Bebop", 3: "Dynamo", 4: "Grey_Thorn", 5: "Haze",
            6: "Infernus", 7: "Ivy", 8: "Kelvin", 9: "Lash", 10: "McGinnis",
            11: "Mo_And_Krill", 12: "Paradox", 13: "Pocket", 14: "Seven",
            15: "Vindicta", 16: "Viscous", 17: "Warden", 18: "Wraith", 19: "Yamato"}


# 2. GENERATE A COMPOSITION META TRAINING DATASET
def generate_meta_training_data(hero_map, num_matches=3000):
    """
    Creates structural synthetic game match files to train the core engine.
    Applies algebraic weights simulating lane counters and wombocombo meta.
    """
    hero_ids = list(hero_map.keys())
    match_records = []

    for _ in range(num_matches):
        # Pick 12 unique heroes out of the active pool (6 for Team A, 6 for Team B)
        selected_draft = np.random.choice(hero_ids, size=12, replace=False)
        team_a = selected_draft[:6]
        team_b = selected_draft[6:]

        # Performance scaling scoring loops (AI pattern mapping)
        score_a = len(team_a)
        score_b = len(team_b)

        # --- DEFINE HERO ROLES BASED ON ID ---
        tanks = {1, 11, 17}
        supports = {3, 7, 8, 16}
        spirit_carries = {14, 2, 13, 6}
        weapon_carries = {5, 18, 15, 10}

        # --- EVALUATE TEAM A ---
        team_a_set = set(team_a)
        # 1. Role Balance Checks (Penalize if missing key fundamentals)
        if not team_a_set.intersection(tanks): score_a -= 1.5  # No frontline tank
        if not team_a_set.intersection(supports): score_a -= 1.0  # No crowd-control or healing utilities

        # 2. Damage Diversity Check (Penalize if entirely one element type)
        if team_a_set.issubset(spirit_carries | supports | tanks):
            score_a -= 2.0  # Lacks weapon scaling; easily countered by spirit armor items

        # 3. Synergy Combinations (Wombo-combos)
        if 14 in team_a_set and 3 in team_a_set: score_a += 2.5  # Seven + Dynamo (Singularity into Storm)
        if 1 in team_a_set and 8 in team_a_set: score_a += 1.5  # Abrams + Kelvin (Heal rate and speed buffing)

        # --- EVALUATE TEAM B ---
        team_b_set = set(team_b)
        # 1. Role Balance Checks
        if not team_b_set.intersection(tanks): score_b -= 1.5
        if not team_b_set.intersection(supports): score_b -= 1.0

        # 2. Damage Diversity Check
        if team_b_set.issubset(spirit_carries | supports | tanks):
            score_b -= 2.0

        # 3. Synergy Combinations
        if 14 in team_b_set and 3 in team_b_set: score_b += 2.5
        if 1 in team_b_set and 8 in team_b_set: score_b += 1.5

        # --- CROSS-TEAM DIRECT COUNTERS (A vs B) ---
        if 2 in team_a_set and 15 in team_b_set: score_a += 1.5  # Bebop hook counters sniper Vindicta
        if 2 in team_b_set and 15 in team_a_set: score_b += 1.5

        if 8 in team_a_set and 5 in team_b_set: score_a += 1.2  # Kelvin ice dome isolates/counters Haze ultimate
        if 8 in team_b_set and 5 in team_a_set: score_b += 1.2

        if 17 in team_a_set and 18 in team_b_set: score_a += 1.0  # Warden's hard root locks down mobile Wraith
        if 17 in team_b_set and 18 in team_a_set: score_b += 1.0

        # Compute winning vector using soft stochastic variation
        is_team_a_winner = 1 if (score_a + np.random.normal(0, 1.2)) > score_b else 0

        # Structure vector footprint: 1 = Team A, -1 = Team B, 0 = Banned/Unpicked
        match_vector = {hid: 0 for hid in hero_ids}
        for hero in team_a: match_vector[hero] = 1
        for hero in team_b: match_vector[hero] = -1
        match_vector['match_outcome'] = is_team_a_winner

        match_records.append(match_vector)

    return pd.DataFrame(match_records)


# 3. INITIALIZE AND TRAIN THE ARTIFICIAL INTELLIGENCE MODEL
def train_ai_model(dataframe):
    """Fits data matrix into a scikit-learn Random Forest Classifier."""
    X = dataframe.drop(columns=['match_outcome'])  # Features
    y = dataframe['match_outcome']  # Target label

    classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    classifier.fit(X, y)
    return classifier


# 4. PREDICT MATCHUP SYNERGY PROBABILITIES
def predict_draft_win_rate(model, hero_map, ally_lineup, enemy_lineup):
    """Maps custom string rosters to data arrays to output a win percentage."""
    normalize_map = {name.lower(): hid for hid, name in hero_map.items()}
    input_footprint = {hid: 0 for hid in hero_map.keys()}

    for hero_name in ally_lineup:
        hero_id = normalize_map.get(hero_name.lower())
        if hero_id: input_footprint[hero_id] = 1

    for hero_name in enemy_lineup:
        hero_id = normalize_map.get(hero_name.lower())
        if hero_id: input_footprint[hero_id] = -1

    prediction_row = pd.DataFrame([input_footprint])

    # Extract prediction probability output vector [[loss_probability, win_probability]]
    win_probabilities = model.predict_proba(prediction_row)[0]
    win_percentage = win_probabilities[1] * 100
    return win_percentage


# --- MAIN EXECUTION FRAMEWORK ---
if __name__ == "__main__":
    print("==============================================")
    print("   DEADLOCK MACHINE LEARNING DRAFT ENGINE    ")
    print("==============================================\n")

    # Pipeline Step 1: Active Ingestion
    hero_dictionary = fetch_deadlock_heroes()
    print(f"[*] Ingested {len(hero_dictionary)} active characters from network data tree.")

    # Pipeline Step 2 & 3: Data modeling and fitting
    print("[*] Generating match matrix histories and fitting training nodes...")
    dataset = generate_meta_training_data(hero_dictionary, num_matches=4000)
    trained_brain = train_ai_model(dataset)
    print("[+] Machine learning ensemble successfully compiled and deployed.\n")

    # Pipeline Step 4: Scenario Matchup simulation
    # Blue has strong synergy (Seven + Dynamo) and a balanced team
    team_blue = ["Seven", "Dynamo", "Haze", "Abrams", "Wraith", "Ivy"]
    # Orange is missing a tank, missing support, and heavily compositionally countered
    team_orange = ["Bebop", "Infernus", "Warden", "Paradox", "Lash", "Vindicta"]

    calculated_win_rate = predict_draft_win_rate(trained_brain, hero_dictionary, team_blue, team_orange)

    print("----------------------------------------------")
    print("             MATCHUP EVALUATION               ")
    print("----------------------------------------------")
    print(f" Blue Squad (Ally):  {', '.join(team_blue)}")
    print(f" Orange Squad (Enemy): {', '.join(team_orange)}")
    print("----------------------------------------------")
    print(f"[AI Verdict]: Calculated Win Probability: {calculated_win_rate:.2f}%\n")