import subprocess
import json
import random
import os
import concurrent.futures
from copy import deepcopy

# Configuration
MATCHES_PER_EVALUATION = 50 # Increase to 30-50 when running on the supercomputer
GENERATIONS = 20
WORKERS = 10 # Number of parallel CPU cores to use

# The bot you want to train against
OPPONENT = "Pigeon" 
MY_BOT = "Grapefruit"

def generate_random_weights():
    """Generate a random DNA strand of heuristic weights."""
    return {
        "prime_exp": random.uniform(1.1, 2.5),   # How aggressively to seek long carpets
        "prime_mult": random.uniform(0.1, 1.0),  # Overall value of primes vs raw points
        "center_bonus": random.uniform(0.0, 2.0), # Value of staying in the middle
        "rat_confidence_threshold": random.uniform(-1.0, 1.0)
    }

def run_single_match(weights):
    """Runs a single game in a subprocess and returns 1 if we win, 0 if we lose."""
    env = os.environ.copy()
    env['AGENT_WEIGHTS'] = json.dumps(weights)
    
    try:
        # Run the engine silently
        result = subprocess.run(
            ['python3', 'engine/run_local_agents.py', MY_BOT, OPPONENT],
            env=env,
            capture_output=True,
            text=True,
            timeout=180 # Prevent infinite loops from stalling the cluster
        )
        
        output = result.stdout
        if f"{MY_BOT} wins" in output or "Player A wins" in output:
            return 1 # Win
        elif "Draw" in output:
            return 0.5 # Draw
        else:
            return 0 # Loss
    except subprocess.TimeoutExpired:
        return 0 # Timed out bots lose automatically

def evaluate_weights(weights):
    """Plays N matches with these weights and returns the win rate."""
    wins = 0
    # Run games in parallel to abuse the supercomputer's CPU cores
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run_single_match, weights) for _ in range(MATCHES_PER_EVALUATION)]
        for future in concurrent.futures.as_completed(futures):
            wins += future.result()
            
    win_rate = wins / MATCHES_PER_EVALUATION
    return win_rate, weights

def mutate(weights):
    """Takes a good set of weights and slightly tweaks them."""
    new_weights = deepcopy(weights)
    key_to_mutate = random.choice(list(new_weights.keys()))
    # Mutate by +/- 15%
    new_weights[key_to_mutate] *= random.uniform(0.85, 1.15)
    return new_weights

if __name__ == "__main__":
    print("🚀 Starting Cluster Evolution Engine...")
    
    # 1. Generate initial population of random bots
    population = [generate_random_weights() for _ in range(WORKERS)]
    best_overall_weights = None
    best_overall_winrate = -1

    for generation in range(GENERATIONS):
        print(f"\n--- GENERATION {generation + 1} ---")
        results = []
        
        # 2. Evaluate the entire population
        for idx, weights in enumerate(population):
            print(f"Evaluating Bot {idx+1}/{len(population)}...", end="", flush=True)
            win_rate, w = evaluate_weights(weights)
            results.append((win_rate, w))
            print(f" Win Rate: {win_rate*100:.1f}%")
            
        # 3. Sort by best performance
        results.sort(key=lambda x: x[0], reverse=True)
        best_gen_winrate, best_gen_weights = results[0]
        
        if best_gen_winrate > best_overall_winrate:
            best_overall_winrate = best_gen_winrate
            best_overall_weights = best_gen_weights
            
        print(f"🏆 Best this generation: {best_gen_winrate*100:.1f}% win rate.")
        print(f"🧬 Best DNA: {json.dumps(best_gen_weights, indent=2)}")
        
        # Save progress to a file so you don't lose data if the cluster shuts down
        with open("optimal_weights.json", "w") as f:
            json.dump(best_overall_weights, f, indent=4)
            
        # 4. Create next generation (Survival of the fittest)
        # Keep the top 20% untouched
        survivors = [w for r, w in results[:max(2, int(WORKERS * 0.2))]]
        population = list(survivors)
        
        # Fill the rest with mutations of the survivors
        while len(population) < WORKERS:
            parent = random.choice(survivors)
            population.append(mutate(parent))
            
    print("\n✅ Evolution Complete!")
    print(f"Final Optimal Weights (Win Rate: {best_overall_winrate*100}%):")
    print(json.dumps(best_overall_weights, indent=2))