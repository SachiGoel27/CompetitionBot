import json
import os
import subprocess
from collections import Counter

MY_BOT = "Grapefruit"
OPPONENT = "Yolanda"
NUM_MATCHES = 100

AGENT_WEIGHTS = None

def classify_result(stdout: str) -> str:
    text = stdout.lower()

    if "draw" in text:
        return "draw"

    if MY_BOT.lower() in text and "wins" in text:
        return "win"
    if OPPONENT.lower() in text and "wins" in text:
        return "loss"

    if "player a wins" in text:
        return "win"
    if "player b wins" in text:
        return "loss"

    return "unknown"

def run_single_match(match_num: int) -> str:
    env = os.environ.copy()
    if AGENT_WEIGHTS is not None:
        env["AGENT_WEIGHTS"] = json.dumps(AGENT_WEIGHTS)

    result = subprocess.run(
        ["python3", "engine/run_local_agents.py", MY_BOT, OPPONENT],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )

    outcome = classify_result(result.stdout)
    print(f"Match {match_num:03d}: {outcome}")
    return outcome

def main():
    results = []
    for i in range(1, NUM_MATCHES + 1):
        try:
            outcome = run_single_match(i)
        except subprocess.TimeoutExpired:
            outcome = "timeout"
            print(f"Match {i:03d}: timeout")
        except Exception as e:
            outcome = "error"
            print(f"Match {i:03d}: error ({e})")
        results.append(outcome)

    counts = Counter(results)
    summary = {
        "my_bot": MY_BOT,
        "opponent": OPPONENT,
        "num_matches": NUM_MATCHES,
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "timeouts": counts["timeout"],
        "errors": counts["error"],
        "unknown": counts["unknown"],
        "win_rate": counts["win"] / NUM_MATCHES,
        "non_loss_rate": (counts["win"] + counts["draw"]) / NUM_MATCHES,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with open("match_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()