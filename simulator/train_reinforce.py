
import os
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim

from wordle_env import WordleEnv
from utils import load_words_from_js, build_word_features
from policy import WordleMLPPolicy


def make_env_and_policy(
    guesses_js_path: str,
    variable_name: str = "official_words",
    max_guesses: int = 6,
    checkpoint_path: Optional[str] = None,
    hidden_dim: int = 256,
    device: str = "cpu",
):
    """
    Build WordleEnv + WordleMLPPolicy.
    If checkpoint_path is provided and exists, load weights from there.
    """
    # 1. Load word list from your JS file
    if guesses_js_path == '':
        word_list = [
                "apple", "brave", "cider", "drape", "eager",
                "flint", "grace", "honey", "ivory", "jolly",
                "knack", "lemon", "morph", "noble", "ocean",
                "prism", "quart", "rival", "sunny", "tiger",
                "umbra", "vigor", "woven", "xenon", "youth",
                "zesty", "adorn", "blitz", "crisp", "dough",
                "ember", "frost", "gleam", "hoist", "inlet",
                "jumpy", "kudos", "latch", "mango", "nylon",
                "octet", "piano", "quill", "raven", "slope",
                "tulip", "upper", "vivid", "waltz", "xerox",
                "yearn", "zonal", "align", "broom", "chant",
                "dizzy", "enjoy", "flair", "grind", "haste",
                "irate", "joint", "kneel", "lodge", "mirth",
                "noisy", "olive", "pleat", "quark", "rouse",
                "shiny", "throb", "unify", "vapor", "whale",
                "xylem", "yield", "zebra", "amend", "bland",
                "crown", "dwell", "earth", "feast", "giant",
                "hatch", "ideal", "jelly", "koala", "laser",
                "medal", "naval", "opine", "pilot", "quake",
                "risky", "stout", "trail", "union", "vowel"
            ]

    
    else:
        word_list: List[str] = load_words_from_js(guesses_js_path, variable_name=variable_name)
    print(f"Loaded {len(word_list)} candidate words.")

    # 2. Create environment
    env = WordleEnv(word_list=word_list, solution_list=word_list, max_guesses=max_guesses)

    # 3. Build word_features tensor
    word_features = build_word_features(word_list)  # [num_words, 26, 5]
    word_features = word_features.to(device)

    # 4. Create or load policy
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading policy weights from: {checkpoint_path}")
        policy = WordleMLPPolicy.load_from_file(
            checkpoint_path,
            word_features=word_features,
            max_guesses=max_guesses,
            hidden_dim=hidden_dim,
            map_location=device,
        )
    else:
        print("Initializing new policy.")
        policy = WordleMLPPolicy(
            word_features=word_features,
            max_guesses=max_guesses,
            hidden_dim=hidden_dim,
        )

    policy.to(device)
    return env, policy


def compute_returns(rewards, gamma: float):
    """
    Compute discounted returns G_t for an episode.
    rewards: list of scalars [r_0, r_1, ..., r_T]
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def train_reinforce(
    guesses_js_path: str,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
    num_episodes: int = 10_000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    log_interval: int = 100,
    save_interval: int = 1_000,
    device: str = "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If resuming, we'll pass this path to make_env_and_policy
    checkpoint_path = None
    if resume_from is not None:
        checkpoint_path = os.path.join(checkpoint_dir, resume_from)

    env, policy = make_env_and_policy(
        guesses_js_path=guesses_js_path,
        variable_name="official_words",   # adapt if needed
        max_guesses=6,
        checkpoint_path=checkpoint_path,
        hidden_dim=256,
        device=device,
    )

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False

        log_probs = []
        rewards = []

        while not done:
            num_guesses = info["num_guesses"]  # from env.step info

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)          # [26, 5, 3]
            guess_tensor = torch.tensor(num_guesses, dtype=torch.long, device=device)   # scalar

            # Get distribution over word indices
            dist = policy.get_distribution(
                obs_tensor.unsqueeze(0),   # [1, 26, 5, 3]
                guess_tensor.unsqueeze(0), # [1]
            )

            action = dist.sample()          # [1]
            log_prob = dist.log_prob(action)[0]  # scalar

            action_idx = action.item()

            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        # Episode finished: compute returns and policy gradient loss
        returns = compute_returns(rewards, gamma=gamma)  # [T]
        if len(returns) > 1:
            # Optional: normalize returns to reduce variance
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_tensor = torch.stack(log_probs)  # [T]

        loss = -(returns.to(device) * log_probs_tensor).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        # Logging
        if episode % log_interval == 0:
            avg_reward = sum(episode_rewards[-log_interval:]) / log_interval
            print(f"Episode {episode}/{num_episodes} | avg reward (last {log_interval}): {avg_reward:.3f}")

        # Periodic checkpointing
        if episode % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"wordle_policy_ep{episode}.pt")
            print(f"Saving checkpoint to: {ckpt_path}")
            policy.save(ckpt_path)

    # Final save
    final_ckpt = os.path.join(checkpoint_dir, f"wordle_policy_final.pt")
    print(f"Saving final checkpoint to: {final_ckpt}")
    policy.save(final_ckpt)


if __name__ == "__main__":
    # TODO: set this to your actual JS path
    GUESSES_JS_PATH = "/path/to/Words.js"

    train_reinforce(
        guesses_js_path=GUESSES_JS_PATH,
        checkpoint_dir="checkpoints",
        resume_from=None,      # e.g. "wordle_policy_ep5000.pt" to resume
        num_episodes=10_000,
        gamma=0.99,
        lr=1e-3,
        log_interval=100,
        save_interval=1000,
        device="cpu",          # or "cuda"
    )
