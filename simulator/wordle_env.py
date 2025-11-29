from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from constants import Feedback

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from functools import cache

from utils import evaluate_guess

# -----------------------------
# Feedback computation (yours)
# -----------------------------

def get_word_count(answer: List[str]) -> Dict[str, int]:
    # Frequency dictionary for letters in the secret word
    secret_freq = {}
    for letter in answer:
        secret_freq[letter] = secret_freq.get(letter, 0) + 1
    return secret_freq




# -----------------------------
# Wordle Gymnasium Environment
# -----------------------------

class WordleEnv(gym.Env):
    """
    Wordle environment with:
      - action_space: Discrete(num_words), action = index into word_list
      - observation_space: Box(shape=(26, 5, 3)), dtype=int8
        obs[letter_idx, position, channel] = 1 if we've seen that feedback
        channel 0 = gray, 1 = yellow, 2 = green
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        word_list: List[str],
        solution_list: Optional[List[str]] = None,
        max_guesses: int = 6,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert len(word_list) > 0, "word_list must be non-empty"
        self.word_list = word_list
        self.solution_list = solution_list if solution_list is not None else word_list
        self.max_guesses = max_guesses

        # Simple checks: all words same length, lower-case alpha, etc.
        word_len = len(self.word_list[0])
        assert all(len(w) == word_len for w in self.word_list), "All words must have same length"
        assert word_len == 5, "This env assumes standard 5-letter Wordle"

        self.word_len = word_len
        self._rng = random.Random(seed)

        # Action space: index of guess in word_list
        self.action_space = spaces.Discrete(len(self.word_list))

        # Observation: (26 letters, 5 positions, 3 feedback types)
        # We'll store values in {0,1}, dtype=int8
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(26, self.word_len, 3),
            dtype=np.int8,
        )

        # Internal state
        self.secret_word: str = ""
        self.num_guesses: int = 0
        self.obs: np.ndarray = np.zeros(self.observation_space.shape, dtype=np.int8)
        self.history: List[Tuple[str, List[str]]] = []  # (guess, feedback)

    # Gymnasium reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)

        self.secret_word = self._rng.choice(self.solution_list)
        self.num_guesses = 0
        self.history = []
        self.obs = np.zeros(self.observation_space.shape, dtype=np.int8)

        info = {
            "num_guesses": self.num_guesses,
            # You normally *wouldn't* expose the answer,
            # but sometimes it's useful for debugging:
            # "secret_word": self.secret_word,
        }
        return self.obs.copy(), info

    # Gymnasium step
    def step(self, action: int):
        # Map action index -> guessed word
        assert self.action_space.contains(action), "Invalid action index"
        guess = self.word_list[int(action)]
        self.num_guesses += 1

        # Compute Wordle feedback using your function
        feedback = evaluate_guess(guess, self.secret_word)
        self.history.append((guess, feedback))

        # Update observation with the new feedback
        self._update_obs_with_feedback(guess, feedback)

        # Termination logic
        terminated = (guess == self.secret_word)
        truncated = (self.num_guesses >= self.max_guesses and not terminated)

        # Reward: 1 if guessed correctly, 0 otherwise (you can change this)
        reward = 1.0 if terminated else 0.0

        info = {
            "num_guesses": self.num_guesses,
            "last_guess": guess,
            "last_feedback": feedback,
            # For analysis after an episode:
            # "secret_word": self.secret_word,
        }

        return self.obs.copy(), reward, terminated, truncated, info

    # Encode feedback into obs[26, 5, 3]
    def _update_obs_with_feedback(self, guess: str, feedback: List[Feedback]) -> None:
        """
        For each position i:
          - letter index l = ord(guess[i]) - ord('a')
          - channel: 0=gray, 1=yellow, 2=green
        We set obs[l, i, channel] = 1 and leave previous info intact.
        """
        for i, (ch, fb) in enumerate(zip(guess, feedback)):
            l_idx = ord(ch) - ord('a')
            if l_idx < 0 or l_idx >= 26:
                # Not strictly needed if your word list is clean
                continue

            self.obs[l_idx, i, fb.value] = 1

    # Simple text renders
    def render(self):
        print(f"Secret word: {'?????' if self.num_guesses < self.max_guesses else self.secret_word}")
        for guess, feedback in self.history:
            print(f"{guess}  ->  {[fb.name for fb in feedback]}")
        print(f"Guesses used: {self.num_guesses}/{self.max_guesses}")


if __name__ == "__main__":
    # Tiny example word list
    words = ["crane", "slate", "trace", "spite", "tried"]

    env = WordleEnv(word_list=words, solution_list=words, max_guesses=6)

    obs, info = env.reset()
    done = False

    while not done:
        # random guess policy for demo
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

    print("Final reward:", reward)
