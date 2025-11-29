

from wordle_env import WordleEnv
from utils import load_words_from_js, build_word_features
from policy import WordleMLPPolicy

import torch

# 1. Build word list (whatever you’re already using)
guesses_path = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Guesses.js'
guess_list = load_words_from_js(guesses_path, variable_name='official_guesses')
word_list = guess_list.copy()
# word_list = load_words_from_js(guesses_path, variable_name="official_words")
# word_list = ["crane", "slate", "trace", "spite", "tried"]  # toy example

# 2. Build env
env = WordleEnv(word_list=word_list, solution_list=word_list, max_guesses=6)

# 3. Precompute word encodings
word_features = build_word_features(word_list)  # [num_words, 26, 5]

# 4. Create policy
policy = WordleMLPPolicy(word_features=word_features, max_guesses=env.max_guesses)

# 5. Run one episode with random policy actions from the network
obs, info = env.reset()
done = False

while not done:
    num_guesses = info["num_guesses"]

    obs_tensor = torch.tensor(obs, dtype=torch.float32)       # [26, 5, 3]
    guess_tensor = torch.tensor(num_guesses, dtype=torch.long)

    action, log_prob = policy.act(obs_tensor, guess_tensor, greedy=False)
    action_idx = action.item()

    obs, reward, terminated, truncated, info = env.step(action_idx)
    # env.render()

    done = terminated or truncated

env.render()
print("Reward:", reward)