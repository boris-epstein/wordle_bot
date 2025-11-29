

from wordle_env import WordleEnv
from utils import load_words_from_js, build_word_features
from policy import WordleMLPPolicy
from train_reinforce import train_reinforce


guesses_path = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Guesses.js'
# guess_list = load_words_from_js(guesses_path, variable_name='official_guesses')

GUESSES_JS_PATH = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Guesses.js'
train_reinforce(
        guesses_js_path=GUESSES_JS_PATH,
        checkpoint_dir="checkpoints_first",
        resume_from=None,      # e.g. "wordle_policy_ep5000.pt" to resume
        num_episodes=40_000,
        gamma=0.99,
        lr=1e-3,
        log_interval=100,
        save_interval=1000,
        device="cpu",          # or "cuda"
    )

