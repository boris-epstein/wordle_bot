

from bot import WordleBot
from wordle import WordleGame
from typing import List

class WordleSimulator:
    def __init__(self, bot: WordleBot, max_guesses=6):
        self.bot = bot
        self.max_guesses = max_guesses


    def play_one(self, wordle_game: WordleGame):
        """Simulate a game with the given secret word. Returns guesses used or None if failed."""
        self.bot.reset()
        
        history = []
        for guess_num in range(1, self.max_guesses+1):
            guess = self.bot.next_guess(guess_num, history)
            if guess == wordle_game.secret_word:
                return guess_num  # win in guess_num attempts
            feedback = wordle_game.guess(guess)  # produce Wordle feedback pattern
            history.append((guess, feedback))
        return None  # failed to guess within max_guesses

    def evaluate_all(self, secrets_list: List[str]):
        results = {}
        wins = 0
        total_guesses = 0
        for secret in secrets_list:
            game = WordleGame(word_list=self.bot.guess_list, secret_word=secret, max_attempts=self.max_guesses)
            attempts = self.play_one(game)
            results[secret] = attempts  # store None if failed, or number of guesses if solved
            if attempts is not None:
                wins += 1
                total_guesses += attempts
        win_rate = wins / len(secrets_list)
        avg_guesses = total_guesses / wins if wins > 0 else None
        return wins, win_rate, avg_guesses, results
