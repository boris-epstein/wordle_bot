import random

class WordleGame:
    def __init__(self, word_list, secret_word=None, max_attempts=6):
        """
        Initialize the Wordle game with a list of valid words and an optional secret word.
        If no secret_word is provided, a random word from word_list will be chosen.
        """
        if not word_list:
            raise ValueError("Word list must not be empty.")
        # Store all words in lower-case for consistency
        self.word_list = [word.lower() for word in word_list]
        self.max_attempts = max_attempts
        # Choose the secret word (use provided one or random choice from list)
        if secret_word:
            secret_word = secret_word.lower()
            if secret_word not in self.word_list:
                raise ValueError("Secret word must be in the provided word list.")
            self.secret_word = secret_word
        else:
            self.secret_word = random.choice(self.word_list)
        # Initialize game state
        self.attempts = 0
        self.solved = False
        self.history = []  # to record guesses and feedback
    
    def guess(self, word):
        """
        Make a guess and get feedback on each letter as 'green', 'yellow', or 'gray'.
        Returns a list of feedback strings corresponding to the guessed word.
        """
        if self.solved or self.attempts >= self.max_attempts:
            raise RuntimeError("Game is over. No more guesses allowed.")
        word = word.lower()
        # Validate guess length and presence in word list
        if len(word) != len(self.secret_word):
            raise ValueError(f"Guess must be a {len(self.secret_word)}-letter word.")
        if word not in self.word_list:
            raise ValueError("Guess is not in the list of valid words.")
        # Evaluate the guess against the secret word to get feedback
        feedback = self._evaluate_guess(word)
        # Record this attempt
        self.attempts += 1
        self.history.append((word, feedback))
        # Check if the guess was correct
        if word == self.secret_word:
            self.solved = True
        return feedback
    
    def _evaluate_guess(self, guess):
        """
        Compute the Wordle feedback for a guess compared to the secret word.
        Returns a list of 'green', 'yellow', 'gray' strings for each letter.
        """
        # Initialize feedback as all "gray"
        feedback = ["gray"] * len(self.secret_word)
        # Frequency dictionary for letters in the secret word
        secret_freq = {}
        for letter in self.secret_word:
            secret_freq[letter] = secret_freq.get(letter, 0) + 1
        # First pass: mark greens and decrement frequency for matched letters
        for i, letter in enumerate(guess):
            if letter == self.secret_word[i]:
                feedback[i] = "green"
                secret_freq[letter] -= 1
        # Second pass: mark yellows (correct letter, wrong position)
        for i, letter in enumerate(guess):
            if feedback[i] == "green":
                continue  # skip letters already marked green
            if secret_freq.get(letter, 0) > 0:
                # The letter is present in secret (not yet fully matched by greens/yellows)
                feedback[i] = "yellow"
                secret_freq[letter] -= 1
            else:
                # Letter not in secret (or no instances left to match)
                feedback[i] = "gray"
        return feedback
    
    def is_over(self):
        """Check if the game has ended (solved or out of attempts)."""
        return self.solved or self.attempts >= self.max_attempts
    
    def remaining_attempts(self):
        """Return the number of guesses remaining."""
        return self.max_attempts - self.attempts
    
    def reset(self, new_secret=None):
        """
        Reset the game to start over. Optionally specify a new secret word.
        If no new_secret is given, a random secret word will be chosen.
        """
        if new_secret:
            new_secret = new_secret.lower()
            if new_secret not in self.word_list:
                raise ValueError("New secret word must be in the word list.")
            self.secret_word = new_secret
        else:
            self.secret_word = random.choice(self.word_list)
        self.attempts = 0
        self.solved = False
        self.history.clear()
    
    def play(self):
        """
        Play an interactive game in the console. 
        This will prompt the user for guesses until the game ends.
        """
        print("Welcome to Wordle!")
        print(f"The secret word has {len(self.secret_word)} letters. You have {self.max_attempts} attempts.")
        # Loop until the game is over
        while not self.is_over():
            guess_word = input("Enter your guess: ").strip()
            try:
                feedback = self.guess(guess_word)
            except ValueError as ve:
                # Invalid guess (wrong length or not in list)
                print(f"Invalid guess: {ve}")
                continue
            except RuntimeError as re:
                # No attempts left or already solved
                print(re)
                break
            # Display feedback using colored squares (like Wordle)
            result_display = ""
            for letter, status in zip(guess_word, feedback):
                if status == "green":
                    result_display += "🟩"  # green square
                elif status == "yellow":
                    result_display += "🟨"  # yellow square
                else:
                    result_display += "⬛"   # black square
            print(f"Feedback: {result_display}")
            if self.solved:
                print("Congratulations! You guessed the word correctly!")
                break
        if not self.solved:
            print(f"Game over. The secret word was '{self.secret_word}'.")



if __name__ == '__main__':
    guess_list = ['bride', 'apple', 'crack', 'plays']
    # guess_list = load_words_from_js(guesses, variable_name="official_guesses")
    game = WordleGame(word_list=guess_list, secret_word='hopes')
    game.play()