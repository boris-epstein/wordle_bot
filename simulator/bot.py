class WordleBot:
    def __init__(self, guess_list, answer_list):
        self.guess_list = guess_list        # all valid guess words
        self.answer_list = answer_list      # possible solution words
        self.possible_solutions = answer_list.copy()  # candidates, narrowed as we go

    def reset(self):
        """Reset between games if needed."""
        self.possible_solutions = self.answer_list.copy()

    def next_guess(self, guess_number, history):
        # Filter possible solutions based on feedback history
        pass


class NaiveBot(WordleBot):
    def next_guess(self, guess_number, history):
        return self.answer_list[0]


class JavaScriptBot(WordleBot):
    def __init__(self, guess_list, answer_list, starting_word="slate"):
        super().__init__(guess_list, answer_list)
        self.starting_word = starting_word.lower()

    def next_guess(self, guess_number, history):
        if guess_number == 1:
            return self.starting_word

        for guess, feedback in history:
            self.possible_solutions = [
                word for word in self.possible_solutions
                if self._feedback_matches(word, guess, feedback)
            ]

        return self.possible_solutions[0] if self.possible_solutions else self.guess_list[0]

    def _feedback_matches(self, candidate, guess, feedback):
        """
        Return True if the feedback between candidate and guess matches the given feedback list.
        Uses the same feedback logic as WordleGame.
        """
        candidate_freq = {}
        result = ["gray"] * len(candidate)

        for c in candidate:
            candidate_freq[c] = candidate_freq.get(c, 0) + 1

        # Mark greens
        for i in range(len(guess)):
            if guess[i] == candidate[i]:
                result[i] = "green"
                candidate_freq[guess[i]] -= 1

        # Mark yellows
        for i in range(len(guess)):
            if result[i] == "green":
                continue
            if candidate_freq.get(guess[i], 0) > 0:
                result[i] = "yellow"
                candidate_freq[guess[i]] -= 1

        return result == feedback
