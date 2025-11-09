


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