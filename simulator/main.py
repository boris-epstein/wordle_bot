from wordle import WordleGame
from bot import NaiveBot
from simulator import WordleSimulator


import re
import ast

def load_words_from_js(file_path, variable_name="official_guesses"):
    """
    Reads a .js file and extracts the list assigned to a given variable.
    Assumes the variable assignment is of the form: const variable_name = ["WORD1", "WORD2", ...];
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Match the variable assignment using regex
    pattern = rf"{variable_name}\s*=\s*(\[[^\]]*\])"
    match = re.search(pattern, content)
    
    if not match:
        raise ValueError(f"Could not find array assigned to {variable_name}")
    
    array_str = match.group(1)

    # Convert JavaScript array to Python list safely
    word_list = ast.literal_eval(array_str.replace("'", '"'))  # Ensure valid JSON-style quotes
    return [word.lower() for word in word_list]  # Normalize to lowercase for consistency

if __name__ == '__main__':
    # answers = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Answers_with_ED.js'
    # guesses = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Guesses.js'
    # answer_list = load_words_from_js(answers, variable_name="official_answers_with_ed")
    # guess_list = load_words_from_js(guesses, variable_name="official_guesses")
    guess_list = ['bride', 'apple', 'crack', 'plays']
    answer_list = ['bride', 'apple']
    bot = NaiveBot(guess_list=guess_list, answer_list=answer_list)
    simulator = WordleSimulator(bot = bot)
    results = simulator.evaluate_all(answer_list)
    print(results)