from wordle import WordleGame
from bot import NaiveBot, HeuristicWordleBot
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
    # answer_list = load_words_from_js(answers, variable_name="official_answers_with_ed")
    
    answers = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Answers.js'
    answer_list = load_words_from_js(answers, variable_name="official_answers")
    
    guesses = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Words.js'
    guess_list = load_words_from_js(guesses, variable_name="official_words")
    limit = 60
    print(len(guess_list))
    print(len(answer_list))
    # guess_list = ['bride', 'apple', 'crack', 'plays']
    # answer_list = ['bride', 'apple']
    bot = NaiveBot(guess_list=guess_list, answer_list=answer_list[:limit])
    simulator = WordleSimulator(bot = bot)
    results = simulator.evaluate_all(answer_list[:limit])
    print(results[0], results[1])
    
    bot2 = HeuristicWordleBot(guess_list=guess_list, answer_list=answer_list[:limit])
    simulator2 = WordleSimulator(bot = bot2)
    results2 = simulator2.evaluate_all(answer_list[:limit])
    print(results2[0], results2[1])
    
    # guess_list = ['bride', 'apple', 'crack', 'plays']
    # guess_list = load_words_from_js(guesses, variable_name="official_guesses")
    # game = WordleGame(word_list=guess_list, secret_word='chalk')
    # game.play()