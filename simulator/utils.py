from typing import List, Dict, Tuple, Optional
import re
import ast
from functools import cache
from constants import Feedback
import torch



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

def get_word_count(answer:List[str])-> Dict[str,int]:
    # Frequency dictionary for letters in the secret word
    secret_freq = {}
    for letter in answer:
        secret_freq[letter] = secret_freq.get(letter, 0) + 1
    
    return secret_freq

@cache
def evaluate_guess(guess: str, answer: str) -> List[Feedback]:
    """
    Same logic as your evaluate_guess, but returns [FB.GRAY, FB.GREEN, ...].
    """
    secret_freq = get_word_count(list(answer))
    feedback = [Feedback.GRAY] * len(answer)

    # First pass: greens
    for i, letter in enumerate(guess):
        if letter == answer[i]:
            feedback[i] = Feedback.GREEN
            secret_freq[letter] -= 1

    # Second pass: yellows
    for i, letter in enumerate(guess):
        if feedback[i] is Feedback.GREEN:
            continue
        if secret_freq.get(letter, 0) > 0:
            feedback[i] = Feedback.YELLOW
            secret_freq[letter] -= 1
        else:
            feedback[i] = Feedback.GRAY

    return feedback

def cache_all_feedback(guess_list: List[str], answer_list: List[str])-> Dict[Tuple[str, str],List[str]]:

    feedback = {}
    
    for answer in answer_list:
        
        for guess in guess_list:
            
            feedback[guess, answer] = evaluate_guess(guess, answer)
            
    return feedback

def build_word_features(word_list: List[str]) -> torch.Tensor:
    """
    Build a [num_words, 26, 5] tensor encoding each word.
    word_features[w, l, p] = 1 if word_list[w][p] == chr(ord('a') + l), else 0.
    Assumes 5-letter lowercase words.
    """
    num_words = len(word_list)
    word_len = len(word_list[0])
    assert all(len(w) == word_len for w in word_list), "All words must have the same length"
    assert word_len == 5, "This helper assumes 5-letter Wordle words"

    features = torch.zeros(num_words, 26, word_len, dtype=torch.float32)
    for w_idx, word in enumerate(word_list):
        for p, ch in enumerate(word):
            l = ord(ch) - ord("a")
            if 0 <= l < 26:
                features[w_idx, l, p] = 1.0
            else:
                raise ValueError(f"Non a–z character '{ch}' in word '{word}'")
    return features


if __name__ == '__main__':
    
    import time
    guesses = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Words.js'
    answers = '/Users/be2297/Documents/wordle_bot/WordLists/NYT/Answers.js'
    guess_list = ['hello', 'chile', 'molly', 'asdas','hheei', 'polly', 'holly', 'rolly']
    # guess_list = load_words_from_js(answers, variable_name='official_answers')
    guess_list =load_words_from_js(guesses, variable_name='official_words')
    print(f'{len(guess_list)} words')

    start = time.time()
    feedback =  cache_all_feedback(guess_list, guess_list)
    end = time.time()
    # print(feedback)
    print(f'took {end-start} seconds to cache feedback for {len(guess_list)**2} pairs.')
    
    
