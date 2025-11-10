from collections import defaultdict
import time


class WordleBot:
    def __init__(self, guess_list, answer_list):
        # Store lower-case for consistency with WordleGame
        self.guess_list = [w.lower() for w in guess_list]
        self.answer_list = [w.lower() for w in answer_list]
        self.possible_solutions = self.answer_list.copy()

    def reset(self):
        """Reset between games."""
        self.possible_solutions = self.answer_list.copy()

    def next_guess(self, guess_number, history):
        """Override in subclasses."""
        raise NotImplementedError


class NaiveBot(WordleBot):
    def next_guess(self, guess_number, history):
        # Totally ignores feedback; just a baseline.
        return self.answer_list[0]


class HeuristicWordleBot(WordleBot):
    """
    JS-style heuristic Wordle bot (single-board, standard Wordle) used as a benchmark.

    Implements:
      - Exact candidate filtering via recomputed feedback.
      - JS calculateAverageBucketSize heuristic:
            weighted = (1/N) * sum_b size_b^2
            adjusted = (1 - threes) * weighted
        with:
            threes tracking collisions (as in the JS).
      - reducesListBest / reducesListMost style:
            pick guesses with minimal adjusted, with early exits.
      - Feedback caching like JS pairings[].
      - Bounded recursive lookahead for top candidates
        (similar to calculateGuessList + countResults):
            * follow only the best next guess at each step
            * stop on depth/max_guesses or time budget.
    """

    CHECK_SIZE = 50          # how many top candidates to refine recursively
    MAX_TIME = 1.0           # per move time budget in seconds (benchmark-style)

    def __init__(self, guess_list, answer_list, max_guesses=6):
        super().__init__(guess_list, answer_list)
        self.max_guesses = max_guesses
        self.word_length = len(self.guess_list[0]) if self.guess_list else 5
        # Cache: (guess, answer) -> pattern "GYBBG"
        self._feedback_cache = {}

    # ---------- Feedback & patterns (local, JS-style) ----------

    def _compute_feedback(self, guess, secret):
        """Pure Wordle feedback, matching wordle.py semantics."""
        guess = guess.lower()
        secret = secret.lower()
        n = len(secret)

        feedback = ["gray"] * n
        freq = {}
        for ch in secret:
            freq[ch] = freq.get(ch, 0) + 1

        # Greens
        for i, ch in enumerate(guess):
            if ch == secret[i]:
                feedback[i] = "green"
                freq[ch] -= 1

        # Yellows
        for i, ch in enumerate(guess):
            if feedback[i] == "green":
                continue
            if freq.get(ch, 0) > 0:
                feedback[i] = "yellow"
                freq[ch] -= 1
            else:
                feedback[i] = "gray"

        return tuple(feedback)

    @staticmethod
    def _pattern_from_feedback(fb):
        mapping = {"green": "G", "yellow": "Y", "gray": "B"}
        return "".join(mapping[c] for c in fb)

    def _cached_pattern(self, guess, secret):
        key = (guess, secret)
        if key in self._feedback_cache:
            return self._feedback_cache[key]
        fb = self._compute_feedback(guess, secret)
        p = self._pattern_from_feedback(fb)
        self._feedback_cache[key] = p
        return p

    # ---------- Consistency filtering (exact, all history) ----------

    def _update_possible_solutions(self, history):
        """
        Keep only answers for which recomputed feedback for every past guess
        matches exactly what was observed.
        """
        if not history:
            self.possible_solutions = self.answer_list.copy()
            return

        mapping = {"green": "G", "yellow": "Y", "gray": "B"}
        candidates = []

        for cand in self.answer_list:
            ok = True
            for prev_guess, prev_fb in history:
                expected = "".join(mapping[c] for c in prev_fb)
                actual = self._cached_pattern(prev_guess.lower(), cand)
                if actual != expected:
                    ok = False
                    break
            if ok:
                candidates.append(cand)

        self.possible_solutions = candidates or self.answer_list.copy()

    # ---------- Heuristic: calculateAverageBucketSize ----------

    def _calculate_average_bucket_size(self, guess, answers):
        """
        Port of calculateAverageBucketSize(guess, answers, ...) for standard Wordle.

        Returns:
            {
              'word': guess,
              'weighted': weighted,
              'threes': threes,
              'adjusted': adjusted,
              'differences': {pattern: [answers_with_that_pattern_excl_all_green]}
            }
        """
        list_size = len(answers)
        if list_size == 0:
            return None

        differences = {}
        weighted = 0.0
        threes = 1.0
        all_green = "G" * self.word_length

        for secret in answers:
            diff = self._cached_pattern(guess, secret)

            if diff not in differences:
                differences[diff] = []

            # JS: don't put CORRECT-repeat patterns into buckets (they are "solved")
            if diff != all_green:
                differences[diff].append(secret)

            freq = len(differences[diff])
            if freq > 0:
                # Incremental form of (1/N) * Σ size_b^2
                weighted += (2.0 * freq - 1.0) / list_size
                if freq > 1:
                    # Track collisions (answers sharing a pattern)
                    threes -= 1.0 / list_size

        adjusted = (1.0 - threes) * weighted

        return {
            "word": guess,
            "weighted": weighted,
            "threes": threes,
            "adjusted": adjusted,
            "differences": differences,
        }

    def _reduces_list_best(self, answers, guesses, future_guess=True):
        """
        JS reducesListMost for our single-board Wordle case:

        - Score each guess with _calculate_average_bucket_size.
        - Track minimal adjusted and collect all guesses achieving it.
        - Early-exit like JS:
            * if weighted < 1 and future_guess
            * if min == 0 and we have >= answers.length candidates
        """
        best_words = []
        min_adj = float("inf")

        for guess in guesses:
            data = self._calculate_average_bucket_size(guess, answers)
            if not data:
                continue

            adj = data["adjusted"]

            if adj < min_adj - 1e-12:
                min_adj = adj
                best_words = [{
                    "word": guess,
                    "average": adj,
                    "differences": data["differences"],
                    "wrong": 0,
                }]
            elif abs(adj - min_adj) <= 1e-12:
                best_words.append({
                    "word": guess,
                    "average": adj,
                    "differences": data["differences"],
                    "wrong": 0,
                })

            if future_guess:
                if data["weighted"] < 1.0:
                    break
                if min_adj == 0 and len(best_words) >= len(answers):
                    break

        best_words.sort(key=lambda x: (x["average"], x["word"]))
        return best_words

    # ---------- Recursive lookahead (calculateGuessList / countResults style) ----------

    def _evaluate_root_guess_with_tree(self, root_guess, answers, all_guesses, guess_count):
        """
        Limited recursive evaluation for a single root guess:
        - Partition answers by root_guess feedback.
        - For each non-solved bucket, recursively:
            * choose next guess via _reduces_list_best
            * follow only that best guess.
        - Respect max_guesses and MAX_TIME.
        Returns (avg_total_guesses, wrong_count).
        """
        start = time.perf_counter()
        used = {root_guess}
        all_green = "G" * self.word_length

        # Partition answers for the root
        buckets = {}
        for secret in answers:
            p = self._cached_pattern(root_guess, secret)
            buckets.setdefault(p, []).append(secret)

        counts = defaultdict(int)  # extra depth (after root) -> how many
        wrong = 0

        for pattern, bucket in buckets.items():
            if pattern == all_green:
                counts[1] += len(bucket)  # solved by root_guess
            else:
                sub_counts, sub_wrong = self._count_results(
                    answers=bucket,
                    all_guesses=all_guesses,
                    used_guesses=used,
                    depth=1,
                    guess_count=guess_count,
                    start_time=start,
                )
                for d, c in sub_counts.items():
                    counts[1 + d] += c
                wrong += sub_wrong

            if time.perf_counter() - start > self.MAX_TIME:
                # Time limit: pessimistically mark remaining unseen answers as unsolved
                remaining = sum(
                    len(b)
                    for pat, b in buckets.items()
                    if pat not in (pattern, all_green)
                )
                wrong += remaining
                break

        avg = sum(depth * c for depth, c in counts.items())
        return avg, wrong

    def _count_results(self, answers, all_guesses, used_guesses, depth, guess_count, start_time):
        """
        Recursive subproblem:
        - If guesses remain:
            * Choose next guess via _reduces_list_best on this reduced set.
            * Partition and recurse only on that best guess.
        - Otherwise, remaining answers are failures.
        """
        if not answers:
            return defaultdict(int), 0

        if time.perf_counter() - start_time > self.MAX_TIME:
            return defaultdict(int), len(answers)

        if guess_count + depth >= self.max_guesses:
            return defaultdict(int), len(answers)

        if len(answers) == 1:
            # Will guess it next.
            counts = defaultdict(int)
            counts[1] = 1
            return counts, 0

        # Candidate guesses: all unused allowed guesses
        candidate_guesses = [g for g in all_guesses if g not in used_guesses]
        if not candidate_guesses:
            return defaultdict(int), len(answers)

        best_words = self._reduces_list_best(answers, candidate_guesses, future_guess=True)
        if not best_words:
            return defaultdict(int), len(answers)

        next_guess = best_words[0]["word"]
        all_green = "G" * self.word_length

        # Partition on next_guess
        buckets = {}
        for secret in answers:
            p = self._cached_pattern(next_guess, secret)
            buckets.setdefault(p, []).append(secret)

        new_used = set(used_guesses)
        new_used.add(next_guess)

        counts = defaultdict(int)
        wrong = 0

        for pattern, bucket in buckets.items():
            if pattern == all_green:
                counts[1] += len(bucket)
            else:
                sub_counts, sub_wrong = self._count_results(
                    answers=bucket,
                    all_guesses=all_guesses,
                    used_guesses=new_used,
                    depth=depth + 1,
                    guess_count=guess_count,
                    start_time=start_time,
                )
                for d, c in sub_counts.items():
                    counts[1 + d] += c
                wrong += sub_wrong

            if time.perf_counter() - start_time > self.MAX_TIME:
                remaining = sum(
                    len(b)
                    for pat, b in buckets.items()
                    if pat not in (pattern, all_green)
                )
                wrong += remaining
                break

        return counts, wrong

    # ---------- Public API ----------

    def reset(self):
        super().reset()
        # We keep the feedback cache across games: patterns are static.

    def next_guess(self, guess_number, history):
        # 1. Update candidates from the entire history
        self._update_possible_solutions(history)

        # 2. Trivial cases
        if len(self.possible_solutions) == 1:
            return self.possible_solutions[0]
        if not self.possible_solutions:
            return self.guess_list[0]

        used = {g.lower() for g, _ in history}

        # 3. Candidate guesses: all unused allowed guesses
        candidate_guesses = [g for g in self.guess_list if g.lower() not in used]
        if not candidate_guesses:
            for c in self.possible_solutions:
                if c not in used:
                    return c
            return self.guess_list[0]

        # 4. Stage 1: heuristic ranking for all candidate guesses
        best_words = self._reduces_list_best(
            self.possible_solutions,
            candidate_guesses,
            future_guess=True,
        )
        if not best_words:
            for c in self.possible_solutions:
                if c not in used:
                    return c
            return candidate_guesses[0]

        # 5. Stage 2: bounded recursive refinement for top CHECK_SIZE guesses
        all_guesses = candidate_guesses
        limit = min(self.CHECK_SIZE, len(best_words))

        for i in range(limit):
            bw = best_words[i]
            avg, wrong = self._evaluate_root_guess_with_tree(
                bw["word"],
                self.possible_solutions,
                all_guesses,
                guess_number,
            )
            bw["average"] = avg
            bw["wrong"] = wrong

        # 6. Final choice:
        #    - prefer guesses with wrong == 0
        #    - then lower average
        #    - then guesses that are valid remaining solutions
        #    - then lexicographically
        best_words[:limit] = sorted(
            best_words[:limit],
            key=lambda bw: (
                bw["wrong"] > 0,
                bw["average"],
                bw["word"] not in self.possible_solutions,
                bw["word"],
            ),
        )

        return best_words[0]["word"]
