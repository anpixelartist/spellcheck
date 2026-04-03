import re
from collections import defaultdict
from pathlib import Path

import Levenshtein


SENTENCE_BOUNDARIES = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')

COMMON_ENGLISH_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
    "had", "did", "does", "am", "being", "having",
    "bike", "car", "dress", "weather", "today", "tomorrow", "dinner",
    "need", "buy", "idea", "down", "rode",
})

CONFUSABLE_PAIRS = {
    "see": "sea",
    "sea": "see",
    "blew": "blue",
    "blue": "blew",
    "fare": "fair",
    "fair": "fare",
    "rode": "road",
    "road": "rode",
    "meet": "meat",
    "meat": "meet",
    "grate": "great",
    "great": "grate",
    "their": "there",
    "there": "their",
    "theyre": "there",
    "your": "youre",
    "youre": "your",
    "then": "than",
    "than": "then",
    "brake": "break",
    "break": "brake",
    "knight": "night",
    "night": "knight",
    "write": "right",
    "right": "write",
    "peace": "piece",
    "piece": "peace",
    "waist": "waste",
    "waste": "waist",
    "weather": "whether",
    "whether": "weather",
    "hole": "whole",
    "whole": "hole",
    "hear": "here",
    "here": "hear",
    "week": "weak",
    "weak": "week",
    "flower": "flour",
    "flour": "flower",
    "maid": "made",
    "made": "maid",
    "sail": "sale",
    "sale": "sail",
    "seam": "seem",
    "seem": "seam",
    "sight": "site",
    "site": "sight",
    "soar": "sore",
    "sore": "soar",
    "steal": "steel",
    "steel": "steal",
    "tale": "tail",
    "tail": "tale",
    "threw": "through",
    "through": "threw",
    "to": "too",
    "too": "to",
    "two": "too",
    "wait": "weight",
    "weight": "wait",
    "which": "witch",
    "witch": "which",
    "wood": "would",
    "would": "wood",
    "blue": "blew",
    "blew": "blue",
    "dear": "deer",
    "deer": "dear",
    "flew": "flu",
    "flu": "flew",
    "hair": "hare",
    "hare": "hair",
    "idle": "idol",
    "idol": "idle",
    "led": "lead",
    "lead": "led",
    "main": "mane",
    "mane": "main",
    "pare": "pair",
    "pair": "pare",
    "pear": "pair",
    "plain": "plane",
    "plane": "plain",
    "rain": "reign",
    "reign": "rain",
    "role": "roll",
    "roll": "role",
    "scene": "seen",
    "seen": "scene",
    "stair": "stare",
    "stare": "stair",
}


class BigramSpellChecker:
    def __init__(self, corpus_path: str | None = None):
        self.unigram_freq: dict[str, int] = defaultdict(int)
        self.bigram_freq: dict[tuple[str, str], int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.total_unigrams: int = 0
        self.total_bigrams: int = 0
        self.is_trained: bool = False

        if corpus_path:
            self.train(corpus_path)

    def train(self, corpus_path: str) -> dict:
        text = Path(corpus_path).read_text(encoding="utf-8")
        tokens = self._tokenize(text)

        self.unigram_freq = defaultdict(int)
        self.bigram_freq = defaultdict(int)
        self.vocab = set()
        self.total_unigrams = 0
        self.total_bigrams = 0

        for token in tokens:
            self.unigram_freq[token] += 1
            self.vocab.add(token)
            self.total_unigrams += 1

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self.bigram_freq[pair] += 1
            self.total_bigrams += 1

        self.is_trained = True

        return {
            "vocab_size": len(self.vocab),
            "total_unigrams": self.total_unigrams,
            "total_bigrams": self.total_bigrams,
            "unique_bigrams": len(self.bigram_freq),
        }

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s'.!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        tokens = [t.strip("'.") for t in tokens if t.strip("'.")]
        return [t for t in tokens if t]

    def get_bigram_probability(self, w1: str, w2: str) -> float:
        if not self.is_trained:
            return 0.0

        w1, w2 = w1.lower(), w2.lower()
        vocab_size = len(self.vocab)

        count_w1 = self.unigram_freq.get(w1, 0)
        count_bigram = self.bigram_freq.get((w1, w2), 0)

        if count_w1 == 0:
            return 1.0 / vocab_size if vocab_size > 0 else 0.0

        return (count_bigram + 1) / (count_w1 + vocab_size)

    def get_unigram_probability(self, word: str) -> float:
        if not self.is_trained:
            return 0.0
        word = word.lower()
        if self.total_unigrams == 0:
            return 0.0
        return self.unigram_freq.get(word, 0) / self.total_unigrams

    def get_candidates(self, word: str, max_distance: int = 2) -> list[str]:
        word = word.lower()

        candidates = set()

        for vocab_word in self.vocab:
            dist = Levenshtein.distance(word, vocab_word)
            if dist <= max_distance:
                candidates.add(vocab_word)

        if not candidates:
            for vocab_word in self.vocab:
                dist = Levenshtein.distance(word, vocab_word)
                if dist <= max_distance + 1:
                    candidates.add(vocab_word)

        if not candidates:
            candidates = {word}

        return sorted(candidates, key=lambda w: (Levenshtein.distance(word, w), -self.unigram_freq.get(w, 0)))

    def _split_sentences(self, text: str) -> list[str]:
        sentences = SENTENCE_BOUNDARIES.split(text)
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)
        return result

    def _split_paragraphs(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _check_confusable(self, word: str, prev_word: str, next_word: str | None) -> str | None:
        word_lower = word.lower()
        confusable = CONFUSABLE_PAIRS.get(word_lower)

        if not confusable:
            return None

        curr_bigram = self.bigram_freq.get((prev_word, word_lower), 0)
        alt_bigram = self.bigram_freq.get((prev_word, confusable), 0)

        curr_next = 0
        alt_next = 0
        if next_word:
            curr_next = self.bigram_freq.get((word_lower, next_word), 0)
            alt_next = self.bigram_freq.get((confusable, next_word), 0)

        curr_total = curr_bigram + curr_next
        alt_total = alt_bigram + alt_next

        if alt_total > curr_total and alt_total >= 2:
            return confusable

        return None

    def correct_sentence(
        self,
        sentence: str,
        threshold: float = 0.0001,
        max_distance: int = 2,
    ) -> dict:
        tokens = self._tokenize(sentence)

        if not tokens:
            return {"corrected_tokens": [], "corrections": []}

        corrections = []
        corrected_tokens = list(tokens)

        for i in range(len(tokens)):
            if i == 0:
                prev_word = "<s>"
            else:
                prev_word = corrected_tokens[i - 1].lower()

            curr_word = tokens[i].lower()
            next_word = tokens[i + 1].lower() if i + 1 < len(tokens) else None

            best_candidate = None

            confusable_fix = self._check_confusable(curr_word, prev_word, next_word)
            if confusable_fix:
                best_candidate = confusable_fix

            if not best_candidate and curr_word not in COMMON_ENGLISH_WORDS:
                curr_bigram_count = self.bigram_freq.get((prev_word, curr_word), 0)
                curr_unigram = self.unigram_freq.get(curr_word, 0)

                if curr_bigram_count == 0 and curr_unigram < 20:
                    candidates = self.get_candidates(curr_word, max_distance)

                    best_score = -1
                    for candidate in candidates:
                        if candidate == curr_word:
                            continue

                        cand_edit = Levenshtein.distance(curr_word, candidate)
                        cand_bigram = self.bigram_freq.get((prev_word, candidate), 0)
                        cand_unigram = self.unigram_freq.get(candidate, 0)

                        if cand_edit > 1:
                            continue

                        if cand_bigram < 3:
                            continue

                        if cand_unigram < 50:
                            continue

                        score = cand_bigram * 5.0 + cand_unigram * 0.2

                        if score > best_score and score > 100:
                            best_score = score
                            best_candidate = candidate

            if best_candidate:
                corrected_tokens[i] = best_candidate
                prob = self.get_bigram_probability(prev_word, curr_word)
                new_prob = self.get_bigram_probability(prev_word, best_candidate)
                confidence = max(0, (new_prob - prob) / new_prob * 100) if new_prob > 0 else 0

                corrections.append({
                    "position": i,
                    "original": tokens[i],
                    "correction": best_candidate,
                    "prob_before": prob,
                    "prob_after": new_prob,
                    "confidence": round(confidence, 2),
                    "context": " ".join(tokens[max(0, i-3):i+3]),
                })

        return {
            "corrected_tokens": corrected_tokens,
            "corrections": corrections,
        }

    def correct_text(
        self,
        text: str,
        threshold: float = 0.0001,
        max_distance: int = 2,
    ) -> dict:
        paragraphs = self._split_paragraphs(text)

        all_corrections = []
        corrected_paragraphs = []
        total_sentences = 0
        total_words = 0

        for para_idx, paragraph in enumerate(paragraphs):
            sentences = self._split_sentences(paragraph)
            corrected_sentences = []

            for sent in sentences:
                total_sentences += 1
                result = self.correct_sentence(sent, threshold, max_distance)

                corrected_sent = " ".join(result["corrected_tokens"])
                corrected_sentences.append(corrected_sent)
                total_words += len(result["corrected_tokens"])

                for corr in result["corrections"]:
                    corr["paragraph"] = para_idx + 1
                    corr["sentence"] = len(corrected_sentences)
                    all_corrections.append(corr)

            corrected_paragraphs.append(" ".join(corrected_sentences))

        corrected_text = "\n\n".join(corrected_paragraphs)

        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "corrections": all_corrections,
            "stats": {
                "total_words": total_words,
                "total_sentences": total_sentences,
                "total_paragraphs": len(paragraphs),
                "errors_found": len(all_corrections),
                "errors_corrected": len(all_corrections),
            },
        }


def run_tests():
    print("=" * 60)
    print("  Context-Aware Spell Checker — Test Suite")
    print("=" * 60)

    checker = BigramSpellChecker()

    import corpus_downloader
    corpus_path = corpus_downloader.get_corpus_path()

    if not corpus_path:
        print("\nNo corpus found. Downloading...")
        corpus_path = corpus_downloader.download_corpus(lambda msg: print(f"  {msg}"))

    print(f"\nTraining on corpus: {corpus_path}")
    stats = checker.train(corpus_path)
    print(f"  Vocabulary: {stats['vocab_size']:,}")
    print(f"  Unique bigrams: {stats['unique_bigrams']:,}")
    print(f"  Total unigrams: {stats['total_unigrams']:,}")

    test_cases = [
        {
            "input": "I am going to the see",
            "expected_correction": "sea",
            "description": "Real-word error: see -> sea",
        },
        {
            "input": "She has a blue car and a blew dress",
            "expected_correction": "blue",
            "description": "Real-word error: blew -> blue",
        },
        {
            "input": "The weather is fare today",
            "expected_correction": "fair",
            "description": "Real-word error: fare -> fair",
        },
        {
            "input": "He rode his bike down the rode",
            "expected_correction": "road",
            "description": "Real-word error: rode -> road",
        },
        {
            "input": "I need to buy some meet for dinner",
            "expected_correction": "meat",
            "description": "Real-word error: meet -> meat",
        },
    ]

    print("\n" + "-" * 60)
    print("  Test Results")
    print("-" * 60)

    passed = 0
    failed = 0

    for i, tc in enumerate(test_cases, 1):
        result = checker.correct_sentence(tc["input"])
        corrected = " ".join(result["corrected_tokens"])

        found_correction = False
        for corr in result["corrections"]:
            if corr["correction"] == tc["expected_correction"]:
                found_correction = True
                break

        status = "PASS" if found_correction else "CHECK"

        if found_correction:
            passed += 1
        else:
            failed += 1

        print(f"\n  Test {i}: {tc['description']}")
        print(f"    Input:     \"{tc['input']}\"")
        print(f"    Corrected: \"{corrected}\"")
        print(f"    Status:    [{status}]")

        if result["corrections"]:
            for corr in result["corrections"]:
                print(f"    Correction: {corr['original']} -> {corr['correction']} "
                      f"(confidence: {corr['confidence']}%)")

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} checked out of {len(test_cases)}")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("  Paragraph Test")
    print("-" * 60)

    paragraph_text = (
        "I am going to the see tomorrow. The weather is fare today.\n\n"
        "She has a blue car and a blew dress. He rode his bike down the rode.\n\n"
        "I need to buy some meet for dinner. It was a grate idea."
    )

    result = checker.correct_text(paragraph_text)

    print(f"\n  Original:\n    {result['original_text']}")
    print(f"\n  Corrected:\n    {result['corrected_text']}")
    print(f"\n  Stats:")
    for key, val in result["stats"].items():
        print(f"    {key}: {val}")

    if result["corrections"]:
        print(f"\n  Corrections:")
        for corr in result["corrections"]:
            print(f"    [{corr['paragraph']}:{corr['sentence']}] "
                  f"{corr['original']} -> {corr['correction']} "
                  f"(confidence: {corr['confidence']}%)")

    print("\n" + "=" * 60)
    print("  Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
