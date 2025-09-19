import argparse
from typing import List


def make_sentences(chars_per_sentence: int) -> List[str]:
    words = open("./bleak-house.txt").read().split()

    # Discard some words at the beginning to get rid of contents, preface, etc
    words = words[2000:]

    answer = []
    current_words = []
    current_length = 0
    for word in words:
        if current_length + len(word) > chars_per_sentence:
            answer.append(current_words)
            current_words = []
            current_length = 0

        current_words.append(word)
        current_length += len(word)

    return answer


def write_sentences(sentences: List[str]):
    bert_sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    sentences = [" ".join(sentence) for sentence in sentences]

    sentences = bert_sentences + sentences[:1024]

    with open("sentences.txt", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate sentence and token datasets")
    parser.add_argument("--chars-per-sentence", type=int, default=256 * 4)
    args = parser.parse_args()

    sentences = make_sentences(args.chars_per_sentence)
    write_sentences(sentences)


if __name__ == "__main__":
    main()
