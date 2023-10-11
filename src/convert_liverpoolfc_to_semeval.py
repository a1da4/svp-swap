import argparse
import logging
import nltk
import re

from typing import List, Dict


def preprocess_body(body: str) -> List[str]:
    # delete \\ and \\n
    cleaned_body = re.sub(r"\\", "", re.sub(r"\\n", "", body))
    sentences: List[str] = nltk.tokenize.sent_tokenize(cleaned_body.lower())
    tokenized_sentences = []
    for sentence in sentences:
        words: List[str] = nltk.tokenize.word_tokenize(sentence)
        tokenized_sentence: str = " ".join(words)
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences


def convert_liverpoolfc_to_semeval(word_path: str, file_pathes: List[str]) -> None:
    """ convert 
     - annotated_words.csv (LiverpoolFC) -> targets.txt, graded.txt (SemEval)
     - LiverpoolFC_1*.txt (json-like style) -> (txt)

    :param word_path: path to annotated target file (annotated_words.csv)
    :param file_pathes: pathes of target corpora
    """
    logging.info(f"[convert_liverpoolfc_to_semeval] {word_path}")
    targets: List[str] = []
    target2gold: Dict[float] = {}
    with open(word_path) as fp:
        for line in fp:
            items = line.strip().split(",")
            word: str = items[1]
            try:
                shift_index: float = float(items[2])
            except:
                continue
            targets.append(word)
            target2gold[word] = shift_index

    with open(f"{word_path}_targets.txt", "w") as fp:
        for target in targets:
            fp.write(f"{target}\n")

    with open(f"{word_path}_graded.txt", "w") as fp:
        for target in targets:
            fp.write(f"{target}\t{target2gold[target]}\n")

    for file_path in file_pathes:
        logging.info(f"[convert_liverpoolfc_to_semeval] file_path: {file_path}")
        with open(f"{file_path}_body.txt", "w") as wp:
            with open(file_path) as fp:
                for line in fp:
                    # obtain main sentence, {, ..., 'body': "MAIN SENTENCE", 'parent_id': }
                    if "'body': " in line:
                        sequence_from_body: str = line.strip().split("'body': ")[1]
                        sequence_body: str = sequence_from_body.split(", 'parent_id'")[0]
                        # remove start and end of ' ' / " "
                        sequence_body = sequence_body[1:-1]
                        tokenized_sentences = preprocess_body(sequence_body) 
                        for tokenized_sentence in tokenized_sentences:
                            wp.write(f"{tokenized_sentence}\n")


def cli_main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_path", help="path to target file (LiverpoolFC, annotated words)")
    parser.add_argument("--file_pathes", nargs=2, help="path to target file (LiverpoolFC, corpora)")

    args = parser.parse_args()
    convert_liverpoolfc_to_semeval(args.word_path, args.file_pathes)

if __name__ == "__main__":
    cli_main()
