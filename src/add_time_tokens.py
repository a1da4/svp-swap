import argparse
import logging
from typing import List

def add_time_tokens(file_pathes: List[str]) -> None:
    """ add time tokens <1> / <2>

    :param file_pathes: list of target pathes

    """
    for time_id, file_path in enumerate(file_pathes):
        logging.info(f"[add_time_token] file_path: {file_path}")
        time_token = f"<{time_id + 1}>"
        logging.debug(f"[add_time_token] time_token: {time_token}")

        added_file_path = f"{file_path[:-4]}_added_time_token.txt"
        logging.info(f"[add_time_token] added_file_path: {added_file_path}")
        with open(added_file_path, "w") as wp:
            with open(file_path) as fp:
                for line in fp:
                    sentence = line.strip()
                    logging.debug(f"[add_time_token] sentence: {sentence}")
                    added_sentence = f"{time_token} {sentence}"
                    logging.debug(f"[add_time_token] added sentence: {added_sentence}")
                    wp.write(f"{added_sentence}\n")


def cli_main():
    logging.basicConfig(filename="../results/add_time_tokens_debug.log", format="%(asctime)s %(message)s", level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_pathes", nargs="*")
    args = parser.parse_args()
    add_time_tokens(args.file_pathes)


if __name__ == "__main__":
    cli_main()
