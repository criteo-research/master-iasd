import multiprocessing
from typing import Optional

import pandas as pd
import time
import random


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def convert_line_to_wv(line: str) -> str:
    splitted_line = line.split("\t")
    label = int(splitted_line[0]) if splitted_line[0] != "" else ""
    label = label if label == 1 else -1
    integer_features = splitted_line[1:14]
    categorical_features = splitted_line[14:]

    integer_features = " ".join([f"I{i}:{f}" for i, f in enumerate(integer_features) if f != ""])
    categorical_features = " ".join(categorical_features)

    vw_line = f"{label} |I {integer_features} |C {categorical_features}"

    return vw_line


@timeit
def convert_to_vw(input_filename: str, output_filename: str):
    with open(input_filename) as input_file, open(output_filename, "w") as output_file:
        for i, line in enumerate(input_file):
            output_file.write(convert_line_to_wv(line))
            if i % 1000 == 1:
                print(f"Converted {i} lines")


@timeit
def convert_to_vw_parallel(input_filename: str, output_filename: str):
    pool = multiprocessing.Pool(processes=12)
    with open(input_filename) as input_file, open(output_filename, "w") as output_file:
        for i, line in enumerate(pool.imap_unordered(convert_line_to_wv, input_file, chunksize=100)):
            output_file.write(line)
            if i % 1000 == 1:
                print(f"Converted {i} lines")


def split_train_test(
    filename: str,
    train_filename: Optional[str] = None,
    test_filename: Optional[str] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    random.seed(seed)
    if train_filename is None:
        train_filename = f"{filename}.train"
    if test_filename is None:
        test_filename = f"{filename}.test"
    with open(train_filename, "w") as train_file, open(test_filename, "w") as test_file, open(filename) as f:
        for i, line in enumerate(f):
            if random.uniform(0, 1) <= test_ratio:
                test_file.write(line)
            else:
                train_file.write(line)
            if i % 100000 == 0:
                print(f"Processed {i} lines")


if __name__ == "__main__":
    split_train_test("train.txt", train_filename="criteo_train.txt", test_filename="criteo_test.txt", test_ratio=0.2)
    convert_to_vw_parallel("criteo_train.txt", "criteo_train.vw")
    convert_to_vw_parallel("criteo_test.txt", "criteo_test.vw")
