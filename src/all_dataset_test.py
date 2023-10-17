
# This script will run the test specify in the argument on all the dataset

from enum import Enum
import os
import argparse

OPENSSH_EMBEDDING_PROJECT_PATH = "/home/clement/Documents/github/phdtrack_openssh_memory_embedding"

TRANSFORMERS_PARENT_FOLDER = os.path.join(OPENSSH_EMBEDDING_PROJECT_PATH, "results/embedding_generation/transformers")
WORD2VEC_PARENT_FOLDER = os.path.join(OPENSSH_EMBEDDING_PROJECT_PATH, "results/embedding_generation/word2vec")






class TestEnum(Enum):
    Coherence = "coherence"
    Quality = "quality"

def convert_arg_to_test(arg: str) -> TestEnum:
    """
    Convert a string argument to a TestEnum.
    """
    if arg == TestEnum.Coherence or str(arg).lower() == "coherence":
        return TestEnum.Coherence
    elif arg == TestEnum.Quality or str(arg).lower() == "quality":
        return TestEnum.Quality
    else:
        raise ValueError(f"Unknown test: {arg}.")

# get the test to run
parser = argparse.ArgumentParser()
parser.add_argument("test", type=convert_arg_to_test, choices=list(TestEnum))
args = parser.parse_args()

def list_folders(directory_path : str) -> list[str]:
    if not os.path.isdir(directory_path):
        return []
    return [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

all_paths = []

# transformers
# list all folder inside TRANSFORMERS_PARENT_FOLDER
TRANSFORMERS_PATHS = list_folders(TRANSFORMERS_PARENT_FOLDER)
all_paths += TRANSFORMERS_PATHS


# word2vec
all_paths += [WORD2VEC_PARENT_FOLDER]



test_to_do : TestEnum = args.test

# run the test on all the paths
for path in all_paths:
    print(f"running test {args.test} on {path}")
    
    match test_to_do :
        case TestEnum.Coherence:
            os.system(f"python3 {OPENSSH_EMBEDDING_PROJECT_PATH}/src/embedding_coherence_main.py -d {path} -otr training -ots validation")
        case TestEnum.Quality:
            os.system(f"python3 {OPENSSH_EMBEDDING_PROJECT_PATH}/src/embedding_quality_main.py -d {path} -otr training -ots validation")
        
        case _:
            print("unknown test")
            exit(1)




