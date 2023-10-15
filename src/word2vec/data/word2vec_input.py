


import pandas as pd
from typing import List

from word2vec.params.params import USER_DATA_COLUMN, ProgramParams


def split_into_chunks(hex_string: str, chunk_size: int = 2) -> List[str]:
    """Splits a string into chunks of given size."""
    return [hex_string[i:i+chunk_size] for i in range(0, len(hex_string), chunk_size)]

def transform_hex_data(params : ProgramParams, df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the specified column of a DataFrame into lists of 2-byte length strings."""
    df[USER_DATA_COLUMN] = df[USER_DATA_COLUMN].apply(lambda x: split_into_chunks(x, params.WORD_BYTE_SIZE))
    return df