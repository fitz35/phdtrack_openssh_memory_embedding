


import pandas as pd
from typing import List

from embedding_generation.params.params import USER_DATA_COLUMN, ProgramParams


def split_into_chunks(hex_string: str, chunk_size: int = 2) -> List[str]:
    """Splits a string into chunks of given size."""
    return [hex_string[i:i+chunk_size] for i in range(0, len(hex_string), chunk_size)]