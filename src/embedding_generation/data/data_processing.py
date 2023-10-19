
from typing import List


def split_into_chunks(hex_string: str, chunk_size: int = 2) -> List[str]:
    """Splits a string into chunks of given size. If the last chunk isn't full, it's filled with zeros."""
    
    # Append zeros to ensure the length is a multiple of chunk_size
    remainder = len(hex_string) % chunk_size
    if remainder:
        hex_string += '0' * (chunk_size - remainder)

    return [hex_string[i:i+chunk_size] for i in range(0, len(hex_string), chunk_size)]
