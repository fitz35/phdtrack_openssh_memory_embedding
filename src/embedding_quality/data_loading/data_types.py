from pandas import DataFrame, Series




from dataclasses import dataclass


@dataclass
class SamplesAndLabels:
    sample : DataFrame
    labels : Series