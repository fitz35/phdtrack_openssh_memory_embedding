
from enum import Enum


class CorrelationType(Enum):
    FE_CORR_PEARSON = 1
    FE_CORR_KENDALL = 2
    FE_CORR_SPEARMAN = 3