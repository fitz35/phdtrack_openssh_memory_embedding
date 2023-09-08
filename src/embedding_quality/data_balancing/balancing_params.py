from enum import Enum

from commons.utils.enum_utils import print_enum_values, convert_str_arg_to_enum_member

class BalancingStrategies(Enum):
    """
    Data balancing strategies
    """
    NO_BALANCING = "no_balancing"
    UNDERSAMPLING = "undersampling"
    SMOTE = "smote"
    ADASYN = "adasyn"
    OVERSAMPLING = "oversampling"

def print_balancing_strategies_names() -> None:
    """
    Print the possible values of the BalancingStrategies.
    """
    print_enum_values(BalancingStrategies)


def convert_str_arg_to_balancing_strategy(arg: str):
    """
    Convert a string argument to a BalancingStrategies.
    """
    return convert_str_arg_to_enum_member(arg, BalancingStrategies)
