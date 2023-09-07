from enum import Enum

def print_enum_values(enum_class: Enum) -> None:
    """
    Print the possible values of the PipelineNames.
    """
    print(f"Possible values of {enum_class.__name__}: {', '.join([member.value for member in enum_class])}")   


def convert_str_arg_to_enum_member(arg: str, enum_class: Enum) -> Enum:
    """
    Convert a string argument to a member of the given Enum class.
    """
    for member in enum_class:
        if arg == member.value or arg.lower() == member.name.lower():
            print(f"Found {enum_class.__name__}: {member}")
            return member

    # If the code reaches here, the argument did not match any Enum member
    print_enum_values(enum_class)
    raise ValueError(f"Unknown {enum_class.__name__}: {arg}.")
