from enum import Enum


class DataOriginEnum(Enum):
    Training = "training"
    Validation = "validation"
    Testing = "performance_test"

def print_data_origin_enum() -> None:
    """
    Print the possible values of the DataOriginEnum.
    """
    print(f"Possible values of DataOriginEnum: {', '.join([origin.value for origin in DataOriginEnum])}")

def convert_str_arg_to_data_origin(arg: str) -> DataOriginEnum:
    """
    Convert a string argument to a DataOriginEnum.
    """
    if arg == DataOriginEnum.Training or arg == "training":
        return DataOriginEnum.Training
    elif arg == DataOriginEnum.Validation or arg == "validation":
        return DataOriginEnum.Validation
    elif arg == DataOriginEnum.Testing or arg == "testing":
        return DataOriginEnum.Testing
    else:
        print_data_origin_enum()
        raise ValueError(f"Unknown data origin: {arg}.")