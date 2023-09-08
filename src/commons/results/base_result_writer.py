import csv
from datetime import datetime
import os
import platform
import psutil
from typing import Optional

from commons.utils.utils import datetime2str


class BaseResultWriter(object):
    """
    This class is used to write the results to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to the results.
    """
    csv_file_path: str
    headers: list[str] # WARN, don't add the list of headers here, it will be shared between all instances of this class
    results: dict[str, Optional[str]]
    __already_written_results: bool # Flag, only write results once

    BASE_HEADERS = [
        "start_time", 
        "end_time", 
        "duration", 
        "random_seed", # keep it everywhere, useful for rebalancing
    ]

    def __init__(
            self, 
            csv_file_path: str, 
            more_header: list[str],
        ):
        self.csv_file_path = csv_file_path
        self.headers = self.BASE_HEADERS + more_header
        self.results = {header: None for header in self.headers + more_header}
        self.__save_system_info()

        self.__already_written_results = False

        # start time
        self.set_result("start_time", datetime2str(datetime.now()))

    def set_result(self, field: str, value: Optional[str]) -> None:
        if field in self.results:
            self.results[field] = value
        else:
            raise ValueError(f"ClassificationResultsWriter: Invalid field name: {field}")
        
    def save_results(self) -> None:
        """
        Write the results to the CSV file.
        WARNING: This function MUST only be called once.
        """
        if self.__already_written_results:
            raise ValueError("ClassificationResultsWriter: Results already written")

        file_exists = os.path.isfile(self.csv_file_path)
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            # write header only if file is empty or does not exist
            if not file_exists or os.stat(self.csv_file_path).st_size == 0:
                writer.writeheader()
            writer.writerow(self.results)
        
        self.__already_written_results = True
    
    def print_results(self) -> None:
        for key, value in self.results.items():
            print(f"{key}: {value}")
    
    
    # utility functions
    def __add_header_column_with_value(self, header: str, value: str) -> None:
        self.__check_no_duplicate_header(header)
        self.headers.append(header)
        self.results[header] = value
    
    def __check_no_duplicate_header(self, header: str) -> None:
        if header in self.headers:
            print (self.headers)
            raise ValueError(f"Header '{header}' is already in the header list.")


    def __save_system_info(self):
        """
        Save machine information to the result saver.
        """
        # Get system information
        uname = platform.uname()

        self.__add_header_column_with_value("system", uname.system)
        self.__add_header_column_with_value("node_name", uname.node)
        self.__add_header_column_with_value("release", uname.release)
        self.__add_header_column_with_value("version", uname.version)
        self.__add_header_column_with_value("machine", uname.machine)
        self.__add_header_column_with_value("processor", uname.processor)

        # Get CPU information
        self.__add_header_column_with_value("physical_cores", str(psutil.cpu_count(logical=False)))
        self.__add_header_column_with_value("total_cores", str(psutil.cpu_count(logical=True)))

        # Get memory information
        mem_info = psutil.virtual_memory()
        self.__add_header_column_with_value("total_memory", str(mem_info.total))
        self.__add_header_column_with_value("available_memory", str(mem_info.available))
