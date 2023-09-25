from enum import Enum
import os
from research_base.params.base_program_params import BaseProgramParams
from embedding_quality.results.result_writer import ResultsWriter
from commons.feature_engineering.correlation_type import CorrelationType
from embedding_quality.data_balancing.balancing_params import BalancingStrategies
from .cli import CLIArguments
from commons.data_loading.data_origin import DataOriginEnum, convert_str_arg_to_data_origin


BALANCING_STRATEGY = BalancingStrategies.UNDERSAMPLING
# info column to drop
INFO_COLUMNS = ["file_path", "f_dtns_addr"]

class Pipeline(Enum):
    PIPELINE="embedding_quality"

class ProgramParams(BaseProgramParams[Pipeline, ResultsWriter]):
    """
    Wrapper class for program parameters.
    """

    ### cli args
    cli_args: CLIArguments
    data_origins_training: set[DataOriginEnum]
    data_origins_testing: set[DataOriginEnum]
    dataset_path : str
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # results
    FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH: str


    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        super().__init__("embedding_quality", Pipeline, ResultsWriter, load_program_argv, debug, dotenv_path=dotenv_path)

        # keep results
        self.__results_manager_init()

        # to be done last
        self._log_program_params()
    
    def __results_manager_init(self):
        """
        Initialize results manager, and start keeping results-related information.
        """

        self.set_result_for(
            "dataset_path",
            self.dataset_path
        )

        self.set_result_for(
            "training_dataset_origin",
            " ".join([origin.value for origin in self.data_origins_training])
        )
        if self.data_origins_testing is not None:
            # NOTE: when DATA_ORIGINS_TESTING to none, we can split the data in the pipeline if needed.
            self.set_result_for(
                "testing_dataset_origin", 
                " ".join([origin.value for origin in self.data_origins_testing])
            )
    
    
    def _load_program_argv(self) -> None:
        """
        Load given program arguments.
        """
        self.cli_args: CLIArguments = CLIArguments()
    
    def _consume_program_argv(self) -> None:
        """
        Consume given program arguments.
        """
        if self.cli_args.args.debug is not None:
            self.DEBUG = self.cli_args.args.debug
            assert isinstance(self.DEBUG, bool)

        if self.cli_args.args.max_ml_workers is not None:
            self.MAX_ML_WORKERS = int(self.cli_args.args.max_ml_workers)
            assert isinstance(self.MAX_ML_WORKERS, int)

        if self.cli_args.args.origins_training is not None:
            try:
                self.data_origins_training = set(map(convert_str_arg_to_data_origin, self.cli_args.args.origins_training))
                assert isinstance(self.data_origins_training, set)
            except ValueError:
                print(f"ERROR: Invalid data origin training: {self.cli_args.args.origins_training}")
                exit(1)
        else:
            print("ERROR: No training data origin given.")
            exit(1)
        
        if self.cli_args.args.origins_testing is not None:
            try:
                self.data_origins_testing = set(map(convert_str_arg_to_data_origin, self.cli_args.args.origins_testing))
                assert isinstance(self.data_origins_testing, set)
            except ValueError:
                print(f"ERROR: Invalid data origin testing: {self.cli_args.args.origins_testing}")
                exit(1)
            # NOTE: when DATA_ORIGINS_TESTING to none, we can split the data in the pipeline if needed.
        
        if self.cli_args.args.dataset_path is not None:
            self.dataset_path = self.cli_args.args.dataset_path
            assert isinstance(self.dataset_path, str)
        else:
            print("ERROR: No dataset path given.")
            exit(1)
    
    
    def set_result_for(self, column_name: str, value: str):
        """
        Set a result for a given pipeline.
        """
        super().set_result_for(Pipeline.PIPELINE, column_name, value)

    def get_results_writer(self) -> ResultsWriter:
        """
        Get the results writer for the current pipeline.
        """
        return self.results_manager.get_result_writer_for(Pipeline.PIPELINE)
