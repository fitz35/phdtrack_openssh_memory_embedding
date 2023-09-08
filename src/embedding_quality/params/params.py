from commons.params.base_program_params import BaseProgramParams
from embedding_quality.results.result_writer import ResultsWriter
from embedding_quality.feature_engineering.correlation_type import CorrelationType
from embedding_quality.data_balancing.balancing_params import BalancingStrategies
from .cli import CLIArguments
from .data_origin import DataOriginEnum, convert_str_arg_to_data_origin

# number of column to keep after feature selection
NB_COLUMNS_TO_KEEP = 8
FEATURE_CORRELATION_TYPE = CorrelationType.FE_CORR_PEARSON
BALANCING_STRATEGY = BalancingStrategies.SMOTE
# info column to drop
INFO_COLUMNS = ["file_path", "f_dtns_addr"]

class ProgramParams(BaseProgramParams):
    """
    Wrapper class for program parameters.
    """
    results_writer: ResultsWriter

    ### cli args
    cli_args: CLIArguments
    data_origins_training: set[DataOriginEnum]
    data_origins_testing: set[DataOriginEnum]
    dataset_path : str
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # results
    CSV_EMBEDDING_QUALITY_RESULTS_PATH: str
    FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH: str


    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        super().__init__(load_program_argv, debug)

        # keep results
        self.__results_manager_init()

        # to be done last
        self._log_program_params()
    
    def __results_manager_init(self):
        """
        Initialize results manager, and start keeping results-related information.
        """
        self.results_writer = ResultsWriter(self.CSV_EMBEDDING_QUALITY_RESULTS_PATH)


        self.set_result_for(
            "random_seed",
            str(self.RANDOM_SEED)
        )

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
    

    # result wrappers
    def save_results_to_csv(self):
        """
        Save results to CSV files.
        """
        self.results_writer.save_results()
    
    def set_result_for(self, column_name: str, value: str):
        """
        Set a result for a given pipeline.
        """
        self.results_writer.set_result(column_name, value)
