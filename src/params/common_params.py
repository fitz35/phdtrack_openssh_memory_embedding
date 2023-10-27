# number of column to keep after feature selection
import os
import resource
from research_base.params.base_program_params import BaseProgramParams
from research_base.utils.enum_utils import convert_str_arg_to_enum_member
from research_base.utils.ml_utils.ml_evaluate import EVALUATE_RESULT_KEYS

from commons.data_loading.data_origin import DataOriginEnum, convert_str_arg_to_data_origin
from commons.feature_engineering.correlation_type import CorrelationType
from commons.results.commons_results_writer import CommonResultsWriter


from params.cli import CLIArguments
from params.pipelines import Pipeline
from embedding_quality.data_balancing.balancing_params import BalancingStrategies


# info column to drop
INFO_COLUMNS = ["file_path", "chn_addr"]

# column name for the user data of each chunk (for the chunk extraction pipeline)
USER_DATA_COLUMN = "hexa_representation"

BALANCING_STRATEGY = BalancingStrategies.UNDERSAMPLING

NB_COLUMNS_TO_KEEP = 8
FEATURE_CORRELATION_TYPE = CorrelationType.FE_CORR_PEARSON


class _DummyResultsWriter(CommonResultsWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """
    ADDITIONAL_HEADERS: list[str] = [
        # data balancing results
    ]

    def __init__(self, pipeline_name : str):
        super().__init__(
            'dummy.csv', 
            self.ADDITIONAL_HEADERS + EVALUATE_RESULT_KEYS, 
            pipeline_name
        )

    def set_result(self, field: str, value: str | None) -> None:
        pass

# Cannot use ABCMeta because of the Generic type
class CommonProgramParams(BaseProgramParams[Pipeline, _DummyResultsWriter]):
    
    ### cli args
    cli_args: CLIArguments
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # max memory usage (catch if not enough memory), in go
    MAX_MEMORY_USAGE : int


    # dev variable
    # max number of samples to use in the ml part, negative value means no limit
    MAX_NUMBERS_OF_SAMPLES_TO_USE: int
    # max number of samples to use after rebalancing, negative value means no limit (limit the memory usage)
    MAX_NUMBERS_OF_SAMPLES_TO_USE_AFTER_REBALANCING: int
    # Transformers batch size, the number of samples to use in one batch for the transformers model
    TRANSFORMERS_BATCH_SIZE : int

    # results
    FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH: str


    # common params

    data_origins_training: set[DataOriginEnum]
    data_origins_testing: set[DataOriginEnum]
    dataset_path : str

    def __init__(
            self, 
            app_name : str,
            dotenv_path: str,
            load_program_argv : bool = True, 
            construct_log : bool = True,
            debug : bool = False
    ):
        super().__init__(
            app_name, 
            Pipeline, 
            _DummyResultsWriter, 
            load_program_argv, 
            debug, 
            dotenv_path,
            construct_log
            )

        # prevent from memory killed
        max_bytes = self.MAX_MEMORY_USAGE * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

        # log the program params
        self.cli_args.log_raw_argv(self.RESULTS_LOGGER)


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

        if self.cli_args.args.output_folder is not None:
            self.OUTPUT_FOLDER = self.cli_args.args.output_folder
            assert isinstance(self.OUTPUT_FOLDER, str)
            assert os.path.isdir(self.OUTPUT_FOLDER), f"The folder '{self.OUTPUT_FOLDER}' does not exist!"
        else:
            print("ERROR: No output folder given.")
            exit(1)

        if self.cli_args.args.pipelines is not None:
            try:
                self.pipelines = convert_str_arg_to_enum_member(self.cli_args.args.pipelines, Pipeline)
                assert isinstance(self.pipelines, Pipeline)

            except ValueError:
                    print(f"ERROR: Invalid pipeline name: {self.cli_args.args.pipelines}")
                    exit(1)
        else:
            print("ERROR: No pipeline given.")
            exit(1)
        
        if self.cli_args.args.no_balancing is not None:
            self.no_balancing = self.cli_args.args.no_balancing
            assert isinstance(self.no_balancing, bool)


# --------------------------------------------

    def get_results_writer(self):
        """
        get the dummy results writer (compatibilities with the old code)
        """
        return _DummyResultsWriter("dummy")
