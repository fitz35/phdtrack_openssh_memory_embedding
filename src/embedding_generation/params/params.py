from enum import Enum
import os

from research_base.utils.enum_utils import convert_str_arg_to_enum_member

from commons.params.common_params import CommonProgramParams
from embedding_generation.params.pipelines import Pipeline
from embedding_generation.results_writer.result_writer import ResultsWriter
from .cli import CLIArguments
from commons.data_loading.data_origin import convert_str_arg_to_data_origin

# column name for the user data of each chunk
USER_DATA_COLUMN = "hexa_representation"

# word2vec window size, in bytes. To have the number of words in a window, divide by the word size in bytes
WORD2VEC_WINDOW_BYTES_SIZE = 8

# word2vec min count
WORD2VEC_MIN_COUNT = 1

# word2vec vector size
WORD2VEC_VECTOR_SIZE = 100




class ProgramParams(CommonProgramParams[Pipeline, ResultsWriter]):
    """
    Wrapper class for program parameters.
    """

    ### cli args
    cli_args: CLIArguments
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI


    # length of the word in bytes for word2vec
    WORD_BYTE_SIZE: int


    # dev variable
    # max number of samples to use, negative value means no limit
    MAX_NUMBERS_OF_SAMPLES_TO_USE: int

    # results
    FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH: str


    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False
    ):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        super().__init__(
            "embedding_generation", 
            dotenv_path, 
            Pipeline, 
            ResultsWriter, 
            load_program_argv, 
            debug
            )

        # to be done last
        self._log_program_params()
    
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
                self.pipelines = set(map(lambda x : convert_str_arg_to_enum_member(x, Pipeline), self.cli_args.args.pipelines))
                assert isinstance(self.pipelines, set)

            except ValueError:
                    print(f"ERROR: Invalid pipeline name: {self.cli_args.args.pipelines}")
                    exit(1)
    
    
    def set_result_for(self, pipeline : Pipeline, column_name: str, value: str):
        """
        Set a result for a given pipeline.
        """
        self.results_manager.set_result_for(pipeline, column_name, value)
        

    def get_results_writer(self, pipeline : Pipeline) -> ResultsWriter:
        """
        Get the results writer for the current pipeline.
        """
        return self.results_manager.get_result_writer_for(pipeline)
