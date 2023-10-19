# number of column to keep after feature selection
from abc import ABC
from enum import Enum
from typing import Generic, Type, TypeVar


from research_base.results.base_result_writer import BaseResultWriter
from research_base.params.base_program_params import BaseProgramParams

from commons.data_loading.data_origin import DataOriginEnum
from commons.feature_engineering.correlation_type import CorrelationType

# column name for the user data of each chunk (for the chunk extraction pipeline)
USER_DATA_COLUMN = "hexa_representation"

NB_COLUMNS_TO_KEEP = 8
FEATURE_CORRELATION_TYPE = CorrelationType.FE_CORR_PEARSON

ResultWriter = TypeVar('ResultWriter', bound=BaseResultWriter)  # CustomResultWriter should be a subtype of BaseResultWriter
PipelineNamesEnum = TypeVar('PipelineNamesEnum', bound=Enum)  # PipelineNamesEnum should be a subtype of Enum

# Cannot use ABCMeta because of the Generic type
class CommonProgramParams(Generic[PipelineNamesEnum, ResultWriter], BaseProgramParams[PipelineNamesEnum, ResultWriter]):

    data_origins_training: set[DataOriginEnum]
    data_origins_testing: set[DataOriginEnum]
    dataset_path : str

    def __init__(
            self, 
            app_name : str,
            dotenv_path: str,
            pipeline : Type[PipelineNamesEnum],
            resultsWriter : Type[ResultWriter],
            load_program_argv : bool = True, 
            debug : bool = False
    ):
        super().__init__(
            app_name, 
            pipeline, 
            resultsWriter, 
            load_program_argv, 
            debug, 
            dotenv_path
            )

        # keep results
        self.__results_manager_init()

    def __results_manager_init(self):
        """
        Initialize results manager, and start keeping results-related information.
        """

        self.set_result_forall(
            "dataset_path",
            self.dataset_path
        )

        self.set_result_forall(
            "training_dataset_origin",
            " ".join([origin.value for origin in self.data_origins_training])
        )
        if self.data_origins_testing is not None:
            # NOTE: when DATA_ORIGINS_TESTING to none, we can split the data in the pipeline if needed.
            self.set_result_forall(
                "testing_dataset_origin", 
                " ".join([origin.value for origin in self.data_origins_testing])
            )

    def set_result_for(self, column_name: str, value: str):
        """
        Set the result for the given column name.
        NOTE : This method must be implemented in the subclass, and call the _set_result_for_pipeline method.
        """
        pass


    def _set_result_for_pipeline(self, pipeline_name: PipelineNamesEnum, column_name: str, value: str):
        """
        Set a result for a given pipeline (manage the manager on single pipeline results).
        """
        super().set_result_for(pipeline_name, column_name, value)