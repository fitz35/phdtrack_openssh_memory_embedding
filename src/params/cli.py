# direct raw access to params
import sys
import argparse

from params.pipelines import Pipeline


# wrapped program flags
class CLIArguments:
    args: argparse.Namespace

    def __init__(self) -> None:
        self.__log_raw_argv()
        self.__parse_argv()
    
    def __log_raw_argv(self) -> None:
        print("Passed program params:")
        for i in range(len(sys.argv)):
            print("param[{0}]: {1}".format(
                i, sys.argv[i]
            ))
    
    def __parse_argv(self) -> None:
        """
        python main [ARGUMENTS ...]

        Parse program arguments.
            -w max ml workers (threads for ML threads pool, -1 for illimited)
            --debug debug
            -d dataset path to use
            -otr origins training
            -ots origins testing
            -o output folder
            -p pipelines to run
            
        """
        parser = argparse.ArgumentParser(description='Program [ARGUMENTS]')
        parser.add_argument(
            '--debug', 
            action='store_true',
            help="debug, True or False"
        )
        parser.add_argument(
            '-w',
            '--max_ml_workers', 
            type=int, 
            default=None,
            help="max ml workers (threads for ML threads pool, -1 for illimited)"
        )
        parser.add_argument(
            "-d",
            "--dataset_path",
            type=str,
            help="Dataset path to use."
        )
        parser.add_argument(
            '-otr',
            '--origins_training',
            type=str,
            nargs='*',
            default=None,
            help="Data origin (training, validation, testing) for training"
        )
        parser.add_argument(
            '-ots',
            '--origins_testing',
            type=str,
            nargs='*',
            default=None,
            help="Data origin (training, validation, testing) for testing"
        )
        parser.add_argument(
            '-o',
            '--output_folder',
            type=str,
            help="the path of the output folder (save the advancement)"
        )
        parser.add_argument(
            '-p',
            '--pipelines',
            type=str,
            default=None,
            help="List of pipelines to run: " + str(list(map(lambda x: x.name.lower(), Pipeline)))
        )
        parser.add_argument(
            '--no_balancing', 
            action='store_true',
            help="unactivate balancing, True or False"
        )
        # save parsed arguments
        self.args = parser.parse_args()