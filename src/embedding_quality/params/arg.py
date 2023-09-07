# wrapped program flags
import argparse
import sys


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
        python src/embedding_quality/main.py [ARGUMENTS ...]

        Parse program arguments.
            -w max ml workers (threads for ML threads pool, -1 for illimited)
            -d debug
            -p pipelines
            -otr origins training
            -ots origins testing
            -h help
            --profile launch profiler
        
        usage example:
            python3 main.py -t /home/onyr/Documents/code/phdtrack/phdtrack_data/Training/Training/scp/V_7_8_P1/16 -e /home/onyr/Documents/code/phdtrack/phdtrack_data/Validation/Validation/scp/V_7_8_P1/16 -d False
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
            '--profiling',
            action='store_true',
            help="Launch profiler"
        )

        # save parsed arguments
        self.args = parser.parse_args()