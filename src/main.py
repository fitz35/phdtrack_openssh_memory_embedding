from params.common_params import CommonProgramParams
from params.pipeline_name_to_pipeline import PIPELINES_NAME_TO_PIPELINE


def main():
    PIPELINES_NAME_TO_PIPELINE[params.pipelines](params)

if __name__ == "__main__":

    print("🚀 Running program...")
    params = CommonProgramParams(
        "embedding_test",
        ".env",
    )
    
    main()