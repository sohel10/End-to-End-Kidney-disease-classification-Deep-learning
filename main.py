from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline


def run_stage(stage_name, pipeline_class):
    try:
        logger.info("\n" + "*" * 60)
        logger.info(f">>>>>> Stage: {stage_name} — STARTED <<<<<<")

        pipeline = pipeline_class()
        pipeline.main()

        logger.info(f">>>>>> Stage: {stage_name} — COMPLETED <<<<<<")
        logger.info("x==========x\n")

    except Exception as e:
        logger.exception(f"Error in stage: {stage_name}")
        raise e


if __name__ == "__main__":

    # 1. DATA INGESTION
    run_stage("Data Ingestion", DataIngestionTrainingPipeline)

    # 2. PREPARE BASE MODEL
    run_stage("Prepare Base Model", PrepareBaseModelTrainingPipeline)

    # 3. TRAINING
    run_stage("Training", ModelTrainingPipeline)

    # 4. EVALUATION
    run_stage("Evaluation", EvaluationPipeline)
