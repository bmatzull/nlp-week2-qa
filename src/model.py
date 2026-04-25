from transformers import pipeline


class ExtractiveQAPipeline():
    def __init__(self, model_name: str):
        """
        Initialize the Hugging Face pipeline for Extractive QA
        :param model_name:
        """

        self.pipeline = pipeline("question-answering", model=model_name)
        self.model_name = model_name

    def predict(self, question: str, context: str) -> str:
        try:
            result = self.pipeline(question=question, context=context)
            return result['answer']
        except Exception as e:
            return f"ERROR: {str(e)}"