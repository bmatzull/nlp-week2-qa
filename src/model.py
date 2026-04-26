from transformers import pipeline


class ExtractiveQAPipeline():
    def __init__(self, model_name: str):
        """
        Initialize the Hugging Face pipeline for Extractive QA for Task 1
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


class GenerativeQAPipeline():
    def __init__(self, model_name: str):
        """
        Initialize the Hugging Face pipeline for Generative QA for Task 2
        :param model_name:
        """
        self.pipeline = pipeline("text2text-generation", model=model_name)
        self.model_name = model_name

    def predict(self, question: str, context: str, temperature: float = 0.1) -> str:
        try:

            prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"

            result = self.pipeline(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=temperature,
            )
            return result[0]["generated_text"]
        except Exception as e:
            return f"ERROR: {str(e)}"