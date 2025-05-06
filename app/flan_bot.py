from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Flan:
    def __init__(self):
        self.model_name = "google/flan-t5-small"  # or "t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        

        self.grammar_model_name = "vennify/t5-base-grammar-correction"
        self.grammar_tokenizer = AutoTokenizer.from_pretrained(self.grammar_model_name)
        self.grammar_model = AutoModelForSeq2SeqLM.from_pretrained(self.grammar_model_name)

    def synthesize_response(self, bot_output, vector_context, user_question):
        prompt = f"""User question: {user_question}
        Answers:
        {bot_output}
        {vector_context}

        Generate a final, accurate, and fluent answer combining the above answers.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
        outputs = self.model.generate(**inputs, max_length=600, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_question(self, text):
        # return text
        prompt = f"""
          Breifly elaborate the following sentences:
          {text}
          """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
        outputs = self.model.generate(**inputs, max_length=1000, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def correct_grammar(self, text):
        input_text = "gec: " + text 
        input_ids = self.grammar_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to("cpu")
        with torch.no_grad():
            outputs = self.grammar_model.generate(input_ids, max_length=1000, num_beams=4, early_stopping=False)
        return self.grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def enhance_response(self, text):
        prompt = f"""
          {text}
          Correct the above sentence.
          """
        #Enhance the clarity, coherence, and overall quality of the above sentence while ensuring that no critical information is lost.
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
        outputs = self.model.generate(**inputs, max_length=1000, num_beams=4, early_stopping=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

