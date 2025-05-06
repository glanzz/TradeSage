import random
from app.transformer.dataset import get_dataset, preprocess_sentence
from app.transformer import model
from app.hyperparams import HYPERPARAMS
from app.semantic_memory import SemanticMemory
from app.context_manager import ContextManager
from app.flan_bot import Flan
import tensorflow as tf


class FinSorosBot:
    def __init__(self, HPARAMS:HYPERPARAMS):
        self.hparams = HPARAMS
        self.__boot__()

    def __boot__(self):
        self.flan = Flan()
        self.data, self.tokenizer = get_dataset(self.hparams)
        self.bot = tf.keras.models.load_model(
            self.hparams.save_model,
            custom_objects={
                "PositionalEncoding": model.PositionalEncoding,
                "MultiHeadAttentionLayer": model.MultiHeadAttentionLayer,
            },
            compile=False,
        )
        self.memory = SemanticMemory()
        self.context_manager = ContextManager(self.hparams.max_length)
    

    def greet(self):
        response = random.choice(
            [
                "Hello there! What financial advice do you seek ?",
                # "Hi! Any help you with finances?",
            ]
        )
        return response

    
    def inference(self, sentence):
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(
            self.hparams.start_token + self.tokenizer.encode(sentence) + self.hparams.end_token, axis=0
        )

        output = tf.expand_dims(self.hparams.start_token, 0)

        for i in range(self.hparams.max_length):
            predictions = self.bot(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.hparams.end_token[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)
    
    def predict(self, sentence):
        prediction = self.inference(sentence)
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size]
        )
        return predicted_sentence
    
    def say(self, sentence):
        print(sentence)

    def run_terminal(self):
        while True:
            user_input = input()
            self.say("You:"+ user_input)

            response = self.get_response(user_input)
            if response == "Exiting chatbot. Goodbye!":
                self.say(response)
                break

            self.say(f"Bot: {response}")

    
    def get_response(self, user_input):
        if user_input.lower() == "quit":
            return "Exiting chatbot. Goodbye!"
        
        inputval = self.context_manager.context_setter(user_input, self.memory)
        inputval += user_input
        
        try:
            output = self.predict(inputval)
        except Exception as e:
            lastmessage = self.memory.last_conversation()
            currentmessage = "I could not understand that can you repeat again ??"
            if lastmessage == currentmessage:
                return "There seems to be some technical difficulties, report this to Team Soros.."
            return currentmessage

        final_version = output
        try:
            for _ in range(2):
                response = self.predict(user_input+ "? "+ final_version + "Complete the full sentence.")
                if (response == output) or (response in final_version) or (output in response):
                    break
                final_version += response + "."
                output = response
                continue
                
        except Exception as e:
            print(e)
        
        print("Final:", final_version)
        final_version = self.flan.correct_grammar(final_version)
        print("Refined:", final_version)
        # lastresponse = memory.last_conversation()
        # if lastresponse and lastresponse["answer"] == output:
        #     print("Running again...")
        #     output = predict(hparams, chatbot, tokenizer, user_input)
    
        self.memory.add_interaction(user_input, final_version)
        return final_version




