from collections import deque

class ContextManager:
    def __init__(self, MAX_INPUT):
        self.MAX_INPUT = MAX_INPUT
    
    def is_valid_context(self, candidates, max_input):
        return (sum(len(x) for x in candidates) < max_input)

    def context_builder(self, contexts):
        text = ""
        for context in contexts:
            text += " " + context
        return text
    
    def context_setter(self, user_input, memory):
        context_candidates = deque([])
        last_question = memory.last_conversation()
        if not last_question:
            return ""
        for relevant in memory.retrieve_relevant_memory(user_input):
            context_candidates.append(relevant)
        context_candidates.append(last_question["question"])

        while(context_candidates and (not self.is_valid_context(context_candidates, self.MAX_INPUT-len(user_input)))):
            context_candidates.popleft()
        return self.context_builder(context_candidates)
