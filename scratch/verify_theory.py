from thefuzz import fuzz
import re

query1 = "i wanna build an AI agent"
query2 = "hi can u suggest me how to learn about various sorts of RAGS"

print("Fuzzy token_set_ratio:")
print(f"  'i wanna build an AI agent' vs 'Agentic System': {fuzz.token_set_ratio(query1.lower(), 'agentic system')}")
print(f"  'hi can u suggest me how to learn about various sorts of RAGS' vs 'RAG': {fuzz.token_set_ratio(query2.lower(), 'rag')}")

# Question framing stripping logic in the code:
q_core_regex = r"^(what is|what's|whats|explain|tell me about|how does|define)\s+"

q_core1 = re.sub(q_core_regex, "", query1.lower(), flags=re.I).strip(" ?!.")
q_core2 = re.sub(q_core_regex, "", query2.lower(), flags=re.I).strip(" ?!.")

print("\nStripped q_core:")
print(f"  Query 1 core: '{q_core1}'")
print(f"  Query 2 core: '{q_core2}'")

# Words after clean query words:
def _clean_query_words(text):
    return {
        word.strip("s?,.!-:;()[]{}\"'")
        for word in (text or "").lower().split()
        if len(word.strip("s?,.!-:;()[]{}\"'")) > 1
    }

print("\nCleaned query words:")
print(f"  Query 1 words: {_clean_query_words(query1)}")
print(f"  Query 2 words: {_clean_query_words(query2)}")
