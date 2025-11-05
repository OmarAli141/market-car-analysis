import re
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ReviewAnalyzer:
    """
    Handles the generation of CV analysis using a local Ollama model.
    """
    def __init__(self):
        """
        Initializes the ChatOllama model.
        """
        self.llm = ChatOllama(
            model="deepseek-r1:1.5b",
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            num_ctx=2048
        )
        print("ReviewAnalyzer initialized with model 'deepseek-r1:1.5b'.")


    def _clean_deepseek_think(self, text: str) -> str:
        """
        Remove DeepSeek R1 <think>...</think> blocks from the output.
        """
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()


    def _build_prompt(self, context_block: str, question: str) -> ChatPromptTemplate:
        """
        Construct a retrieval-augmented prompt that restricts answers to context.
        """
        template = (
            "You are an assistant analyzing car owner reviews.\n"
            "Answer the user question using ONLY the provided context.\n"
            "If the answer cannot be derived from the context, say 'I don't know'.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return ChatPromptTemplate.from_template(template).partial(context=context_block, question=question)


    def analyze_with_results(self, results: dict, question: str, max_items: int = 5) -> str:
        """
        Take a Chroma query result dict and a question, then generate an answer.
        """
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        top_docs = docs[0][:max_items] if docs else []
        top_metas = metas[0][:max_items] if metas else []

        contexts = []
        for doc, meta in zip(top_docs, top_metas):
            model = (meta or {}).get("Vehicle_Model", "Unknown")
            rating = (meta or {}).get("Rating", "?")
            contexts.append(f"[Model: {model}, Rating: {rating}] {doc}")

        context_block = "\n\n".join(contexts) if contexts else "(no context)"
        prompt = self._build_prompt(context_block, question)

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({})
        return self._clean_deepseek_think(raw)


    def analyze_with_context(self, context_items: list[str], question: str) -> str:
        """
        Take a list of context strings and a question, then generate an answer.
        """
        context_block = "\n\n".join(context_items)
        prompt = self._build_prompt(context_block, question)
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({})
        return self._clean_deepseek_think(raw)
