from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streaming_handler import StreamingHandler
from streamlit_handler import StreamlitStreamingHandler
from llm_factory import LLMFactory
from document_processor import DocumentProcessor

class RAGPipeline:
    def __init__(self, config, handler=None):
        self.config = config
        llm_mode = self.config["mode"]
        llm_model_name = self.config["model_name"]
        callbacks = [handler] if handler else []
        self.llm = LLMFactory.create_llm(llm_mode, llm_model_name, callbacks)
        self.chain = None

    def setup(self):
        if self.config.get("ingest_docs"):
            self._setup_rag_chain()
        else:
            self._setup_chatbot_chain()

    def _setup_rag_chain(self):
        prompt = self._create_rag_prompt()
        doc_processor = DocumentProcessor(self.config)
        vector_store = doc_processor.get_vector_store()

        if vector_store:
            print(f"Number of documents in vector store: {vector_store._collection.count()}")
        else:
            print("Vector store is not initialized. RAG functionality will not work.")
            return

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def _setup_chatbot_chain(self):
        prompt = self._create_chatbot_prompt()
        self.chain = prompt | self.llm

    def _create_rag_prompt(self):
        prompt_template = (
            "You are an intelligent assistant that answers questions based on provided documents.\n"
            "Use the following context to answer the question at the end.\n"
            "If you don't know the answer, just say \"I don't know\" rather than making up an answer.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def _create_chatbot_prompt(self):
        return PromptTemplate(
            template="You are a helpful assistant. Answer the following question.\nQuestion: {query}\nAnswer:",
            input_variables=["query"],
        )

    def chat(self):
        print("\nChat Interface - Type 'quit' to exit")
        print("=" * 50)
        if self.config.get("ingest_docs"):
            print("Ready to answer questions based on your documents!")
        else:
            print("Ready for a chat!")

        print(f"{self.config['mode'].replace('_', ' ').title()} model: {self.config['model_name']}")

        while True:
            if not self.chain:
                print("The RAG chain is not set up. Please check your configuration.")
                break
            question = input("\nAsk a question (or type 'quit' to exit): ")
            if question.lower().strip() == "quit":
                print("Goodbye!")
                break
            if not question.strip():
                continue

            try:
                print("Thinking...")
                print("-" * 30)
                response = self.chain.invoke({"query": question})
                if self.config.get("ingest_docs"):
                    print("\n\n--- Source Documents ---")
                    for doc in response['source_documents']:
                        print(f"  - {doc.metadata['source']}")
                    print("----------------------\n")
                elif not self.config.get("ingest_docs"):
                    print(response.content)
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")
