"""
RAG Pipeline implementation using Haystack 2.x.
Combines retrieval and generation for Turkish language chatbot.
"""

import os
from typing import List, Dict, Any, Optional
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TurkishRAGPipeline:
    """RAG Pipeline for Turkish language chatbot using Haystack 2.x and Google Gemini."""
    
    def __init__(self, document_store: InMemoryDocumentStore, api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            document_store: Haystack document store with indexed documents
            api_key: Google API key (if not provided, will try to get from environment)
        """
        self.document_store = document_store
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            print("Warning: No Google API key provided. Generation will not work, but retrieval will.")
        
        # Configure Google Gemini (only if API key is available)
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
        
        # Initialize components
        self.retriever = None
        self.prompt_builder = None
        self.pipeline = None
        
        self._setup_components()
        self._create_pipeline()
    
    def _setup_components(self):
        """Setup Haystack components for the RAG pipeline."""
        # Initialize Turkish embedding model
        print("Loading Turkish embedding model...")
        self.embedding_model = SentenceTransformer('trmteb/turkish-embedding-model')
        
        # Create document embedder
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model="trmteb/turkish-embedding-model"
        )
        
        # Create text embedder for queries
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="trmteb/turkish-embedding-model"
        )
        
        # Create retriever
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5  # Retrieve top 5 most relevant documents
        )
        
        # Create prompt template for Turkish responses
        prompt_template = """Sen Türkçe bir futbol asistanısın. Aşağıdaki bağlam bilgilerini kullanarak kullanıcının sorusunu Türkçe olarak yanıtla.

Bağlam:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Soru: {{ query }}

Yanıt:"""
        
        # Create prompt builder
        self.prompt_builder = PromptBuilder(template=prompt_template)
    
    def _create_pipeline(self):
        """Create the Haystack RAG pipeline."""
        self.pipeline = Pipeline()
        self.pipeline.add_component("text_embedder", self.text_embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        
        # Connect components
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User question in Turkish
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and retrieved documents
        """
        try:
            # Update retriever top_k if different
            if self.retriever.top_k != top_k:
                self.retriever.top_k = top_k
            
            # Run the pipeline to get documents
            result = self.pipeline.run({
                "text_embedder": {"text": question},
                "prompt_builder": {"query": question}
            })
            
            # Extract retrieved documents
            documents = result.get('retriever', {}).get('documents', [])
            
            # Format retrieved documents
            retrieved_docs = []
            for doc in documents:
                retrieved_docs.append({
                    'content': doc.content,
                    'title': doc.meta.get('title', 'Unknown'),
                    'url': doc.meta.get('url', ''),
                    'score': getattr(doc, 'score', 0.0)
                })
            
            # Generate answer using Gemini (if available)
            if self.model:
                context_text = "\n\n".join([doc['content'] for doc in retrieved_docs])
                prompt = f"""Sen Türkçe bir futbol asistanısın. Aşağıdaki bağlam bilgilerini kullanarak kullanıcının sorusunu Türkçe olarak yanıtla.

Bağlam:
{context_text}

Soru: {question}

Yanıt:"""
                
                # Generate answer using Gemini
                response = self.model.generate_content(prompt)
                answer = response.text
            else:
                # No API key - return retrieval info only
                answer = f"API key bulunamadı. {len(retrieved_docs)} belge bulundu. API key ayarlayarak tam yanıt alabilirsiniz."
            
            return {
                'answer': answer,
                'documents': retrieved_docs,
                'query': question
            }
            
        except Exception as e:
            print(f"Error in RAG pipeline query: {e}")
            return {
                'answer': f"Üzgünüm, sorunuzu yanıtlayamadım. Hata: {str(e)}",
                'documents': [],
                'query': question
            }
    
    def get_context_only(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get only the retrieved context documents without generation.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document dictionaries
        """
        try:
            # Update retriever top_k
            self.retriever.top_k = top_k
            
            # Get documents from retriever
            result = self.pipeline.run({
                "text_embedder": {"text": question}
            })
            documents = result.get('retriever', {}).get('documents', [])
            
            # Format documents
            retrieved_docs = []
            for doc in documents:
                retrieved_docs.append({
                    'content': doc.content,
                    'title': doc.meta.get('title', 'Unknown'),
                    'url': doc.meta.get('url', ''),
                    'score': getattr(doc, 'score', 0.0)
                })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer_with_context(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using provided context documents.
        
        Args:
            question: User question
            context_docs: List of context documents
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context text
            context_text = "\n\n".join([doc['content'] for doc in context_docs])
            
            # Create prompt
            prompt = f"""Sen Türkçe bir futbol asistanısın. Aşağıdaki bağlam bilgilerini kullanarak kullanıcının sorusunu Türkçe olarak yanıtla.

Bağlam:
{context_text}

Soru: {question}

Yanıt:"""
            
            # Generate answer using Gemini (if available)
            if self.model:
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return f"API key bulunamadı. {len(context_docs)} belge bulundu. API key ayarlayarak tam yanıt alabilirsiniz."
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Üzgünüm, yanıt oluşturamadım. Hata: {str(e)}"


class RAGPipelineManager:
    """Manager class for RAG pipeline operations."""
    
    def __init__(self, document_store: InMemoryDocumentStore, api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline manager.
        
        Args:
            document_store: Document store with indexed documents
            api_key: Google API key
        """
        self.pipeline = TurkishRAGPipeline(document_store, api_key)
    
    def ask_question(self, question: str, show_context: bool = False) -> Dict[str, Any]:
        """
        Ask a question to the RAG pipeline.
        
        Args:
            question: User question
            show_context: Whether to include context documents in response
            
        Returns:
            Response dictionary
        """
        result = self.pipeline.query(question)
        
        if show_context:
            return result
        else:
            return {
                'answer': result['answer'],
                'query': result['query']
            }
    
    def get_similar_documents(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar documents for a question.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        return self.pipeline.get_context_only(question, top_k)


if __name__ == "__main__":
    # Test the RAG pipeline
    from data_loader import SuperLigDataLoader, create_document_store
    
    # Load data and create document store
    loader = SuperLigDataLoader()
    documents = loader.load_and_prepare_data()
    doc_store = create_document_store(documents)
    
    # Initialize RAG pipeline
    try:
        rag_manager = RAGPipelineManager(doc_store)
        
        # Test query
        test_question = "Galatasaray hakkında bilgi ver"
        result = rag_manager.ask_question(test_question, show_context=True)
        
        print(f"Soru: {result['query']}")
        print(f"Yanıt: {result['answer']}")
        print(f"Bulunan belgeler: {len(result['documents'])}")
        
    except Exception as e:
        print(f"RAG pipeline test failed: {e}")
        print("Make sure to set GOOGLE_API_KEY environment variable")