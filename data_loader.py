"""
Data loader for Turkish SuperLig Wikipedia dataset.
Loads and preprocesses the dataset for RAG pipeline.
"""

import re
from typing import List, Dict, Any
from datasets import load_dataset
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


class SuperLigDataLoader:
    """Data loader for SuperLig Wikipedia dataset."""
    
    def __init__(self, dataset_name: str = "aldemirburak/superligwikipedia"):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the SuperLig Wikipedia dataset
        """
        self.dataset_name = dataset_name
        self.documents = []
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the SuperLig Wikipedia dataset from Hugging Face.
        
        Returns:
            List of document dictionaries
        """
        try:
            # Load SuperLig Wikipedia dataset
            dataset = load_dataset("aldemirburak/superligwikipedia")
            print(f"Loaded SuperLig dataset with {len(dataset['train'])} documents")
            return dataset['train']
        except Exception as e:
            print(f"Error loading SuperLig dataset: {e}")
            print("Falling back to sample data...")
            # Fallback: create sample SuperLig data
            return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """
        Create sample SuperLig data if dataset loading fails.
        
        Returns:
            List of sample document dictionaries
        """
        sample_data = [
            {
                "title": "Süper Lig",
                "text": "Süper Lig, Türkiye'nin en üst düzey futbol ligidir. 1959 yılında kurulmuştur ve 20 takımdan oluşur. Galatasaray, Fenerbahçe ve Beşiktaş en başarılı takımlardır. Ligde şu takımlar bulunmaktadır: Galatasaray, Fenerbahçe, Beşiktaş, Trabzonspor, Başakşehir, Alanyaspor, Antalyaspor, Gaziantep FK, Hatayspor, İstanbulspor, Kayserispor, Konyaspor, Pendikspor, Rizespor, Sivasspor, Adana Demirspor, Fatih Karagümrük, Kasımpaşa, Kayseri Erciyesspor ve Ankaragücü.",
                "url": "https://tr.wikipedia.org/wiki/S%C3%BCper_Lig"
            },
            {
                "title": "Galatasaray SK",
                "text": "Galatasaray Spor Kulübü, 1905 yılında kurulmuş İstanbul merkezli spor kulübüdür. En başarılı Türk futbol kulübüdür. Sarı-kırmızı renklerini kullanır. Ali Sami Yen Spor Kompleksi'nde oynar. Avrupa'da en başarılı Türk takımıdır. UEFA Kupası ve Süper Kupa kazanmıştır.",
                "url": "https://tr.wikipedia.org/wiki/Galatasaray_SK"
            },
            {
                "title": "Fenerbahçe SK",
                "text": "Fenerbahçe Spor Kulübü, 1907 yılında kurulmuş İstanbul merkezli spor kulübüdür. Sarı-lacivert renklerini kullanır. Kadıköy'de bulunur. Şükrü Saraçoğlu Stadyumu'nda oynar. Türkiye'nin en büyük taraftar kitlesine sahip kulüplerinden biridir.",
                "url": "https://tr.wikipedia.org/wiki/Fenerbah%C3%A7e_SK"
            },
            {
                "title": "Beşiktaş JK",
                "text": "Beşiktaş Jimnastik Kulübü, 1903 yılında kurulmuş İstanbul merkezli spor kulübüdür. Siyah-beyaz renklerini kullanır. Beşiktaş ilçesinde bulunur. Vodafone Park'ta oynar. Kara Kartallar lakabıyla bilinir.",
                "url": "https://tr.wikipedia.org/wiki/Be%C5%9Fikta%C5%9F_JK"
            },
            {
                "title": "Trabzonspor",
                "text": "Trabzonspor, 1967 yılında kurulmuş Trabzon merkezli futbol kulübüdür. Bordo-mavi renklerini kullanır. Karadeniz bölgesinin en başarılı takımıdır. Medical Park Stadyumu'nda oynar. Bordo Mavi renkleriyle tanınır.",
                "url": "https://tr.wikipedia.org/wiki/Trabzonspor"
            },
            {
                "title": "Başakşehir FK",
                "text": "Medipol Başakşehir Futbol Kulübü, 1990 yılında kurulmuş İstanbul merkezli futbol kulübüdür. Turuncu-lacivert renklerini kullanır. Başakşehir Fatih Terim Stadyumu'nda oynar. 2019-20 sezonunda ilk kez Süper Lig şampiyonu olmuştur.",
                "url": "https://tr.wikipedia.org/wiki/Başakşehir_FK"
            },
            {
                "title": "Alanyaspor",
                "text": "Alanyaspor, 1948 yılında kurulmuş Antalya merkezli futbol kulübüdür. Kırmızı-beyaz renklerini kullanır. Bahçeşehir Okulları Stadyumu'nda oynar. Akdeniz bölgesinin önemli takımlarından biridir.",
                "url": "https://tr.wikipedia.org/wiki/Alanyaspor"
            },
            {
                "title": "Antalyaspor",
                "text": "Antalyaspor, 1966 yılında kurulmuş Antalya merkezli futbol kulübüdür. Kırmızı-beyaz renklerini kullanır. Antalya Stadyumu'nda oynar. Akdeniz bölgesinin köklü kulüplerinden biridir.",
                "url": "https://tr.wikipedia.org/wiki/Antalyaspor"
            },
            {
                "title": "Gaziantep FK",
                "text": "Gaziantep Futbol Kulübü, 1988 yılında kurulmuş Gaziantep merkezli futbol kulübüdür. Kırmızı-siyah renklerini kullanır. Kalyon Stadyumu'nda oynar. Güneydoğu Anadolu bölgesinin önemli takımlarından biridir.",
                "url": "https://tr.wikipedia.org/wiki/Gaziantep_FK"
            },
            {
                "title": "Hatayspor",
                "text": "Hatayspor, 1967 yılında kurulmuş Hatay merkezli futbol kulübüdür. Kırmızı-beyaz renklerini kullanır. Yeni Hatay Stadyumu'nda oynar. Akdeniz bölgesinin köklü kulüplerinden biridir.",
                "url": "https://tr.wikipedia.org/wiki/Hatayspor"
            }
        ]
        print("Using enhanced sample SuperLig data")
        return sample_data
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Turkish characters
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ.,!?()-]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclamation = chunk.rfind('!')
                
                last_sentence_end = max(last_period, last_question, last_exclamation)
                if last_sentence_end > chunk_size * 0.7:  # If we found a good break point
                    chunk = chunk[:last_sentence_end + 1]
                    end = start + last_sentence_end + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def create_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """
        Create Haystack Document objects from dataset.
        
        Args:
            data: List of document dictionaries
            
        Returns:
            List of Haystack Document objects
        """
        documents = []
        
        for item in data:
            # Get text content - handle different possible field names
            text = None
            if 'text' in item:
                text = item['text']
            elif 'content' in item:
                text = item['content']
            elif 'article' in item:
                text = item['article']
            elif 'body' in item:
                text = item['body']
            else:
                # Try to find any text field
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 100:
                        text = value
                        break
            
            if not text:
                continue
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Create documents for each chunk
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                doc = Document(
                    content=chunk,
                    meta={
                        'title': item.get('title', 'Unknown'),
                        'url': item.get('url', ''),
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'source': 'wikipedia'
                    }
                )
                documents.append(doc)
        
        self.documents = documents
        print(f"Created {len(documents)} document chunks")
        return documents
    
    def load_and_prepare_data(self) -> List[Document]:
        """
        Load dataset and prepare documents for RAG pipeline.
        
        Returns:
            List of prepared Document objects
        """
        print("Loading Turkish Wikipedia dataset...")
        raw_data = self.load_dataset()
        
        print("Preprocessing and chunking documents...")
        documents = self.create_documents(raw_data)
        
        return documents


def create_document_store(documents: List[Document]) -> InMemoryDocumentStore:
    """
    Create and populate an InMemoryDocumentStore with documents.
    
    Args:
        documents: List of Document objects to store
        
    Returns:
        Configured InMemoryDocumentStore
    """
    # Create document store
    document_store = InMemoryDocumentStore()
    
    # Generate embeddings for documents first
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    
    print("Generating embeddings for documents...")
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="trmteb/turkish-embedding-model"
    )
    
    # Warm up the embedder
    doc_embedder.warm_up()
    
    # Generate embeddings for documents
    result = doc_embedder.run(documents=documents)
    embedded_documents = result["documents"]
    
    # Write embedded documents to store
    document_store.write_documents(embedded_documents)
    
    print(f"Document store created with {len(documents)} documents and embeddings")
    return document_store


if __name__ == "__main__":
    # Test the data loader
    loader = SuperLigDataLoader()
    documents = loader.load_and_prepare_data()
    
    # Create document store
    doc_store = create_document_store(documents)
    
    print("Data loading completed successfully!")
