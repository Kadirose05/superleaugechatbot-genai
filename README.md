# ⚽ Türkçe RAG Chatbot - Süper Lig Asistanı

Bu proje, Türkçe dilinde çalışan bir RAG (Retrieval-Augmented Generation) chatbot'udur. Süper Lig hakkında soruları yanıtlamak için Haystack, Google Gemini 2.0 Flash ve Türkçe embedding modeli kullanır.

## ⚽ Canlı Demo Linki

https://superleaugechatbot-genai-vdnl6479kkmpysu93wkpsh.streamlit.app

## 🧠 Teknoloji Stack'i

- **RAG Framework**: Haystack
- **UI Framework**: Streamlit
- **Embedding Model**: trmteb/turkish-embedding-model (Sentence Transformers)
- **LLM**: Google Gemini 2.0 Flash
- **Vector Store**: InMemoryDocumentStore
- **Dataset**: Türkçe Wikipedia (Süper Lig) (https://huggingface.co/datasets/aldemirburak/superligwikipedia)
- **Dil**: Türkçe

## 📁 Proje Yapısı

```
rag_chatbot/
│
├── app.py                 # Streamlit web arayüzü
├── rag_pipeline.py        # RAG pipeline (Haystack)
├── data_loader.py         # Veri yükleme ve işleme
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
└── .env                  # Çevre değişkenleri (oluşturulacak)
```

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.10+
- Google Gemini API anahtarı

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. API Key Ayarlama

Proje klasöründe `.env` dosyası oluşturun:

```bash
# Windows
echo GOOGLE_API_KEY=your_google_api_key_here > .env

# Linux/Mac  
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

**Örnek .env dosyası:**
```
GOOGLE_API_KEY=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Google Gemini API Anahtarı Alma

1. [Google AI Studio](https://makersuite.google.com/app/apikey) adresine gidin
2. Google hesabınızla giriş yapın
3. "Create API Key" butonuna tıklayın
4. API anahtarınızı kopyalayın ve `.env` dosyasına ekleyin

## 🎯 Kullanım

### Streamlit Web Arayüzü

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

### Özellikler

- **Türkçe Soru-Cevap**: Süper Lig hakkında Türkçe sorular sorabilirsiniz
- **Kaynak Gösterimi**: Bulunan kaynak belgeleri görüntüleyebilirsiniz
- **Örnek Sorular**: Hazır örnek sorular ile hızlı başlangıç
- **Skor Gösterimi**: Bulunan belgelerin benzerlik skorlarını görebilirsiniz
- **Temiz Arayüz**: Modern ve kullanıcı dostu Streamlit arayüzü

### Örnek Sorular

- "Galatasaray hakkında bilgi ver"
- "Süper Lig'de hangi takımlar var?"
- "Fenerbahçe'nin renkleri nelerdir?"
- "Trabzonspor ne zaman kuruldu?"

## 🔧 Geliştirici Kullanımı

### Doğrudan RAG Pipeline Kullanımı

```python
from data_loader import SuperLigDataLoader, create_document_store
from rag_pipeline import RAGPipelineManager

# Veri yükleme
loader = SuperLigDataLoader()
documents = loader.load_and_prepare_data()
doc_store = create_document_store(documents)

# RAG pipeline başlatma
rag_manager = RAGPipelineManager(doc_store, api_key="your_api_key")

# Soru sorma
result = rag_manager.ask_question("Galatasaray hakkında bilgi ver")
print(result['answer'])
```

### Sadece Belge Arama

```python
# Benzer belgeleri bulma
similar_docs = rag_manager.get_similar_documents("Galatasaray", top_k=3)
for doc in similar_docs:
    print(f"Başlık: {doc['title']}")
    print(f"İçerik: {doc['content']}")
    print(f"Skor: {doc['score']}")
```

## 📊 Sistem Mimarisi

```
Kullanıcı Sorusu
       ↓
   Streamlit UI
       ↓
   RAG Pipeline
       ↓
┌─────────────────┐
│   Retriever     │ ← Turkish Embedding Model
│   (Haystack)    │
└─────────────────┘
       ↓
   Document Store
       ↓
┌─────────────────┐
│   Generator     │ ← Google Gemini 2.0 Flash
│   (Haystack)    │
└─────────────────┘
       ↓
   Türkçe Yanıt
```

## 🛠️ Özelleştirme

### Farklı Veri Seti Kullanma

`data_loader.py` dosyasında `SuperLigDataLoader` sınıfını değiştirerek farklı veri setleri kullanabilirsiniz:

```python
# Farklı Wikipedia dil versiyonu
loader = SuperLigDataLoader(dataset_name="wikipedia", language="en")

# Özel veri seti
custom_data = [{"title": "Başlık", "text": "İçerik", "url": "URL"}]
documents = loader.create_documents(custom_data)
```

### Embedding Model Değiştirme

`rag_pipeline.py` dosyasında farklı embedding modelleri kullanabilirsiniz:

```python
# Farklı Türkçe model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

### Retrieval Parametreleri

```python
# Daha fazla belge getir
self.retriever.top_k = 10

# Farklı benzerlik eşiği
self.retriever.similarity_threshold = 0.7
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

1. **API Key Hatası**
   ```
   ValueError: Google API key is required
   ```
   - `.env` dosyasında `GOOGLE_API_KEY` değişkenini kontrol edin

2. **Model Yükleme Hatası**
   ```
   OSError: Model not found
   ```
   - İnternet bağlantınızı kontrol edin
   - Model otomatik olarak indirilecektir

3. **Memory Hatası**
   ```
   OutOfMemoryError
   ```
   - `chunk_size` parametresini küçültün
   - Daha az belge kullanın

### Performans Optimizasyonu

- **Embedding Cache**: İlk çalıştırmada modeller indirilir, sonraki çalıştırmalarda cache kullanılır
- **Document Chunking**: Belge boyutunu optimize edin
- **Top-K Ayarlama**: Daha az belge = daha hızlı yanıt

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

**⚽ Türkçe RAG Chatbot - Süper Lig Asistanı ile futbol dünyasını keşfedin!**
