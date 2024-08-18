class DocumentProcessor:
    def __init__(self, collection_name, search_only: bool):
        self.text_processor = TextPreprocessor()
        self.vector_db = VectorDBOperations()
        self.embedder = Embedder()
        if search_only is False:
            self.vector_db.create_collection(collection_name)
        self.collection_name = collection_name