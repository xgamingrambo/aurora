import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Document processing
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings and Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Environment setup
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDBPipeline:
    """Complete pipeline for creating vector database from Excel and PDF files"""
    
    def __init__(self, 
                 data_folder: str = "data",
                 index_name: str = "health-docs",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the pipeline
        
        Args:
            data_folder: Folder containing Excel and PDF files
            index_name: Name for Pinecone index
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.data_folder = Path(data_folder)
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Validate environment variables
        self._validate_environment()
        
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        logger.info("Environment variables validated successfully")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and create/connect to index"""
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'  # Change as needed
                    )
                )
                logger.info("Index created successfully")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to check for duplicates"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _load_pdf_documents(self, file_path: Path) -> List[Document]:
        """Load and process PDF documents"""
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Add metadata to each document
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': 'pdf',
                    'file_size': file_path.stat().st_size,
                    'page_number': i + 1,
                    'total_pages': len(documents),
                    'file_hash': self._get_file_hash(file_path),
                    'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'processed_date': datetime.now().isoformat()
                })
            
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def _load_excel_documents(self, file_path: Path) -> List[Document]:
        """Load and process Excel documents"""
        try:
            logger.info(f"Loading Excel: {file_path}")
            
            # Read Excel file to get sheet information
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            documents = []
            
            for sheet_name in sheet_names:
                try:
                    # Load each sheet as a document
                    loader = UnstructuredExcelLoader(
                        str(file_path),
                        mode="elements"
                    )
                    sheet_docs = loader.load()
                    
                    # Process each sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Convert DataFrame to text representation
                    sheet_content = f"Sheet: {sheet_name}\n\n"
                    
                    # Add column information
                    sheet_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                    
                    # Add data summary
                    sheet_content += f"Rows: {len(df)}\n"
                    sheet_content += f"Summary:\n{df.describe(include='all').to_string()}\n\n"
                    
                    # Add actual data (first 100 rows to avoid too large chunks)
                    display_df = df.head(100)
                    sheet_content += f"Data:\n{display_df.to_string()}"
                    
                    # Create document for this sheet
                    doc = Document(
                        page_content=sheet_content,
                        metadata={
                            'source': str(file_path),
                            'file_name': file_path.name,
                            'file_type': 'excel',
                            'file_size': file_path.stat().st_size,
                            'sheet_name': sheet_name,
                            'sheet_index': sheet_names.index(sheet_name),
                            'total_sheets': len(sheet_names),
                            'rows_count': len(df),
                            'columns_count': len(df.columns),
                            'columns': df.columns.tolist(),
                            'file_hash': self._get_file_hash(file_path),
                            'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            'processed_date': datetime.now().isoformat()
                        }
                    )
                    
                    documents.append(doc)
                    
                except Exception as sheet_error:
                    logger.warning(f"Error processing sheet '{sheet_name}' in {file_path}: {str(sheet_error)}")
                    continue
            
            logger.info(f"Loaded {len(documents)} sheets from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Excel {file_path}: {str(e)}")
            return []
    
    def _load_all_documents(self) -> List[Document]:
        """Load all PDF and Excel documents from the data folder"""
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder '{self.data_folder}' does not exist")
        
        all_documents = []
        
        # Define supported file extensions
        pdf_extensions = ['.pdf']
        excel_extensions = ['.xlsx', '.xls', '.xlsm']
        
        # Process all files in the data folder
        for file_path in self.data_folder.rglob('*'):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                
                if file_extension in pdf_extensions:
                    documents = self._load_pdf_documents(file_path)
                    all_documents.extend(documents)
                    
                elif file_extension in excel_extensions:
                    documents = self._load_excel_documents(file_path)
                    all_documents.extend(documents)
                    
                else:
                    logger.info(f"Skipping unsupported file: {file_path}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        logger.info("Chunking documents...")
        
        chunked_documents = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{doc.metadata.get('file_name', 'unknown')}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.page_content)
                })
            
            chunked_documents.extend(chunks)
        
        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents
    
    def _create_vector_store(self, documents: List[Document]):
        """Create vector store and add documents"""
        try:
            logger.info("Creating vector store...")
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            # Add documents in batches to avoid rate limits
            batch_size = 50
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                try:
                    self.vector_store.add_documents(batch)
                    logger.info(f"Successfully added batch {batch_num}")
                except Exception as batch_error:
                    logger.error(f"Error adding batch {batch_num}: {str(batch_error)}")
                    continue
            
            logger.info("Vector store creation completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            logger.info("Starting vector database pipeline...")
            
            # Step 1: Initialize components
            logger.info("Step 1: Initializing components...")
            self._initialize_pinecone()
            self._initialize_embeddings()
            
            # Step 2: Load documents
            logger.info("Step 2: Loading documents...")
            documents = self._load_all_documents()
            
            if not documents:
                logger.warning("No documents found to process")
                return
            
            # Step 3: Chunk documents
            logger.info("Step 3: Chunking documents...")
            chunked_documents = self._chunk_documents(documents)
            
            # Step 4: Create vector store
            logger.info("Step 4: Creating vector store...")
            self._create_vector_store(chunked_documents)
            
            logger.info("Pipeline completed successfully!")
            
            # Print summary
            self._print_summary(documents, chunked_documents)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _print_summary(self, documents: List[Document], chunked_documents: List[Document]):
        """Print pipeline summary"""
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        
        # Count files by type
        pdf_count = sum(1 for doc in documents if doc.metadata.get('file_type') == 'pdf')
        excel_count = sum(1 for doc in documents if doc.metadata.get('file_type') == 'excel')
        
        print(f"Files processed:")
        print(f"  - PDF files: {pdf_count}")
        print(f"  - Excel files: {excel_count}")
        print(f"  - Total documents: {len(documents)}")
        print(f"  - Total chunks: {len(chunked_documents)}")
        print(f"  - Index name: {self.index_name}")
        print(f"  - Chunk size: {self.chunk_size}")
        print(f"  - Chunk overlap: {self.chunk_overlap}")
        print("="*50)
    
    def search_similarity(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Run the pipeline first.")
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    'score': score,
                    'metadata': doc.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
        



def main():
    """Main function to run the pipeline"""
    
    # Create .env file template if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_template = """# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here
"""
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        print("Created .env file template. Please add your API keys and run again.")
        return
    
    try:
        # Initialize and run pipeline
        pipeline = VectorDBPipeline(
            data_folder="data",
            index_name="health-docs",
            chunk_size=800,
            chunk_overlap=200
        )
        
        pipeline.run_pipeline()
        
        # Example search
        print("\n" + "="*50)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()