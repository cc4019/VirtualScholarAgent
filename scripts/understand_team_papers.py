import os
import logging
from pathlib import Path
import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logging.error("No OpenAI API key found in environment variables!")
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")

# Directories
DATA_DIR = Path(__file__).resolve().parent / "data"
TEAM_DOCS_DIR = DATA_DIR / "team_docs"
OUTPUT_DIR = DATA_DIR / "analysis_results"

# Ensure directories exist
for path in [TEAM_DOCS_DIR, OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
            
def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF files."""
    logging.debug(f"Starting PDF extraction for: {pdf_path}")
    logging.debug(f"File exists: {pdf_path.exists()}")
    logging.debug(f"File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            logging.debug("Creating PDF reader...")
            pdf_reader = PyPDF2.PdfReader(file, strict=False)
            
            logging.debug(f"PDF loaded. Pages: {len(pdf_reader.pages)}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                logging.debug(f"Processing page {page_num}")
                try:
                    page_text = page.extract_text()
                    text += page_text
                    logging.debug(f"Page {page_num} extracted: {len(page_text)} chars")
                except Exception as e:
                    logging.error(f"Error on page {page_num}: {str(e)}")
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
    
    logging.debug(f"Total text extracted: {len(text)} chars")
    return text

# Define the expected output structure
class DocumentAnalysis(BaseModel):
    key_research_areas: List[str] = Field(
        description="Key research areas identified in the document",
        min_items=1
    )
    critical_analysis: List[str] = Field(
        description="Critical analysis points from the document",
        min_items=1
    )
    research_gaps: List[str] = Field(
        description="Identified research gaps",
        min_items=1
    )
    keywords: List[str] = Field(
        description="Five key keywords from the document",
        min_items=5,
        max_items=5
    )

    @classmethod
    def validate_output(cls, data: dict) -> tuple[bool, str]:
        """Validate the analysis output meets requirements"""
        try:
            if not isinstance(data, dict):
                return False, "Output must be a dictionary"
                
            instance = cls(**data)
            validations = [
                (len(instance.keywords) == 5, "Must have exactly 5 keywords"),
                (len(instance.key_research_areas) > 0, "Must have at least one research area"),
                (len(instance.critical_analysis) > 0, "Must have at least one critical analysis point"),
                (len(instance.research_gaps) > 0, "Must have at least one research gap")
            ]
            
            for is_valid, message in validations:
                if not is_valid:
                    return False, message
                    
            return True, "Valid output"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

def create_document_analyzer():
    """Create and configure the document analysis chain"""
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.1,
        max_retries=3
    )

    # Updated prompt to be more direct
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document analysis expert. Your task is to analyze documents and return a JSON object with specific insights.
        Do not include any additional text or explanations - only return the JSON object."""),
        ("user", """Analyze this text and return ONLY a JSON object with exactly this structure:
        {{
            "key_research_areas": ["area1", "area2", ...],
            "critical_analysis": ["point1", "point2", ...],
            "research_gaps": ["gap1", "gap2", ...],
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
        }}
        
        Text to analyze: {text}
        
        Return ONLY the JSON object, no other text.""")
    ])

    return prompt | llm

def analyze_document(text: str, chain) -> dict:
    """
    Analyze a document and return structured results.
    
    Raises:
        ValueError: If validation fails or no valid analysis is produced
        Exception: For other unexpected errors
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = chain.invoke({"text": text})
            if not response:
                raise ValueError("Empty response from LLM")
                
            # Clean and parse JSON
            json_str = response.content
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            json_start = json_str.find('{')
            json_end = json_str.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON found in response")
            
            json_str = json_str[json_start:json_end]
            
            # Parse with Pydantic
            try:
                result = DocumentAnalysis.model_validate_json(json_str)
            except Exception as e:
                raise ValueError(f"JSON validation failed: {str(e)}")
            
            # Return validated result
            return {
                "key_research_areas": result.key_research_areas,
                "critical_analysis": result.critical_analysis,
                "research_gaps": result.research_gaps,
                "keywords": result.keywords
            }
            
        except Exception as e:
            logging.warning(f"Attempt {retry_count + 1} failed: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                raise ValueError(f"Failed after {max_retries} attempts: {str(e)}")

def analyze_internal_documents(input_dir: Path, output_dir: Path):
    """
    Analyze all documents in the input directory and save results.
    
    Raises:
        FileNotFoundError: If input files cannot be found or read
        ValueError: If document analysis fails
        IOError: If output cannot be written
    """
    chain = create_document_analyzer()
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")
    
    total_files = len(pdf_files)
    successful = 0
    failed = 0
    
    logging.info(f"Found {total_files} PDF files to process")

    # Define output file path
    output_file = output_dir / "analysis_results.json"
    
    # Load existing results if file exists
    existing_results = {}
    if output_file.exists():
        try:
            with open(output_file, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Could not read existing results from {output_file}, starting fresh")

    for index, file_path in enumerate(pdf_files, 1):
        logging.info(f"Processing {index}/{total_files}: {file_path}")
        
        try:
            text = extract_text_from_pdf(file_path)
            if not text.strip():
                raise ValueError(f"No text extracted from {file_path}")

            result = analyze_document(text, chain)
            
            # Add result to existing results with document name as key
            existing_results[file_path.stem] = result
            
            try:
                # Write updated results to file
                with open(output_file, "w", encoding='utf-8') as f:
                    json.dump(existing_results, f, indent=4, ensure_ascii=False)
            except IOError as e:
                raise IOError(f"Failed to write results to {output_file}: {str(e)}")
            
            successful += 1
            logging.info(f"Analysis for {file_path.stem} added to {output_file}")

        except Exception as e:
            failed += 1
            raise Exception(f"Error processing {file_path}: {str(e)}")

    if failed > 0:
        raise ValueError(f"Processing complete but with errors. Success: {successful}, Failed: {failed}")
    
    logging.info(f"Processing complete. All {successful} files processed successfully.")

def main():
    """
    Main entry point for document analysis.
    
    Raises:
        Exception: If any part of the analysis process fails
    """
    logging.info("Starting document analysis...")
    try:
        analyze_internal_documents(TEAM_DOCS_DIR, OUTPUT_DIR)
        logging.info("All documents analyzed successfully.")
    except Exception as e:
        logging.error(f"Error during document analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()