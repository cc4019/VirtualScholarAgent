import json
import logging
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, TypedDict, Literal
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.llms import Ollama
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangSmith imports
from langsmith import Client as LangSmithClient
from langsmith.run_helpers import traceable
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

# ArXiv imports
from arxiv import Client, Search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Paper-Retrieval-Workflow"
# Set these in your environment or add them here
# os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Load environment variables from .env file
load_dotenv()

# Check if the API key is available
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment variables. Make sure it's set in your .env file.")

# Initialize LangSmith client
try:
    langsmith_client = LangSmithClient()
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize LangSmith client: {str(e)}")
    langsmith_client = None

# State definition for the graph
class GraphState(TypedDict):
    analysis_points: Dict[str, List[str]]
    context: str
    retrieved_papers: List[Dict]
    paper_evaluations: List[Dict]
    refined_queries: List[Dict]
    avg_score: float
    need_refinement: bool

@traceable(run_type="chain")
def load_analysis_results():
    """Load and process analysis results from JSON"""
    analysis_file = Path("scripts/data/analysis_results/analysis_results.json")
    
    try:
        with open(analysis_file, "r", encoding='utf-8') as f:
            results = json.load(f)
        
        analysis_points = {
            "research_gaps": [],
            "key_research_areas": [],
            "critical_analysis": [],
            "keywords": []
        }
        
        if isinstance(results, dict):
            for doc_name, analysis in results.items():
                if isinstance(analysis, dict):
                    analysis_points["research_gaps"].extend(analysis.get("research_gaps", []))
                    analysis_points["key_research_areas"].extend(analysis.get("key_research_areas", []))
                    analysis_points["critical_analysis"].extend(analysis.get("critical_analysis", []))
                    analysis_points["keywords"].extend(analysis.get("keywords", []))
        elif isinstance(results, list):
            analysis_points["research_gaps"].extend(results)
        
        for key in analysis_points:
            analysis_points[key] = list(dict.fromkeys(analysis_points[key]))
        
        if not any(analysis_points.values()):
            raise ValueError("No analysis points found in results")
        
        return analysis_points
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Analysis results file not found at {analysis_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {analysis_file}")

# Helper function to extract JSON from text
@traceable(run_type="chain")
def extract_json(text: str) -> dict:
    """Extract JSON from text output"""
    try:
        # Try direct JSON parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON object in text
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return empty dict
        logger.warning("Could not extract JSON from text")
        return {}

# Node functions
@traceable(run_type="chain")
def initialize(state: GraphState) -> GraphState:
    """Initialize the workflow state"""
    # Create context from analysis points
    analysis_points = state.get("analysis_points", {})
    context = create_context_from_analysis(analysis_points)
    
    # Update state with context
    new_state = dict(state)
    new_state["context"] = context
    
    logger.info("Initialized workflow state with context")
    return new_state

@traceable(run_type="agent")
def paper_search_agent(state: GraphState) -> GraphState:
    """Agent node for searching and retrieving academic papers"""
    context = state["context"]
    analysis_points = state["analysis_points"]
    
    # Get date filter from state or use default
    date_filter = state.get("date_filter", "last_month")
    
    # Step 1: Generate optimized search queries using LLM
    logger.info("Generating search queries with LLM")
    search_queries = generate_search_queries(context, analysis_points)
    
    # Step 2: Execute searches and collect papers with date filter
    logger.info(f"Executing {len(search_queries)} search queries with {date_filter} filter")
    papers = execute_paper_searches(search_queries, date_filter=date_filter)
    
    # Step 3: Update state with retrieved papers
    # If we already have papers, append new ones without duplicates
    existing_papers = state.get("retrieved_papers", [])
    
    # Combine existing and new papers, avoiding duplicates
    combined_papers = existing_papers.copy()
    for paper in papers:
        if not any(p.get("arxiv_id") == paper.get("arxiv_id") for p in combined_papers):
            combined_papers.append(paper)
    
    state["retrieved_papers"] = combined_papers
    logger.info(f"Retrieved {len(papers)} new papers with {date_filter} filter, total: {len(combined_papers)}")
    
    return state

@traceable(run_type="chain")
def evaluate_papers(state: GraphState) -> GraphState:
    """Node for evaluating paper relevance using parallel processing"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    context = state["context"]
    retrieved_papers = state.get("retrieved_papers", [])
    
    if not retrieved_papers:
        logger.warning("No papers to evaluate")
        state["paper_evaluations"] = []
        state["avg_score"] = 0.0
        state["need_refinement"] = True
        return state
    
    # Log each paper to be evaluated
    logger.info(f"Starting evaluation of {len(retrieved_papers)} papers with detailed criteria")
    for i, paper in enumerate(retrieved_papers):
        if isinstance(paper, dict) and "title" in paper:
            logger.info(f"Paper {i+1}: {paper['title']} (ID: {paper.get('arxiv_id', 'Unknown')})")
        else:
            logger.warning(f"Paper {i+1} has invalid format: {paper}")
    
    # Process papers in parallel with a thread pool
    all_evaluations = []
    max_workers = min(5, len(retrieved_papers))  # Reduce to 5 workers to avoid rate limits
    
    # Create a list to track which papers were processed
    processed_papers = []
    
    try:
        # Create a dictionary to map papers to their futures
        paper_futures = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all papers for evaluation
            for paper in retrieved_papers:
                if isinstance(paper, dict) and "title" in paper and "arxiv_id" in paper:
                    future = executor.submit(evaluate_single_paper, paper, context)
                    paper_futures[future] = paper
                    logger.info(f"Submitted paper for evaluation: {paper['title']} (ID: {paper['arxiv_id']})")
                else:
                    logger.warning(f"Skipping invalid paper: {paper}")
            
            # Wait for all futures to complete
            logger.info(f"Waiting for {len(paper_futures)} paper evaluations to complete...")
            
            # Use as_completed to process results as they finish
            for future in as_completed(paper_futures):
                paper = paper_futures[future]
                try:
                    result = future.result(timeout=180)  # 3 minute timeout
                    
                    # Ensure the result has the arxiv_id
                    if "arxiv_id" not in result and "arxiv_id" in paper:
                        result["arxiv_id"] = paper["arxiv_id"]
                        
                    all_evaluations.append(result)
                    processed_papers.append(paper.get('arxiv_id', 'Unknown'))
                    logger.info(f"Successfully evaluated paper: {paper.get('title', 'Unknown')} (ID: {paper.get('arxiv_id', 'Unknown')})")
                except Exception as e:
                    logger.error(f"Thread execution failed for paper {paper.get('title', 'Unknown')}: {str(e)}")
                    # Add a minimal evaluation to ensure we have something for each paper
                    all_evaluations.append({
                        "title": paper.get("title", "Unknown"),
                        "arxiv_id": paper.get("arxiv_id", "Unknown"),
                        "criteria_scores": {
                            "research_gap_alignment": 5,
                            "methodological_relevance": 5,
                            "theoretical_contribution": 5,
                            "practical_application": 5,
                            "innovation": 5
                        },
                        "overall_score": 0.5,
                        "relevance_points": ["Evaluation failed but paper might be relevant"],
                        "application_suggestions": ["Consider reviewing this paper manually"],
                        "strengths": ["Could not be automatically evaluated"],
                        "limitations": ["Automatic evaluation failed"]
                    })
        
        # Log completion of all evaluations
        logger.info(f"All paper evaluations completed. Processed {len(processed_papers)} papers.")
        
    except Exception as e:
        logger.error(f"Error in ThreadPoolExecutor: {str(e)}")
    
    # Log summary of processed papers
    logger.info(f"Processed {len(processed_papers)} out of {len(retrieved_papers)} papers")
    logger.info(f"Collected {len(all_evaluations)} evaluations")
    
    # Check if any papers were missed
    missing_papers = []
    for paper in retrieved_papers:
        if isinstance(paper, dict) and "arxiv_id" in paper:
            if paper["arxiv_id"] not in processed_papers:
                missing_papers.append((paper["arxiv_id"], paper.get("title", "Unknown")))
    
    if missing_papers:
        logger.warning(f"Missing evaluations for {len(missing_papers)} papers: {missing_papers}")
        # Process missing papers sequentially as a fallback
        for paper in retrieved_papers:
            if isinstance(paper, dict) and "arxiv_id" in paper and paper["arxiv_id"] in [p[0] for p in missing_papers]:
                try:
                    logger.info(f"Processing missed paper sequentially: {paper['title']} (ID: {paper['arxiv_id']})")
                    result = evaluate_single_paper(paper, context)
                    
                    # Ensure the result has the arxiv_id
                    if "arxiv_id" not in result:
                        result["arxiv_id"] = paper["arxiv_id"]
                        
                    all_evaluations.append(result)
                    logger.info(f"Successfully evaluated missed paper: {paper['title']} (ID: {paper['arxiv_id']})")
                except Exception as e:
                    logger.error(f"Sequential evaluation failed for paper {paper['title']}: {str(e)}")
    
    # Process results
    try:
        # Update state
        state["paper_evaluations"] = all_evaluations
        
        # Calculate average score
        scores = [eval_data.get("overall_score", 0.0) for eval_data in all_evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Make sure avg_score is a float
        avg_score = float(avg_score)
        
        # Update state with average score
        state["avg_score"] = avg_score
        
        # Log average score
        logger.info(f"Average paper relevance score: {avg_score:.2f}")
        
        # Determine if refinement is needed
        need_refinement = avg_score < 0.6
        state["need_refinement"] = need_refinement
        
        if need_refinement:
            logger.info("Low average score, refinement may be needed")
        else:
            logger.info("Satisfactory average score, no refinement needed")
        
        return state
    except Exception as e:
        logger.error(f"Error processing evaluation results: {str(e)}")
        # Set default values in case of error
        state["paper_evaluations"] = all_evaluations
        state["avg_score"] = 0.5  # Default to middle score
        state["need_refinement"] = True
        return state

@traceable(run_type="chain")
def format_and_save_results(state: GraphState) -> GraphState:
    """Format and save the final results"""
    # Create output directory if it doesn't exist
    output_dir = Path("scripts/data/paper_retrieval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data from state
    retrieved_papers = state.get("retrieved_papers", [])
    paper_evaluations = state.get("paper_evaluations", [])
    
    # Create a mapping of paper evaluations by arxiv_id
    evaluations_by_id = {eval_data.get("arxiv_id"): eval_data for eval_data in paper_evaluations if "arxiv_id" in eval_data}
    
    # Filter papers by score > 0.6 and limit to top 5
    filtered_papers = []
    paper_scores = []
    
    for paper in retrieved_papers:
        if isinstance(paper, dict) and "arxiv_id" in paper:
            arxiv_id = paper["arxiv_id"]
            if arxiv_id in evaluations_by_id:
                eval_data = evaluations_by_id[arxiv_id]
                score = eval_data.get("overall_score", 0.0)
                if score > 0.6:
                    paper_scores.append((paper, score))
    
    # Sort by score (highest first) and take top 5
    paper_scores.sort(key=lambda x: x[1], reverse=True)
    filtered_papers = [paper for paper, _ in paper_scores[:5]]
    
    logger.info(f"Filtered to {len(filtered_papers)} papers with score > 0.6 (top 5)")
    
    # Create results dictionary with filtered papers
    results = {
        "retrieved_papers": filtered_papers,
        "paper_evaluations": evaluations_by_id,
        "avg_score": state.get("avg_score", 0.0),
        "search_iteration": state.get("search_iteration", 0),
        "date_filter": state.get("date_filter", "last_month")
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"paper_retrieval_results_{timestamp}.json"
    
    # Save to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file} with {len(filtered_papers)} top papers (score > 0.6, limited to top 5)")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
    
    # Generate research application summary
    summary_file = save_research_application_summary(results, output_dir)
    
    # Update state with formatted results
    state["formatted_results"] = {
        "results_file": str(output_file),
        "summary_file": str(summary_file) if summary_file else None,
        "paper_count": len(filtered_papers),
        "evaluation_count": len(paper_evaluations),
        "avg_score": state.get("avg_score", 0.0)
    }
    
    # Set next to END to signal completion
    state["next"] = "END"
    
    return state

@traceable(run_type="chain")
def create_context_from_analysis(analysis_points):
    """Create a context string from analysis points"""
    context_parts = []
    
    if "research_gaps" in analysis_points and analysis_points["research_gaps"]:
        context_parts.append("Research Gaps:")
        for i, gap in enumerate(analysis_points["research_gaps"], 1):
            context_parts.append(f"{i}. {gap}")
        context_parts.append("")
    
    if "key_research_areas" in analysis_points and analysis_points["key_research_areas"]:
        context_parts.append("Key Research Areas:")
        for i, area in enumerate(analysis_points["key_research_areas"], 1):
            context_parts.append(f"{i}. {area}")
        context_parts.append("")
    
    if "critical_analysis" in analysis_points and analysis_points["critical_analysis"]:
        context_parts.append("Critical Analysis Points:")
        for i, point in enumerate(analysis_points["critical_analysis"], 1):
            context_parts.append(f"{i}. {point}")
        context_parts.append("")
    
    if "keywords" in analysis_points and analysis_points["keywords"]:
        context_parts.append("Keywords: " + ", ".join(analysis_points["keywords"]))
    
    context = "\n".join(context_parts)
    logger.info(f"Created context with {len(context_parts)} sections")
    return context

@traceable(run_type="chain")
def generate_search_queries(context, analysis_points):
    """Generate search queries directly from keywords in analysis points"""
    # Extract keywords from analysis points
    keywords = analysis_points.get("keywords", [])
    
    if not keywords or len(keywords) < 2:
        logger.warning("Not enough keywords found in analysis points, using fallback method")
        # Fall back to LLM-based query generation
        return generate_search_queries_with_llm(context, analysis_points)
    
    # Log the keywords
    logger.info(f"Using {len(keywords)} keywords from analysis results")
    for i, keyword in enumerate(keywords, 1):
        logger.info(f"Keyword {i}: {keyword}")
    
    # Create search queries from keyword combinations
    search_queries = []
    
    # Clean and format keywords
    clean_keywords = []
    for keyword in keywords:
        clean_keyword = keyword.strip().replace("'", "").replace('"', '')
        if clean_keyword:
            clean_keywords.append(clean_keyword)
    
    # Generate all possible pairs of keywords
    from itertools import combinations
    keyword_pairs = list(combinations(clean_keywords, 2))
    
    # Limit to a reasonable number of combinations
    max_combinations = min(8, len(keyword_pairs))
    selected_pairs = keyword_pairs[:max_combinations]
    
    # Create combined queries with AND
    for kw1, kw2 in selected_pairs:
        search_queries.append(f'"{kw1}" AND "{kw2}"')
    
    # If we have fewer than 3 queries, add some individual keywords as well
    if len(search_queries) < 3 and clean_keywords:
        for keyword in clean_keywords[:3]:
            if len(search_queries) < 5:  # Limit to 5 total queries
                search_queries.append(f'"{keyword}"')
    
    # Ensure we have at least some queries
    if not search_queries:
        logger.warning("Failed to create queries from keywords, using fallback method")
        return generate_search_queries_with_llm(context, analysis_points)
    
    # Log the generated queries
    logger.info(f"Generated {len(search_queries)} search queries from keyword combinations")
    for i, query in enumerate(search_queries, 1):
        logger.info(f"Query {i}: {query}")
    
    return search_queries

@traceable(run_type="chain")
def generate_search_queries_with_llm(context, analysis_points):
    """Generate optimized search queries for ArXiv using LLM (fallback method)"""
    # Create prompt template for query generation
    query_prompt = ChatPromptTemplate.from_template("""
    You are a research assistant helping to find relevant academic papers on ArXiv.
    
    Based on the following research context, generate 3-5 specific search queries that will help find the most relevant papers.
    
    Research Context:
    {context}
    
    Create search queries that:
    1. Are specific enough to return relevant results
    2. Use proper ArXiv search syntax
    3. Include key terms and concepts from the research gaps
    4. Are diverse to cover different aspects of the research
    
    For each query:
    - Use quotes for exact phrases
    - Use AND/OR operators appropriately
    - Include relevant keywords
    - Focus on recent developments
    
    Return a list of search queries as a JSON array of strings.
    """)
    
    # Create LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Create query generation chain
    query_chain = query_prompt | llm | StrOutputParser() | extract_json
    
    # Execute chain
    try:
        result = query_chain.invoke({"context": context})
        
        # Handle different return formats
        if isinstance(result, list):
            queries = result
        elif isinstance(result, dict) and "queries" in result:
            queries = result["queries"]
        else:
            # Default queries from keywords if LLM output is invalid
            logger.warning("Invalid LLM output format for queries, using fallback")
            queries = [f'"{keyword}"' for keyword in analysis_points.get("keywords", [])[:5]]
        
        # Ensure we have at least some queries
        if not queries:
            # Fallback to basic queries from research gaps
            logger.warning("No queries generated, using fallback from research gaps")
            queries = [gap.split(".")[0] for gap in analysis_points.get("research_gaps", [])[:3]]
        
        # Log the generated queries
        logger.info(f"Generated {len(queries)} search queries with LLM")
        for i, query in enumerate(queries, 1):
            logger.info(f"Query {i}: {query}")
            
        return queries
        
    except Exception as e:
        logger.error(f"Error generating search queries with LLM: {str(e)}")
        # Fallback to basic queries from keywords
        fallback_queries = [f'"{keyword}"' for keyword in analysis_points.get("keywords", [])[:5]]
        logger.info(f"Using {len(fallback_queries)} fallback queries")
        return fallback_queries

@traceable(run_type="chain")
def check_refinement_needed(state: GraphState) -> tuple[GraphState, Literal["continue", "refine", "end"]]:
    """Check if we need to refine the search by expanding the date window"""
    search_iteration = state.get("search_iteration", 0)
    avg_score = state.get("avg_score", 0.0)
    no_date_limit = state.get("no_date_limit", False)
    
    if no_date_limit:
        logger.info("No date limit mode - skipping refinement")
        return state, "end"
    
    if avg_score < 0.6:
        if search_iteration >= 6:
            logger.info("Reached maximum search iterations (6 months). Ending search.")
            return state, "end"
        else:
            # Expand search window by one month
            search_iteration += 1
            months_to_search = search_iteration
            date_filter = f"last_{months_to_search}_months"
            
            logger.info(f"Low average score ({avg_score:.2f}). Expanding search window to {months_to_search} months (iteration {search_iteration})")
            
            # Update state with new search parameters
            new_state = dict(state)  # Create a new state to avoid modifying the original
            new_state["search_iteration"] = search_iteration
            new_state["date_filter"] = date_filter
            return new_state, "refine"
    else:
        logger.info(f"Satisfactory average score ({avg_score:.2f}). No refinement needed.")
        return state, "end"

@traceable(run_type="chain")
def execute_paper_searches(search_queries, max_results_per_query=10, date_filter="last_month"):
    """Execute ArXiv searches for each query and collect papers"""
    all_papers = []
    seen_arxiv_ids = set()
    
    # Get current date for filtering
    current_date = datetime.now()
    
    # Calculate date range based on filter
    if date_filter.endswith("_months"):
        # Extract number of months from filter string (e.g., "last_3_months")
        try:
            months = int(date_filter.split("_")[1])
            start_date = current_date - timedelta(days=30 * months)
            logger.info(f"Searching papers from the last {months} months (since {start_date.strftime('%Y-%m-%d')})")
        except (IndexError, ValueError):
            start_date = current_date - timedelta(days=30)
            logger.info("Invalid month format, defaulting to last month")
    elif date_filter == "last_month":
        start_date = current_date - timedelta(days=30)
        logger.info(f"Searching papers from the last month (since {start_date.strftime('%Y-%m-%d')})")
    elif date_filter == "last_week":
        start_date = current_date - timedelta(days=7)
        logger.info(f"Searching papers from the last week (since {start_date.strftime('%Y-%m-%d')})")
    elif date_filter == "last_year":
        start_date = current_date - timedelta(days=365)
        logger.info(f"Searching papers from the last year (since {start_date.strftime('%Y-%m-%d')})")
    else:
        start_date = current_date - timedelta(days=30)
        logger.warning(f"Unrecognized date filter '{date_filter}', defaulting to last month")
    
    # Create ArXiv client
    client = Client()
    
    for query in search_queries:
        try:
            # Format the query to include date filtering
            date_filter_str = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000 TO {current_date.strftime('%Y%m%d')}235959]"
            filtered_query = f"{query} AND {date_filter_str}"
            
            logger.info(f"Searching ArXiv for: {filtered_query}")
            
            # Create search object with date-filtered query
            # Use the basic approach without sort parameters
            search = Search(
                query=filtered_query,
                max_results=max_results_per_query
            )
            
            # Execute search
            results = list(client.results(search))
            logger.info(f"Found {len(results)} papers for query: {query}")
            
            # Process results
            for result in results:
                # Skip if we've already seen this paper
                if result.entry_id in seen_arxiv_ids:
                    continue
                
                # Extract ArXiv ID from entry_id
                # Format is typically http://arxiv.org/abs/2101.12345v1
                arxiv_id_match = re.search(r'arxiv.org/abs/([^/]+)$', result.entry_id)
                arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else "unknown"
                
                # Add to seen IDs
                seen_arxiv_ids.add(result.entry_id)
                
                # Create paper object
                paper = {
                    "title": result.title,
                    "arxiv_id": arxiv_id,
                    "abstract": result.summary,
                    "authors": [author.name for author in result.authors],
                    "published_date": result.published.isoformat() if result.published else None,
                    "updated_date": result.updated.isoformat() if result.updated else None,
                    "categories": result.categories,
                    "entry_id": result.entry_id,
                    "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}"
                }
                
                all_papers.append(paper)
                
            logger.info(f"Processed {len(results)} papers for query: {query}")
            
        except Exception as e:
            logger.error(f"Error searching ArXiv for query '{query}': {str(e)}")
    
    # Remove any remaining duplicates by arxiv_id
    unique_papers = []
    seen_ids = set()
    
    for paper in all_papers:
        if paper["arxiv_id"] not in seen_ids:
            seen_ids.add(paper["arxiv_id"])
            unique_papers.append(paper)
    
    logger.info(f"Retrieved {len(unique_papers)} unique papers from ArXiv")
    return unique_papers

@traceable(run_type="chain")
def retrieve_papers(state: GraphState) -> GraphState:
    """Node for retrieving papers using the paper search agent"""
    # Simply call the paper search agent
    return paper_search_agent(state)

@traceable(run_type="chain")
def save_research_application_summary(results, output_dir):
    """Save a concise summary of how each paper can help with the research"""
    # Create a simplified summary structure
    summary = {
        "summary_date": datetime.now().strftime("%Y-%m-%d"),
        "papers": []
    }
    
    # Extract relevant papers and their application suggestions
    retrieved_papers = results.get("retrieved_papers", [])
    paper_evaluations = results.get("paper_evaluations", {})
    
    # Create a lookup dictionary for paper evaluations by arxiv_id
    evaluations_by_id = {}
    
    # First, create a mapping from title to arxiv_id using retrieved_papers
    title_to_arxiv = {}
    for paper in retrieved_papers:
        if isinstance(paper, dict) and "title" in paper and "arxiv_id" in paper:
            title_to_arxiv[paper["title"]] = paper["arxiv_id"]
    
    # Now create evaluations_by_id using both the arxiv_id in eval_data and the title mapping
    for title, eval_data in paper_evaluations.items():
        if isinstance(eval_data, dict):
            # If eval_data already has arxiv_id, use it
            if "arxiv_id" in eval_data:
                arxiv_id = eval_data["arxiv_id"]
                evaluations_by_id[arxiv_id] = eval_data
            # Otherwise, try to get arxiv_id from title_to_arxiv mapping
            elif title in title_to_arxiv:
                arxiv_id = title_to_arxiv[title]
                # Add arxiv_id to eval_data for future reference
                eval_data["arxiv_id"] = arxiv_id
                evaluations_by_id[arxiv_id] = eval_data
    
    logger.info(f"Created evaluation lookup with {len(evaluations_by_id)} entries by arxiv_id")
    
    for paper in retrieved_papers:
        print('Paper Title', paper['title'], '(ID:', paper.get('arxiv_id', 'Unknown'), ')')
        if not isinstance(paper, dict) or "arxiv_id" not in paper:
            continue
            
        title = paper.get("title", "Unknown Title")
        arxiv_id = paper.get("arxiv_id", "Unknown ID")
        
        # Skip papers without evaluations - check with arxiv_id
        if arxiv_id not in evaluations_by_id:
            logger.info(f"Skipping paper '{title}' - no evaluation data (ID: {arxiv_id})")
            continue
            
        # Get evaluation data using arxiv_id
        eval_data = evaluations_by_id[arxiv_id]
        overall_score = eval_data.get("overall_score", eval_data.get("relevance_score", 0.0))
        
        # Skip papers with relevance score < 0.6 (increased from 0.5)
        if overall_score < 0.6:
            logger.info(f"Skipping paper '{title}' - low relevance score: {overall_score}")
            continue
        
        # Create paper summary
        paper_summary = {
            "title": title,
            "arxiv_id": arxiv_id,
            "abstract": paper.get("abstract", "No abstract available")[:500] + "..." if len(paper.get("abstract", "")) > 500 else paper.get("abstract", "No abstract available"),
            "overall_score": overall_score,
            "criteria_scores": eval_data.get("criteria_scores", {}),
            "research_applications": eval_data.get("application_suggestions", []),
            "strengths": eval_data.get("strengths", []),
            "limitations": eval_data.get("limitations", [])
        }
        
        # Add top relevance points
        if "relevance_points" in eval_data and eval_data["relevance_points"]:
            paper_summary["key_relevance_points"] = eval_data["relevance_points"][:3]
        
        summary["papers"].append(paper_summary)
    
    # Sort papers by overall score (highest first)
    summary["papers"] = sorted(summary["papers"], key=lambda p: p.get("overall_score", 0), reverse=True)
    
    # Limit to top 5 papers
    summary["papers"] = summary["papers"][:5]
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"research_application_summary_{timestamp}.json"
    
    # Save to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Research application summary saved with {len(summary['papers'])} top relevant papers (score > 0.6, limited to top 5)")
        return output_file
    except Exception as e:
        error_msg = f"Failed to save research application summary: {str(e)}"
        logger.error(error_msg)
        return None

@traceable(run_type="chain")
def evaluate_single_paper(paper, context):
    """Evaluate a single paper with more detailed scoring criteria"""
    try:
        # Create evaluation prompt template with more detailed scoring criteria
        evaluation_prompt = ChatPromptTemplate.from_template("""
        You are a research evaluator assessing the relevance of an academic paper.
        
        Analyze this paper considering the following research context:
        
        {context}
        
        Paper to evaluate:
        {paper}
        
        Evaluate on these specific criteria:
        1. Research Gap Alignment (0-10): How well it addresses the specific research gaps
        2. Methodological Relevance (0-10): How useful its methods are to the research
        3. Theoretical Contribution (0-10): How it contributes to theoretical understanding
        4. Practical Application (0-10): How applicable its findings are to practical problems
        5. Innovation (0-10): How novel or innovative the approach is
        
        Also provide:
        - At least 3 specific points about how the paper is relevant
        - 2-3 specific ways this paper can help with the existing research
        
        Provide a JSON response with:
        - "title": Title of the paper
        - "arxiv_id": ArXiv ID of the paper
        - "criteria_scores": Object with the 5 criteria scores
        - "overall_score": The average of all criteria scores, normalized to 0.0-1.0
        - "relevance_points": List of at least 3 points about how the paper is relevant
        - "application_suggestions": List of 2-3 specific ways this paper can help with the existing research
        - "strengths": List of 2-3 strengths of this paper
        - "limitations": List of 1-2 limitations or weaknesses of this paper
        """)
        
        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Create evaluation chain
        evaluation_chain = evaluation_prompt | llm | StrOutputParser() | extract_json
        
        # Format paper for prompt
        paper_text = f"Title: {paper['title']}\nArXiv ID: {paper['arxiv_id']}\nAbstract: {paper.get('abstract', 'No abstract available')}"
        
        # Execute evaluation
        result = evaluation_chain.invoke({
            "context": context,
            "paper": paper_text
        })
        
        # Ensure we have all expected fields
        if "arxiv_id" not in result:
            result["arxiv_id"] = paper.get("arxiv_id", "Unknown")
            
        if "criteria_scores" not in result:
            result["criteria_scores"] = {
                "research_gap_alignment": 0,
                "methodological_relevance": 0,
                "theoretical_contribution": 0,
                "practical_application": 0,
                "innovation": 0
            }
        
        if "overall_score" not in result:
            # Calculate overall score from criteria if missing
            scores = result["criteria_scores"].values()
            result["overall_score"] = sum(scores) / (len(scores) * 10) if scores else 0.0
            
        if "strengths" not in result:
            result["strengths"] = []
            
        if "limitations" not in result:
            result["limitations"] = []
        
        logger.info(f"Evaluated paper: {paper['title']} (ID: {paper['arxiv_id']}) with score {result['overall_score']:.2f}")
        return result
    except Exception as e:
        logger.error(f"Error evaluating paper {paper.get('title', 'Unknown')}: {str(e)}")
        # Return a minimal valid structure
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": paper.get("arxiv_id", "Unknown"),
            "criteria_scores": {
                "research_gap_alignment": 0,
                "methodological_relevance": 0,
                "theoretical_contribution": 0,
                "practical_application": 0,
                "innovation": 0
            },
            "overall_score": 0.0,
            "relevance_points": ["Evaluation failed"],
            "application_suggestions": ["Evaluation failed"],
            "strengths": [],
            "limitations": []
        }

@traceable(run_type="chain")
def refinement_router(state: GraphState) -> GraphState:
    """Route to the next node based on refinement needs"""
    search_iteration = state.get("search_iteration", 0)
    avg_score = state.get("avg_score", 0.0)
    
    # Hard limit on iterations to prevent infinite loops
    MAX_ITERATIONS = 12
    
    # Debug logging
    logger.info(f"Refinement check: iteration={search_iteration}, avg_score={avg_score:.2f}")
    
    if avg_score < 0.6 and search_iteration < MAX_ITERATIONS:
        # Expand search window by one month
        search_iteration += 1
        months_to_search = search_iteration
        date_filter = f"last_{months_to_search}_months"
        
        logger.info(f"Low average score ({avg_score:.2f}). Expanding search window to {months_to_search} months (iteration {search_iteration}/{MAX_ITERATIONS})")
        
        # Create a new state to avoid modifying the original
        new_state = dict(state)
        new_state["search_iteration"] = search_iteration
        new_state["date_filter"] = date_filter
        new_state["next"] = "retrieve"
        
        # Debug log the updated state
        logger.info(f"Updated state: iteration={new_state['search_iteration']}, date_filter={new_state['date_filter']}, next={new_state['next']}")
        
        return new_state
    else:
        if search_iteration >= MAX_ITERATIONS:
            logger.info(f"Reached maximum search iterations ({MAX_ITERATIONS} months). Ending search.")
        else:
            logger.info(f"Satisfactory average score ({avg_score:.2f}). No refinement needed.")
        
        # Create a new state to avoid modifying the original
        new_state = dict(state)
        new_state["next"] = "format_and_save"
        
        # Debug log the updated state
        logger.info(f"Updated state: next={new_state['next']}")
        
        return new_state

@traceable(run_type="chain")
def main():
    """Main execution function"""
    try:
        # Load analysis results
        analysis_points = load_analysis_results()
        
        # Create workflow graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("initialize", initialize)
        workflow.add_node("retrieve", paper_search_agent)
        workflow.add_node("evaluate", evaluate_papers)
        workflow.add_node("check_refinement", refinement_router)
        workflow.add_node("format_and_save", format_and_save_results)
        
        # Add edges
        workflow.add_edge("initialize", "retrieve")
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_edge("evaluate", "check_refinement")
        
        # Add conditional edges based on the 'next' field in state
        workflow.add_conditional_edges(
            "check_refinement",
            lambda state: state["next"]
        )
        
        workflow.add_edge("format_and_save", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Compile graph
        app = workflow.compile()
        
        # Initialize state
        initial_state = {
            "analysis_points": analysis_points,
            "context": "",
            "retrieved_papers": [],
            "paper_evaluations": [],
            "refined_queries": [],
            "avg_score": 0.0,
            "need_refinement": False,
            "date_filter": "last_month",
            "search_iteration": 0,
            "next": None  # Initialize the 'next' field
        }
        
        # Run workflow manually to avoid recursion issues
        logger.info("Starting paper retrieval workflow with manual execution")
        
        # Manual execution approach
        current_state = initial_state
        max_iterations = 10
        
        # First step: initialize
        logger.info("Step 1: Initialize")
        current_state = initialize(current_state)
        
        # Second step: retrieve papers (first iteration)
        logger.info("Step 2: Retrieve papers (initial search)")
        current_state = paper_search_agent(current_state)
        
        # Third step: evaluate papers
        logger.info("Step 3: Evaluate papers")
        current_state = evaluate_papers(current_state)
        
        # Refinement loop
        for iteration in range(max_iterations):
            logger.info(f"Refinement iteration {iteration+1}/{max_iterations}")
            
            # Check if refinement is needed
            refinement_state = refinement_router(current_state)
            next_step = refinement_state.get("next")
            
            logger.info(f"Refinement decision: {next_step}")
            
            if next_step == "format_and_save" or next_step == "END":
                logger.info("No further refinement needed, proceeding to final step")
                current_state = refinement_state
                break
                
            if next_step == "retrieve":
                logger.info(f"Refinement needed, expanding search (iteration {iteration+1})")
                # Update current state with refinement changes
                current_state = refinement_state
                
                # Retrieve more papers with expanded date range
                logger.info(f"Retrieving papers with expanded date filter: {current_state.get('date_filter')}")
                current_state = paper_search_agent(current_state)
                
                # Evaluate the new papers
                logger.info("Evaluating new papers")
                current_state = evaluate_papers(current_state)
                
                # Continue to next iteration of refinement
                continue
                
            # If we get here, something unexpected happened
            logger.warning(f"Unexpected next step: {next_step}, stopping refinement")
            break
            
        # Final step: format and save results
        logger.info("Final step: Format and save results")
        final_state = format_and_save_results(current_state)
        
        # Debug log the final state
        logger.info(f"Final state keys: {list(final_state.keys())}")
        
        # Wait for all traces to be uploaded
        wait_for_all_tracers()
        
        return final_state.get("formatted_results", {})
        
    except Exception as e:
        logger.exception("Critical error occurred:")
        raise RuntimeError(f"Execution failed: {str(e)}") from e

if __name__ == "__main__":
    main()