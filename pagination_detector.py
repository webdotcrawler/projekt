import os
import json
from typing import List, Dict, Tuple, Union
from pydantic import BaseModel, Field
import tiktoken
from dotenv import load_dotenv
from groq import Groq
from assets import PROMPT_PAGINATION, PRICING, LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME

load_dotenv()
import logging

class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list, description="List of pagination URLs, including 'Next' button URL if present")

def calculate_pagination_price(token_counts: Dict[str, int], model: str) -> float:
    """
    Calculate the price for pagination based on token counts and the selected model.
    
    Args:
    token_counts (Dict[str, int]): A dictionary containing 'input_tokens' and 'output_tokens'.
    model (str): The name of the selected model.

    Returns:
    float: The total price for the pagination operation.
    """
    input_tokens = token_counts['input_tokens']
    output_tokens = token_counts['output_tokens']
    
    input_price = input_tokens * PRICING[model]['input']
    output_price = output_tokens * PRICING[model]['output']
    
    return input_price + output_price

def detect_pagination_elements(url: str, indications: str, selected_model: str, markdown_content: str) -> Tuple[Union[PaginationData, Dict, str], Dict, float]:
    """
    Uses AI models to analyze markdown content and extract pagination elements.

    Args:
        url (str): The URL of the webpage being analyzed.
        indications (str): Specific instructions or user indications.
        selected_model (str): The name of the AI model being used.
        markdown_content (str): The markdown content to analyze.

    Returns:
        Tuple[PaginationData, Dict, float]: Parsed pagination data, token counts, and the price for the operation.
    """
    try:
        prompt_pagination = PROMPT_PAGINATION + "\n The url of the page to extract pagination from is: " + url + \
                            "\n\n If the URLs that you find are not complete, combine them intelligently in a way that fits the pattern. " \
                            "**ALWAYS PROVIDE FULL URLs**."
        
        if indications:
            prompt_pagination += "\n\nUser Indications: " + indications + \
                                 "\n\nBelow is the markdown content of the website:\n\n"
        else:
            prompt_pagination += "\n\nNo user indications were provided.\n\nBelow is the markdown content of the website:\n\n"

        # Groq Llama3.1 70b model integration
        if selected_model == "Groq Llama3.1 70b":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            # **Fix: Assign the result to 'completion'**
            completion = client.chat.completions.create(
                model=GROQ_LLAMA_MODEL_FULLNAME,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
                response_format=PaginationData
            )

            # **Fix: Extract the parsed response from 'completion'**
            parsed_response = completion.choices[0].message.parsed

            # Calculate tokens using tiktoken
            encoder = tiktoken.encoding_for_model(selected_model)
            input_token_count = len(encoder.encode(markdown_content))
            output_token_count = len(encoder.encode(json.dumps(parsed_response.dict())))
            token_counts = {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count
            }

            # Calculate the price
            pagination_price = calculate_pagination_price(token_counts, selected_model)

            return parsed_response, token_counts, pagination_price

        # Handle different models, if applicable
        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        logging.error(f"An error occurred in detect_pagination_elements: {e}")
        # Return default values if an error occurs
        return PaginationData(page_urls=[]), {"input_tokens": 0, "output_tokens": 0}, 0.0
