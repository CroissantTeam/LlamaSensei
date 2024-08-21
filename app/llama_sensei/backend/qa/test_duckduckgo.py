import re
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from duckduckgo_search import DDGS

def test_duckduckgo_search(query: str):
    # Initialize the search tool
    # search_tool = DuckDuckGoSearchResults()
    
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search_tool.results(query, max_results = 5)

#     # results = DDGS().text("What method do we use if we want to predict house price in an area?", max_results = 5)
#     # ddg = DDGS()
#     # results = DDGS().answers("What method do we use if we want to predict house price in an area?")
    
#     # Dictionary to store the parsed results
#     # results_dict = {}

#     # # Check if results are in the expected format (a string with specific patterns)
#     # if isinstance(results, str):
#     #     # Parse the results string
#     #     results_list = parse_results_string(results)
        
#     #     # Store the parsed results in a dictionary
#     #     for i, result in enumerate(results_list):
#     #         results_dict[f"Result_{i + 1}"] = {
#     #             "title": result.get('title', 'No Title'),
#     #             "snippet": result.get('snippet', 'No Snippet'),
#     #             "link": result.get('link', 'No Link')
#     #         }
#     # else:
#     #     print("Unexpected format for results:", results)
    
#     # # Return the dictionary containing all results
#     # return results_dict
#     return results

# def parse_results_string(results_str):
#     # Regular expression to match the pattern for snippet, title, and link
#     pattern = r"\[snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(.*?)\]"
    
#     # Use re.findall to extract all matches
#     matches = re.findall(pattern, results_str)
    
#     # Convert matches into a list of dictionaries
#     results = []
#     for match in matches:
#         result_dict = {
#             "snippet": match[0].strip(),
#             "title": match[1].strip(),
#             "link": match[2].strip()
#         }
#         results.append(result_dict)
    
#     return results

# # Example usage
# if __name__ == "__main__":
#     test_query = "What method do we use if we want to predict house price in an area?"
#     results_dict = test_duckduckgo_search(test_query)
#     # results_dict = DDGS().answers("area")
#     # results_dict = DDGS().answers("sun")
    
#     print(results_dict)
# Create an instance
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)

# Define the search query
query = "What method do we use if we want to predict house price in an area?"

# Call the results method
results = search_wrapper.results(query=query, max_results = 5)

print(results[0]['snippet'][100:])
