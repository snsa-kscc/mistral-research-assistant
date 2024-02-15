import os
import json
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

load_dotenv()


def web_scraper(url: str):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        page_txt = soup.get_text(separator=" ", strip=True)
        return page_txt

    except Exception as e:
        print(e)
        return f"Failed to retrieve content from {url}, error: {e}"


ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = 3) -> list:
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


t_llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1,
    base_url="https://api.together.xyz",
    openai_api_key=os.environ["TOGETHER_API_KEY"],
)


summary_template = """{text}

-------------
Using the above text summarize the following question:
> {question}

-------------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats, etc.
"""

summary_prompt = ChatPromptTemplate.from_template(summary_template)


runnable_chain = (
    RunnablePassthrough.assign(text=lambda x: web_scraper(x["url"])[:10_000])
    | summary_prompt
    | t_llm
    | StrOutputParser()
)


duck_chain = (
    RunnablePassthrough.assign(urls=lambda x: web_search(x["question"]))
    | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]])
    | runnable_chain.map()
)


search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to serach online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query1", "query2", "query3"].\n'
            "The queries must be as objective as possible.",
        )
    ]
)


generate_questions_chain = search_prompt | t_llm | StrOutputParser() | json.loads


def process_questions(questions):
    result = []
    for q in questions:
        result.append({"question": q})
    return result


composite_chain = generate_questions_chain | process_questions | duck_chain.map()


writer_system_prompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

research_report_template = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""


research_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", writer_system_prompt),
        ("user", research_report_template),
    ]
)


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


flatten_chain = (
    RunnablePassthrough.assign(
        research_summary=composite_chain | collapse_list_of_lists
    )
    | research_prompt
    | t_llm
    | StrOutputParser()
)
