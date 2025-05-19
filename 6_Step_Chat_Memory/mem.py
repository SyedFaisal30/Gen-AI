from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))


def extract_triples(text):
    """
    A dummy/simple triple extractor - replace this with a better NLP method (like spaCy, OpenAI GPT prompt, or others)
    For demo, we expect simple pattern: "X is Y" or "X works at Y"
    Returns list of (subject, predicate, object) triples.
    """
    triples = []
    text = text.lower()
    if " is " in text:
        parts = text.split(" is ")
        if len(parts) == 2:
            triples.append((parts[0].strip(), "is", parts[1].strip()))
    elif " works at " in text:
        parts = text.split(" works at ")
        if len(parts) == 2:
            triples.append((parts[0].strip(), "works_at", parts[1].strip()))
    else:
        # fallback - just one node
        triples.append((text, "related_to", "unknown"))
    return triples


def add_to_neo4j(triples):
    with driver.session() as session:
        for subj, pred, obj in triples:
            session.run(
                """
                MERGE (a:Entity {name: $subj})
                MERGE (b:Entity {name: $obj})
                MERGE (a)-[r:RELATION {type: $pred}]->(b)
                """,
                subj=subj,
                obj=obj,
                pred=pred
            )


def chat_to_graph(user_input):
    triples = extract_triples(user_input)
    add_to_neo4j(triples)
    return f"Graph updated with: {triples}"


if __name__ == "__main__":
    print("Enter text to convert to graph. Type 'exit' or 'quit' to stop.")
    while True:
        inp = input(">> ")
        if inp.lower() in ["exit", "quit"]:
            break
        response = chat_to_graph(inp)
        print(response)
