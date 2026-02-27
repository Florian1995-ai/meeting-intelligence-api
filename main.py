"""
Meeting Intelligence Query API — Standalone FastAPI

Two web endpoints for querying the Neo4j transcript knowledge graph:
1. GET  /briefing?person={name}  — Full person briefing (meetings, connections)
2. POST /query                   — Natural language question -> graph answer
3. GET  /health                  — Neo4j connectivity check

Deploy via Coolify (Docker) on the same VPS as Neo4j for low-latency queries.

Environment variables required:
    NEO4J_URI          bolt://168.231.64.2:7688
    NEO4J_USER         neo4j
    NEO4J_PASSWORD     <password>
    OPENAI_API_KEY     <key>
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("meeting-intelligence")

app = FastAPI(title="Meeting Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Neo4j Query Functions
# ---------------------------------------------------------------------------

def create_driver():
    """Create Neo4j driver from environment variables."""
    from neo4j import GraphDatabase
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def person_briefing(driver, person_name: str) -> Dict:
    """Get comprehensive briefing on a person -- meetings, category, connections."""
    briefing = {}

    with driver.session() as session:
        result = session.run("""
            MATCH (p:Person)
            WHERE toLower(p.canonical_name) CONTAINS toLower($name)
            RETURN p
            LIMIT 1
        """, name=person_name)
        record = result.single()
        if not record:
            return {"error": f"Person not found: {person_name}"}

        person = dict(record["p"])
        briefing["person"] = {
            "name": person.get("canonical_name"),
            "email": person.get("primary_email"),
            "category": person.get("category", "UNCATEGORIZED"),
            "relationship_type": person.get("relationship_type"),
            "total_meetings": person.get("total_meetings"),
            "first_seen": person.get("first_seen"),
            "last_seen": person.get("last_seen"),
        }

        canonical = person.get("canonical_name")

        # Recent meetings
        result = session.run("""
            MATCH (p:Person {canonical_name: $name})-[r:ATTENDED]->(m:Meeting)
            RETURN m.human_name AS meeting, m.date AS date,
                   m.meeting_type AS type, r.role AS role
            ORDER BY m.date DESC
            LIMIT 10
        """, name=canonical)
        briefing["recent_meetings"] = [dict(r) for r in result]

        # Top co-attendees with categories
        result = session.run("""
            MATCH (a:Person {canonical_name: $name})-[:ATTENDED]->(m:Meeting)<-[:ATTENDED]-(b:Person)
            WHERE a <> b
            WITH b, count(DISTINCT m) AS shared
            RETURN b.canonical_name AS name, b.category AS category, shared
            ORDER BY shared DESC
            LIMIT 10
        """, name=canonical)
        briefing["top_connections"] = [dict(r) for r in result]

    return briefing


def find_co_attendees(driver, person_name: str, limit: int = 20) -> List[Dict]:
    """Find people who attended meetings with a given person, ranked by frequency."""
    cypher = """
    MATCH (a:Person)-[:ATTENDED]->(m:Meeting)<-[:ATTENDED]-(b:Person)
    WHERE toLower(a.canonical_name) CONTAINS toLower($name)
      AND a <> b
    WITH b, count(DISTINCT m) AS shared_meetings,
         collect(DISTINCT m.date) AS dates,
         collect(DISTINCT m.human_name) AS meeting_names
    RETURN b.canonical_name AS name,
           b.primary_email AS email,
           b.category AS category,
           b.relationship_type AS relationship,
           shared_meetings,
           dates[0] AS most_recent_date,
           meeting_names[..3] AS sample_meetings,
           b.total_meetings AS total_meetings
    ORDER BY shared_meetings DESC
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(cypher, name=person_name, limit=limit)
        return [dict(r) for r in result]


def find_commonalities(driver, name_a: str, name_b: str) -> Dict:
    """Find what two people have in common (shared meetings, mutual connections)."""
    results = {"between": [name_a, name_b], "shared": {}}

    with driver.session() as session:
        result = session.run("""
            MATCH (a:Person)-[:ATTENDED]->(m:Meeting)<-[:ATTENDED]-(b:Person)
            WHERE toLower(a.canonical_name) CONTAINS toLower($a)
              AND toLower(b.canonical_name) CONTAINS toLower($b)
            RETURN count(DISTINCT m) AS shared_meetings,
                   collect(DISTINCT m.human_name)[..5] AS sample_meetings,
                   collect(DISTINCT m.date)[..5] AS dates
        """, a=name_a, b=name_b)
        record = result.single()
        if record and record["shared_meetings"] > 0:
            results["shared"]["meetings"] = {
                "count": record["shared_meetings"],
                "samples": record["sample_meetings"],
                "dates": record["dates"],
            }

        result = session.run("""
            MATCH (a:Person)-[:ATTENDED]->(m1:Meeting)<-[:ATTENDED]-(mutual:Person)
            WHERE toLower(a.canonical_name) CONTAINS toLower($a) AND a <> mutual
            WITH mutual
            MATCH (b:Person)-[:ATTENDED]->(m2:Meeting)<-[:ATTENDED]-(mutual)
            WHERE toLower(b.canonical_name) CONTAINS toLower($b) AND b <> mutual
            RETURN mutual.canonical_name AS name, mutual.category AS category
            LIMIT 10
        """, a=name_a, b=name_b)
        mutuals = [dict(r) for r in result]
        if mutuals:
            results["shared"]["mutual_connections"] = mutuals

    return results


def find_by_category(driver, category: str, limit: int = 50) -> List[Dict]:
    """Find all people in a specific relationship category."""
    cypher = """
    MATCH (p:Person)
    WHERE toLower(p.category) CONTAINS toLower($cat)
    RETURN p.canonical_name AS name,
           p.category AS category,
           p.primary_email AS email,
           p.total_meetings AS meetings,
           p.first_seen AS first_seen,
           p.last_seen AS last_seen
    ORDER BY p.total_meetings DESC
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(cypher, cat=category, limit=limit)
        return [dict(r) for r in result]


def run_cypher(driver, query: str) -> List[Dict]:
    """Run a raw Cypher query."""
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


def graph_stats(driver) -> Dict[str, Any]:
    """Get complete graph statistics."""
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n) AS label, count(n) AS count
            ORDER BY count DESC
        """)
        nodes = {str(r["label"]): r["count"] for r in result}

        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """)
        rels = {r["rel_type"]: r["count"] for r in result}

    return {"nodes": nodes, "relationships": rels}


# ---------------------------------------------------------------------------
# LLM Query Parsing (gpt-4o-mini)
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT = """You are a query parser for a meeting knowledge graph. Given a natural language question, determine the best query type and extract parameters.

The graph contains:
- Person nodes with: canonical_name, category, email, total_meetings, first_seen, last_seen
- Meeting nodes with: human_name, date, meeting_type
- ATTENDED relationships (Person -> Meeting)
- CO_ATTENDED relationships (Person <-> Person)

Categories include: 1FRIENDS, 2FREE5WEEKCHALLENGEPROSPECTS, 3CURRENTCLIENTS, 4GENERALNETWORKING, 5PASTCLIENTSCHURNED, 111PARTNERS, 111GROWWITHJACK, 14BNICONTACTS, etc.

Respond with JSON only. Choose one of these query types:

1. {"type": "briefing", "person": "<name>"}
   Use for: "Tell me about X", "When did I last meet X?", "What's X's history?"

2. {"type": "co_attendees", "person": "<name>", "limit": 20}
   Use for: "Who are X's connections?", "Who does X meet with?"

3. {"type": "commonalities", "person_a": "<name>", "person_b": "<name>"}
   Use for: "What do X and Y have in common?", "How are X and Y connected?"

4. {"type": "category", "category": "<category>", "limit": 20}
   Use for: "List all current clients", "Show me BNI contacts"

5. {"type": "cypher", "query": "<cypher query>"}
   Use for: complex graph queries that don't fit other types.
   IMPORTANT: Only generate READ queries (MATCH/RETURN). Never generate CREATE/DELETE/SET queries.

6. {"type": "stats"}
   Use for: "How many people/meetings are in the graph?"
"""


def parse_query_with_llm(question: str, context_person: Optional[str] = None) -> Dict:
    """Use gpt-4o-mini to parse a natural language question into a structured query."""
    from openai import OpenAI
    client = OpenAI()

    user_msg = question
    if context_person:
        user_msg = f"[Context: Currently in a meeting with {context_person}]\n{question}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=500,
    )

    return json.loads(response.choices[0].message.content)


def execute_parsed_query(driver, parsed: Dict) -> Dict:
    """Execute a parsed query against Neo4j."""
    query_type = parsed.get("type", "briefing")

    if query_type == "briefing":
        return {"type": "briefing", "data": person_briefing(driver, parsed["person"])}
    elif query_type == "co_attendees":
        limit = parsed.get("limit", 20)
        return {"type": "co_attendees", "data": find_co_attendees(driver, parsed["person"], limit)}
    elif query_type == "commonalities":
        return {"type": "commonalities", "data": find_commonalities(driver, parsed["person_a"], parsed["person_b"])}
    elif query_type == "category":
        limit = parsed.get("limit", 20)
        return {"type": "category", "data": find_by_category(driver, parsed["category"], limit)}
    elif query_type == "cypher":
        cypher = parsed.get("query", "")
        upper = cypher.upper().strip()
        if any(kw in upper for kw in ["CREATE", "DELETE", "SET ", "MERGE", "REMOVE", "DROP"]):
            return {"type": "error", "data": {"error": "Write queries are not allowed"}}
        return {"type": "cypher", "data": run_cypher(driver, cypher)}
    elif query_type == "stats":
        return {"type": "stats", "data": graph_stats(driver)}
    else:
        return {"type": "error", "data": {"error": f"Unknown query type: {query_type}"}}


def format_response(result: Dict) -> str:
    """Format a query result into a human-readable string."""
    query_type = result.get("type")
    data = result.get("data", {})

    if query_type == "error":
        return data.get("error", "Unknown error")

    if query_type == "briefing":
        if "error" in data:
            return data["error"]
        p = data.get("person", {})
        lines = [f"**{p.get('name', '?')}** ({p.get('category', '?')})"]
        lines.append(f"Meetings: {p.get('total_meetings', 0)} | Last seen: {p.get('last_seen', 'N/A')}")
        meetings = data.get("recent_meetings", [])
        if meetings:
            lines.append(f"\nRecent meetings ({len(meetings)}):")
            for m in meetings[:5]:
                lines.append(f"  {m.get('date', '?')} - {m.get('meeting', '?')}")
        connections = data.get("top_connections", [])
        if connections:
            lines.append(f"\nTop connections:")
            for c in connections[:5]:
                lines.append(f"  {c['name']} ({c['shared']} shared) [{c.get('category', '?')}]")
        return "\n".join(lines)

    if query_type == "co_attendees":
        if not data:
            return "No co-attendees found."
        lines = [f"Found {len(data)} co-attendees:"]
        for r in data[:10]:
            lines.append(f"  {r['name']} ({r['shared_meetings']} shared) [{r.get('category', '?')}]")
        return "\n".join(lines)

    if query_type == "commonalities":
        shared = data.get("shared", {})
        if not shared:
            return f"No commonalities found between {data.get('between', ['?', '?'])[0]} and {data.get('between', ['?', '?'])[1]}."
        lines = []
        if "meetings" in shared:
            m = shared["meetings"]
            lines.append(f"Shared meetings: {m['count']}")
            for name in m.get("samples", []):
                lines.append(f"  - {name}")
        if "mutual_connections" in shared:
            lines.append(f"\nMutual connections:")
            for mc in shared["mutual_connections"]:
                lines.append(f"  - {mc['name']} [{mc.get('category', '?')}]")
        return "\n".join(lines) if lines else "No commonalities found."

    if query_type == "category":
        if not data:
            return "No people found in that category."
        lines = [f"Found {len(data)} people:"]
        for r in data[:15]:
            lines.append(f"  {r['name']} ({r.get('meetings', 0)} meetings, last: {r.get('last_seen', '?')})")
        return "\n".join(lines)

    if query_type == "stats":
        lines = ["Graph Statistics:"]
        for label, count in data.get("nodes", {}).items():
            lines.append(f"  {label}: {count}")
        for rel, count in data.get("relationships", {}).items():
            lines.append(f"  {rel}: {count}")
        return "\n".join(lines)

    if query_type == "cypher":
        if not data:
            return "Query returned no results."
        lines = [f"{len(data)} rows:"]
        for r in data[:20]:
            lines.append(f"  {r}")
        return "\n".join(lines)

    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    context_person: Optional[str] = None


@app.get("/briefing")
def briefing_endpoint(person: str = Query(..., description="Person name to look up")):
    """Full person briefing from Neo4j."""
    logger.info(f"Briefing request for: {person}")
    driver = create_driver()
    try:
        result = person_briefing(driver, person)
        co = find_co_attendees(driver, person, limit=10)
        result["co_attendees"] = co
        return result
    finally:
        driver.close()


@app.post("/query")
def query_endpoint(req: QueryRequest):
    """Natural language question -> graph answer."""
    logger.info(f"Query: {req.question} (context: {req.context_person})")

    parsed = parse_query_with_llm(req.question, req.context_person)
    logger.info(f"Parsed: {parsed}")

    driver = create_driver()
    try:
        result = execute_parsed_query(driver, parsed)
        formatted = format_response(result)
        return {
            "question": req.question,
            "parsed": parsed,
            "result": result.get("data"),
            "formatted": formatted,
        }
    finally:
        driver.close()


@app.get("/health")
def health_endpoint():
    """Check Neo4j connectivity."""
    try:
        driver = create_driver()
        stats = graph_stats(driver)
        driver.close()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )
