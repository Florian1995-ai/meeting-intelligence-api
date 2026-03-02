"""
Meeting Intelligence API + Live Transcription Web App

Endpoints:
  GET  /                        — Meetings web app (live transcription + chat)
  GET  /briefing?person={name}  — Full person briefing (meetings, connections)
  POST /query                   — Natural language question -> graph answer
  WS   /ws/transcribe           -- Deepgram WebSocket proxy (mic audio -> transcript)
  POST /transcript/save         -- Save a meeting transcript
  GET  /transcripts             — List saved transcripts
  GET  /health                  — Neo4j connectivity check
  POST /playbook/suggest        — Match signals to Hormozi playbooks, suggest questions

Deploy via Coolify (Docker) on the same VPS as Neo4j for low-latency queries.

Environment variables required:
    NEO4J_URI          bolt://168.231.64.2:7688
    NEO4J_USER         neo4j
    NEO4J_PASSWORD     <password>
    OPENAI_API_KEY     <key>
    DEEPGRAM_API_KEY   <key>
    OPENROUTER_API_KEY <key>   (optional - for DeepSeek playbook matching)
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4
import re as regex_module
import httpx

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("meeting-intelligence")

app = FastAPI(title="Meeting Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Transcript storage directory
TRANSCRIPT_DIR = Path("/data/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hormozi Coaching Playbooks (loaded at startup)
# ---------------------------------------------------------------------------

PLAYBOOKS_FILE = Path(__file__).parent / "playbooks.json"
PLAYBOOKS = []
TRIGGER_INDEX = []  # [{signal_words: set, playbook_idx: int, signal_text: str}]


def _normalize(text: str) -> set:
    """Normalize text to a set of lowercase words for matching."""
    return set(regex_module.sub(r'[^a-z0-9\s]', '', text.lower()).split())


def load_playbooks():
    """Load Hormozi playbook index at startup."""
    global PLAYBOOKS, TRIGGER_INDEX
    if not PLAYBOOKS_FILE.exists():
        logger.warning(f"Playbooks file not found: {PLAYBOOKS_FILE}")
        return
    try:
        PLAYBOOKS = json.loads(PLAYBOOKS_FILE.read_text(encoding="utf-8"))
        TRIGGER_INDEX = []
        for idx, pb in enumerate(PLAYBOOKS):
            for sig in pb.get("trigger_signals", []):
                TRIGGER_INDEX.append({
                    "signal_words": _normalize(sig),
                    "signal_text": sig,
                    "playbook_idx": idx,
                })
        logger.info(f"Loaded {len(PLAYBOOKS)} Hormozi playbooks, {len(TRIGGER_INDEX)} trigger signals")
    except Exception as e:
        logger.error(f"Failed to load playbooks: {e}")


load_playbooks()


def match_playbooks_to_signals(detected_signals: dict, top_n: int = 5) -> list:
    """Match detected coaching signals against Hormozi playbook trigger signals.

    Uses word overlap scoring: for each detected signal item, compute overlap
    with each trigger signal. Return top-N matched playbooks with their
    recommended questions.
    """
    if not TRIGGER_INDEX:
        return []

    signal_texts = []
    for category in ("challenges", "goals", "personal"):
        for item in detected_signals.get(category, []):
            text = item if isinstance(item, str) else item.get("item", "")
            if text:
                signal_texts.append(text)

    if not signal_texts:
        return []

    playbook_scores = {}
    playbook_matched = {}

    for trigger in TRIGGER_INDEX:
        trigger_words = trigger["signal_words"]
        if len(trigger_words) < 2:
            continue

        for sig_text in signal_texts:
            sig_words = _normalize(sig_text)
            if len(sig_words) < 2:
                continue
            overlap = len(trigger_words & sig_words)
            score = overlap / min(len(trigger_words), len(sig_words))

            pb_idx = trigger["playbook_idx"]
            if score > playbook_scores.get(pb_idx, 0):
                playbook_scores[pb_idx] = score
                playbook_matched[pb_idx] = trigger["signal_text"]

    MIN_SCORE = 0.25
    ranked = sorted(
        [(idx, score) for idx, score in playbook_scores.items() if score >= MIN_SCORE],
        key=lambda x: -x[1]
    )[:top_n]

    results = []
    for pb_idx, score in ranked:
        pb = PLAYBOOKS[pb_idx]
        results.append({
            "playbook_id": pb.get("id", ""),
            "title": pb.get("title", ""),
            "categories": pb.get("problem_categories", []),
            "match_score": round(score, 2),
            "matched_trigger": playbook_matched.get(pb_idx, ""),
            "questions": pb.get("recommended_questions", {}),
            "turning_points": pb.get("turning_points", []),
            "top_diagnostic": [q["q"] for q in pb.get("question_sequence", [])[:3]],
        })

    return results


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

QUERY_SYSTEM_PROMPT_BASE = """You are a query parser for a meeting knowledge graph. Given a natural language question, determine the best query type and extract parameters.

The graph contains:
- Person nodes with: canonical_name, category, email, total_meetings, first_seen, last_seen
- Meeting nodes with: human_name, date, meeting_type
- ATTENDED relationships (Person -> Meeting)
- CO_ATTENDED relationships (Person <-> Person)

The graph also has detailed entity nodes extracted from meeting transcripts. These include topics discussed, personal details mentioned, business ideas, action items, etc. You can search these using Cypher full-text or pattern matching.

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
   Use for: complex graph queries, personal details, topics discussed, or anything that doesn't fit other types.
   IMPORTANT: Only generate READ queries (MATCH/RETURN). Never generate CREATE/DELETE/SET queries.
   For personal detail questions (kids, hobbies, business details), search entity nodes:
   MATCH (n) WHERE toLower(n.entity_name) CONTAINS toLower('keyword') OR toLower(n.description) CONTAINS toLower('keyword') RETURN n.entity_name, n.description LIMIT 20

6. {"type": "stats"}
   Use for: "How many people/meetings are in the graph?"
"""

QUERY_TRANSCRIPT_ADDENDUM = """
7. {"type": "transcript_search", "keywords": ["keyword1", "keyword2"]}
   Use for: Questions about the CURRENT CALL's live transcript only, e.g. "What did they just say about pricing?", "Summarize the last few minutes".
"""


def parse_query_with_llm(question: str, context_person: Optional[str] = None,
                         live_context: Optional[str] = None) -> Dict:
    """Use gpt-4o-mini to parse a natural language question into a structured query."""
    from openai import OpenAI
    client = OpenAI()

    # Only include transcript_search option when live context is actually present
    system_prompt = QUERY_SYSTEM_PROMPT_BASE
    if live_context:
        system_prompt += QUERY_TRANSCRIPT_ADDENDUM

    user_msg = question
    if context_person:
        user_msg = f"[Context: Currently in a meeting with {context_person}]\n{question}"
    if live_context:
        user_msg += f"\n\n[Live transcript from current call (last 5 minutes):\n{live_context}\n]"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=500,
    )

    return json.loads(response.choices[0].message.content)


def search_live_transcript(question: str, live_context: str, keywords: list) -> str:
    """Use gpt-4o-mini to answer a question about the live transcript."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are analyzing a live meeting transcript. Answer the user's question based solely on the transcript provided. Be concise and specific. If the answer isn't in the transcript, say so."},
            {"role": "user", "content": f"Transcript:\n{live_context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=500,
    )

    return response.choices[0].message.content


def execute_parsed_query(driver, parsed: Dict, live_context: Optional[str] = None,
                         question: Optional[str] = None) -> Dict:
    """Execute a parsed query against Neo4j."""
    query_type = parsed.get("type", "briefing")

    if query_type == "transcript_search":
        if not live_context:
            return {"type": "error", "data": {"error": "No live transcript available"}}
        keywords = parsed.get("keywords", [])
        answer = search_live_transcript(question or "", live_context, keywords)
        return {"type": "transcript_search", "data": {"answer": answer, "keywords": keywords}}
    elif query_type == "briefing":
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

    if query_type == "transcript_search":
        return data.get("answer", "No answer from transcript search.")

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
    live_context: Optional[str] = None  # Rolling transcript from current call


class SignalExtractionRequest(BaseModel):
    utterances: List[str]  # Last 60-120 seconds of final utterances


class PlaybookSuggestRequest(BaseModel):
    signals: Dict[str, list]  # {challenges: [...], goals: [...], personal: [...]}
    utterances: Optional[List[str]] = None  # Recent utterances for contextual refinement


@app.get("/briefing")
def briefing_endpoint(person: str = Query(..., description="Person name to look up")):
    """Full person briefing from Neo4j."""
    logger.info(f"Briefing request for: {person}")
    try:
        driver = create_driver()
        try:
            result = person_briefing(driver, person)
            co = find_co_attendees(driver, person, limit=10)
            result["co_attendees"] = co
            return result
        finally:
            driver.close()
    except Exception as e:
        logger.error(f"Briefing error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


@app.post("/query")
def query_endpoint(req: QueryRequest):
    """Natural language question -> graph answer."""
    logger.info(f"Query: {req.question} (context: {req.context_person}, live_context: {bool(req.live_context)})")

    try:
        parsed = parse_query_with_llm(req.question, req.context_person, req.live_context)
        logger.info(f"Parsed: {parsed}")

        driver = create_driver()
        try:
            result = execute_parsed_query(driver, parsed, live_context=req.live_context,
                                          question=req.question)
            formatted = format_response(result)
            return {
                "question": req.question,
                "parsed": parsed,
                "result": result.get("data"),
                "formatted": formatted,
            }
        finally:
            driver.close()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return JSONResponse(
            content={"error": str(e), "formatted": f"Error: {str(e)}"},
            status_code=500,
        )


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


# ---------------------------------------------------------------------------
# Real-Time Signal Extraction (Coaching Insights)
# ---------------------------------------------------------------------------

SIGNAL_EXTRACTION_PROMPT = """You are analyzing a live coaching/meeting conversation. Extract coaching-relevant signals from the utterances below.

For each signal found, provide:
- item: A concise description (max 15 words)
- evidence_quote: The exact phrase from the transcript that triggered this signal
- speaker: The speaker label (e.g. "Speaker 0") or null if unclear
- confidence: high/medium/low

Categorize into exactly three categories:
1. **challenges** — Pain points, obstacles, constraints, frustrations, problems they're facing
2. **goals** — Desired outcomes, targets, timelines, aspirations, what they want to achieve
3. **personal** — Values, motivations, identity, family details, fears, proud moments, hobbies, personal context

Only extract signals that are clearly present. Do NOT invent or infer signals that aren't explicitly stated.
If no signals are found in a category, return an empty array for it.

Respond with JSON only:
{
  "challenges": [{"item": "...", "evidence_quote": "...", "speaker": "...", "confidence": "..."}],
  "goals": [{"item": "...", "evidence_quote": "...", "speaker": "...", "confidence": "..."}],
  "personal": [{"item": "...", "evidence_quote": "...", "speaker": "...", "confidence": "..."}]
}"""


@app.post("/signals/extract-live")
def extract_live_signals(req: SignalExtractionRequest):
    """Extract coaching signals (challenges/goals/personal) from recent utterances."""
    if not req.utterances:
        return {"challenges": [], "goals": [], "personal": []}

    logger.info(f"Signal extraction: {len(req.utterances)} utterances")

    try:
        from openai import OpenAI
        client = OpenAI()

        transcript_text = "\n".join(req.utterances)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SIGNAL_EXTRACTION_PROMPT},
                {"role": "user", "content": transcript_text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=800,
        )

        result = json.loads(response.choices[0].message.content)
        # Ensure all three categories exist
        for key in ("challenges", "goals", "personal"):
            if key not in result:
                result[key] = []

        logger.info(f"Signals found: {len(result['challenges'])}C {len(result['goals'])}G {len(result['personal'])}P")
        return result

    except Exception as e:
        logger.error(f"Signal extraction error: {e}")
        return JSONResponse(
            content={"error": str(e), "challenges": [], "goals": [], "personal": []},
            status_code=500,
        )



# ---------------------------------------------------------------------------
# Hormozi Playbook Matching & Question Suggestions
# ---------------------------------------------------------------------------

PLAYBOOK_SUGGEST_PROMPT = """You are a coaching intelligence assistant. Based on the detected coaching signals and recent conversation, suggest the most relevant questions to ask right now.

You have access to Alex Hormozi's coaching playbook patterns. Given the signals detected in the conversation, pick the TOP 3-5 most impactful questions the coach should ask next.

For each question, explain briefly WHY it's relevant to what was just discussed.

DETECTED SIGNALS:
{signals_text}

RECENT CONVERSATION:
{utterances_text}

MATCHED PLAYBOOK PATTERNS:
{matched_patterns}

Respond with JSON only:
{{
  "suggested_questions": [
    {{
      "question": "The exact question to ask",
      "why": "Why this question is relevant right now (1 sentence)",
      "category": "discovery|deepening|commitment|diagnostic",
      "source_playbook": "playbook title or 'general'"
    }}
  ]
}}"""


@app.post("/playbook/suggest")
def suggest_playbook_questions(req: PlaybookSuggestRequest):
    """Match detected signals against Hormozi playbooks and suggest questions."""
    logger.info("Playbook suggestion request")

    try:
        # Step 1: Fast keyword matching against trigger signals
        matched = match_playbooks_to_signals(req.signals)

        if not matched and not req.utterances:
            return {"suggested_questions": [], "matched_playbooks": []}

        # Step 2: If we have utterances and an OpenRouter key, use DeepSeek for contextual refinement
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        if openrouter_key and req.utterances:
            try:
                signals_text = ""
                for cat in ("challenges", "goals", "personal"):
                    items = req.signals.get(cat, [])
                    if items:
                        signals_text += f"\n{cat.upper()}:\n"
                        for item in items:
                            text = item if isinstance(item, str) else item.get("item", "")
                            signals_text += f"  - {text}\n"

                utterances_text = "\n".join(req.utterances[-15:])

                matched_patterns = ""
                for m in matched[:3]:
                    matched_patterns += f"\nPlaybook: {m['title']} (categories: {', '.join(m['categories'])})"
                    matched_patterns += f"\n  Matched signal: {m['matched_trigger']}"
                    for q in m["top_diagnostic"][:2]:
                        matched_patterns += f"\n  Diagnostic Q: {q}"
                    qs = m.get("questions", {})
                    for q in qs.get("discovery", [])[:1]:
                        matched_patterns += f"\n  Discovery Q: {q}"
                    for q in qs.get("deepening", [])[:1]:
                        matched_patterns += f"\n  Deepening Q: {q}"

                prompt = PLAYBOOK_SUGGEST_PROMPT.format(
                    signals_text=signals_text or "No specific signals detected yet",
                    utterances_text=utterances_text,
                    matched_patterns=matched_patterns or "No specific playbook matches",
                )

                resp = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek/deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 800,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                llm_result = resp.json()
                content_str = llm_result["choices"][0]["message"]["content"]
                suggestions = json.loads(content_str)

                logger.info(f"DeepSeek suggestions: {len(suggestions.get('suggested_questions', []))} questions")
                return {
                    "suggested_questions": suggestions.get("suggested_questions", []),
                    "matched_playbooks": [
                        {"id": m["playbook_id"], "title": m["title"], "score": m["match_score"], "categories": m["categories"]}
                        for m in matched
                    ],
                    "source": "deepseek",
                }

            except Exception as e:
                logger.warning(f"DeepSeek refinement failed, falling back to keyword matching: {e}")

        # Step 3: Keyword-only fallback (no LLM)
        questions = []
        seen = set()
        for m in matched:
            qs = m.get("questions", {})
            for cat in ("discovery", "deepening", "commitment"):
                for q in qs.get(cat, []):
                    if q not in seen:
                        seen.add(q)
                        questions.append({
                            "question": q,
                            "why": f"Matched pattern: {m['matched_trigger']}",
                            "category": cat,
                            "source_playbook": m["title"],
                        })
            for tp in m.get("turning_points", [])[:1]:
                q = tp.get("q", "")
                if q and q not in seen:
                    seen.add(q)
                    questions.append({
                        "question": q,
                        "why": tp.get("why", "Key turning-point question"),
                        "category": "diagnostic",
                        "source_playbook": m["title"],
                    })

        return {
            "suggested_questions": questions[:5],
            "matched_playbooks": [
                {"id": m["playbook_id"], "title": m["title"], "score": m["match_score"], "categories": m["categories"]}
                for m in matched
            ],
            "source": "keyword_match",
        }

    except Exception as e:
        logger.error(f"Playbook suggestion error: {e}")
        return JSONResponse(
            content={"error": str(e), "suggested_questions": []},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Last Meeting Lookup
# ---------------------------------------------------------------------------

@app.get("/last-meeting")
def last_meeting_endpoint(person: str = Query(..., description="Person name to look up")):
    """Get the most recent meeting with a person — summary, action items, open loops."""
    logger.info(f"Last meeting request for: {person}")

    try:
        driver = create_driver()
        try:
            with driver.session() as session:
                # Find person's canonical name
                result = session.run("""
                    MATCH (p:Person)
                    WHERE toLower(p.canonical_name) CONTAINS toLower($name)
                    RETURN p.canonical_name AS canonical_name
                    LIMIT 1
                """, name=person)
                record = result.single()
                if not record:
                    return {"error": f"Person not found: {person}"}

                canonical = record["canonical_name"]

                # Get most recent meeting
                result = session.run("""
                    MATCH (p:Person {canonical_name: $name})-[r:ATTENDED]->(m:Meeting)
                    RETURN m.human_name AS title, m.date AS date,
                           m.meeting_type AS type, r.role AS role
                    ORDER BY m.date DESC
                    LIMIT 1
                """, name=canonical)
                meeting_record = result.single()

                if not meeting_record:
                    return {"error": f"No meetings found with {canonical}"}

                meeting_data = {
                    "person": canonical,
                    "date": meeting_record["date"],
                    "title": meeting_record["title"],
                    "type": meeting_record["type"],
                    "role": meeting_record["role"],
                }

                # Try to find related entity nodes (topics, action items) from LightRAG
                result = session.run("""
                    MATCH (p:Person {canonical_name: $name})-[:ATTENDED]->(m:Meeting)
                    WHERE m.date = $date
                    OPTIONAL MATCH (m)-[r]->(e)
                    WHERE type(r) <> 'ATTENDED'
                    RETURN type(r) AS rel_type, e.entity_name AS entity,
                           e.description AS description
                    LIMIT 20
                """, name=canonical, date=meeting_record["date"])
                entities = [dict(r) for r in result]

                if entities:
                    meeting_data["entities"] = entities

            # Also check saved transcripts on disk
            saved_summary = None
            for f in sorted(TRANSCRIPT_DIR.glob("*.json"), reverse=True):
                try:
                    data = json.loads(f.read_text())
                    if data.get("person") and canonical.lower() in data["person"].lower():
                        saved_summary = {
                            "saved_at": data.get("saved_at"),
                            "line_count": data.get("line_count", 0),
                        }
                        break
                except Exception:
                    continue

            if saved_summary:
                meeting_data["saved_transcript"] = saved_summary

            return meeting_data

        finally:
            driver.close()

    except Exception as e:
        logger.error(f"Last meeting error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# WebSocket Proxy — Deepgram Transcription
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    """Proxy mic audio from browser to Deepgram, return transcript JSON.

    The Deepgram API key stays server-side — browser never sees it.
    Browser sends raw audio blobs, receives transcript JSON objects.
    """
    await ws.accept()
    logger.info("WebSocket client connected for transcription")

    import websockets

    dg_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not dg_key:
        await ws.send_json({"error": "DEEPGRAM_API_KEY not configured on server"})
        await ws.close()
        return

    # Get sample rate from browser (AudioContext may use 44100, 48000, etc.)
    sample_rate = ws.query_params.get("sample_rate", "48000")
    logger.info(f"Browser audio sample rate: {sample_rate}")

    dg_params = (
        f"model=nova-2-meeting&encoding=linear16&sample_rate={sample_rate}&channels=1"
        "&diarize=true&interim_results=true"
        "&smart_format=true&punctuate=true&language=en&utterance_end_ms=1000"
    )
    dg_url = f"wss://api.deepgram.com/v1/listen?{dg_params}"

    try:
        async with websockets.connect(
            dg_url, additional_headers={"Authorization": f"Token {dg_key}"}
        ) as dg_ws:
            logger.info("Connected to Deepgram upstream")

            # Tell browser Deepgram is ready
            await ws.send_text(json.dumps({"type": "status", "message": "Connected to Deepgram"}))

            async def forward_audio():
                """Browser -> Deepgram: forward audio chunks."""
                chunk_count = 0
                total_bytes = 0
                try:
                    while True:
                        data = await ws.receive_bytes()
                        await dg_ws.send(data)
                        chunk_count += 1
                        total_bytes += len(data)
                        if chunk_count == 1:
                            logger.info(f"First audio chunk received: {len(data)} bytes")
                        if chunk_count % 40 == 0:  # ~every 10 seconds at 250ms intervals
                            logger.info(f"Audio: {chunk_count} chunks, {total_bytes} bytes total")
                except WebSocketDisconnect:
                    logger.info(f"Browser disconnected after {chunk_count} chunks ({total_bytes} bytes)")
                except Exception as e:
                    logger.warning(f"Audio forward error after {chunk_count} chunks: {type(e).__name__}: {e}")

            async def forward_transcript():
                """Deepgram -> Browser: forward transcript JSON."""
                msg_count = 0
                try:
                    async for message in dg_ws:
                        if isinstance(message, bytes):
                            message = message.decode("utf-8")
                        await ws.send_text(message)
                        msg_count += 1
                        if msg_count == 1:
                            logger.info(f"First Deepgram response forwarded")
                except Exception as e:
                    logger.warning(f"Transcript forward error after {msg_count} msgs: {e}")

            # Run both directions concurrently
            audio_task = asyncio.create_task(forward_audio())
            transcript_task = asyncio.create_task(forward_transcript())

            done, pending = await asyncio.wait(
                [audio_task, transcript_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.error(f"Deepgram proxy error: {e}")
        try:
            await ws.send_json({"error": f"Deepgram connection failed: {str(e)}"})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Transcription WebSocket closed")


# ---------------------------------------------------------------------------
# Transcript Storage
# ---------------------------------------------------------------------------

class TranscriptSaveRequest(BaseModel):
    person: Optional[str] = None
    date: Optional[str] = None
    lines: List[Dict[str, Any]]  # [{speaker, text, timestamp}, ...]


@app.post("/transcript/save")
def save_transcript(req: TranscriptSaveRequest):
    """Save a meeting transcript to persistent storage."""
    transcript_id = uuid4().hex[:12]
    ts = req.date or datetime.now().strftime("%Y-%m-%d_%H-%M")
    person_label = (req.person or "unknown").replace(" ", "_")

    filename = f"{ts}_{person_label}_{transcript_id}.json"
    filepath = TRANSCRIPT_DIR / filename

    data = {
        "id": transcript_id,
        "person": req.person,
        "date": ts,
        "saved_at": datetime.now().isoformat(),
        "line_count": len(req.lines),
        "lines": req.lines,
    }

    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved transcript: {filename} ({len(req.lines)} lines)")

    return {"id": transcript_id, "filename": filename, "lines": len(req.lines)}


@app.get("/transcripts")
def list_transcripts():
    """List all saved transcripts."""
    transcripts = []
    for f in sorted(TRANSCRIPT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            transcripts.append({
                "id": data.get("id"),
                "person": data.get("person"),
                "date": data.get("date"),
                "saved_at": data.get("saved_at"),
                "lines": data.get("line_count", 0),
                "filename": f.name,
            })
        except Exception:
            continue
    return {"transcripts": transcripts}


# ---------------------------------------------------------------------------
# Static File Serving — Meetings Web App
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
async def serve_app():
    """Serve the meetings web app."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return JSONResponse({"error": "Frontend not deployed"}, status_code=404)

# Mount static assets (CSS, JS, images) if the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
