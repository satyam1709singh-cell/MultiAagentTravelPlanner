from typing import TypedDict


class TripState(TypedDict, total=False):
    # --- user inputs ---
    destination: str
    origin: str
    start_date: str
    end_date: str
    num_travelers: int
    budget: str          # e.g. "1500 USD total" - kept as free text, LLM interprets
    interests: str        # free text, e.g. "food, hiking, museums"

    # --- pipeline outputs (filled in by each agent) ---
    research_notes: str
    flights_info: str
    hotels_info: str
    itinerary: str
    from ddgs import DDGS

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo and return formatted results."""
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}.\n"
            f"Title: {r.get('title')}\n"
            f"URL: {r.get('href')}\n"
            f"Snippet: {r.get('body')}\n"
        )
    return "\n".join(formatted)
import ollama

# TripState and web_search already defined above in this notebook

MODEL = "llama3.2"


def _ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Single-turn call to the local Ollama model (no tool calling needed here -
    we do the web_search ourselves and just ask the model to summarize/reason)."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 1. Research agent
# ---------------------------------------------------------------------------
def research_node(state: TripState) -> TripState:
    print("\n[Agent] Researching destination...")
    query = f"travel guide {state['destination']} attractions best time to visit safety tips"
    raw = web_search(query, max_results=5)

    summary = _ask_llm(
        system_prompt=(
            "You are a destination research assistant. Summarize the search "
            "results into concise, useful notes for a trip planner: top "
            "attractions, best time to visit, safety/visa notes, and local "
            "tips. Include source URLs at the end."
        ),
        user_prompt=(
            f"Destination: {state['destination']}\n"
            f"Traveler interests: {state.get('interests', 'general sightseeing')}\n\n"
            f"Search results:\n{raw}"
        ),
    )
    return {"research_notes": summary}


# ---------------------------------------------------------------------------
# 2. Flights agent
# ---------------------------------------------------------------------------
def flights_node(state: TripState) -> TripState:
    print("[Agent] Searching flights...")
    query = (
        f"flights from {state['origin']} to {state['destination']} "
        f"{state['start_date']} to {state['end_date']} price"
    )
    raw = web_search(query, max_results=5)

    summary = _ask_llm(
        system_prompt=(
            "You are a flight search assistant. Based on the (imperfect) web "
            "search snippets, give the user a realistic sense of flight "
            "options and approximate price ranges from origin to destination "
            "for the given dates. Be explicit that prices are estimates from "
            "web snippets, not live fares, and include source URLs."
        ),
        user_prompt=(
            f"Origin: {state['origin']}\n"
            f"Destination: {state['destination']}\n"
            f"Dates: {state['start_date']} to {state['end_date']}\n"
            f"Travelers: {state.get('num_travelers', 1)}\n"
            f"Budget: {state.get('budget', 'not specified')}\n\n"
            f"Search results:\n{raw}"
        ),
    )
    return {"flights_info": summary}


# ---------------------------------------------------------------------------
# 3. Hotels agent
# ---------------------------------------------------------------------------
def hotels_node(state: TripState) -> TripState:
    print("[Agent] Searching hotels...")
    query = (
        f"best hotels in {state['destination']} for {state.get('num_travelers', 1)} "
        f"travelers {state.get('budget', '')}"
    )
    raw = web_search(query, max_results=5)

    summary = _ask_llm(
        system_prompt=(
            "You are a hotel search assistant. Based on the web search "
            "snippets, suggest 2-4 accommodation options across different "
            "price points that fit the traveler's budget and interests. "
            "Note this is based on web snippets, not live availability, and "
            "include source URLs."
        ),
        user_prompt=(
            f"Destination: {state['destination']}\n"
            f"Travelers: {state.get('num_travelers', 1)}\n"
            f"Budget: {state.get('budget', 'not specified')}\n"
            f"Interests: {state.get('interests', 'general')}\n\n"
            f"Search results:\n{raw}"
        ),
    )
    return {"hotels_info": summary}


# ---------------------------------------------------------------------------
# 4. Itinerary agent (final synthesis)
# ---------------------------------------------------------------------------
def itinerary_node(state: TripState) -> TripState:
    print("[Agent] Building itinerary...")
    summary = _ask_llm(
        system_prompt=(
            "You are a travel itinerary planner. Using the research notes, "
            "flight info, and hotel info provided, produce a clear day-by-day "
            "itinerary as plain text. For each day include: morning/afternoon/"
            "evening activities, a suggested hotel area if relevant, and a "
            "rough running budget note. Keep the tone practical and specific. "
            "End with a short 'Budget Summary' section estimating total cost "
            "vs the traveler's stated budget, flagging if it looks tight or "
            "comfortable."
        ),
        user_prompt=(
            f"Trip details:\n"
            f"- Destination: {state['destination']}\n"
            f"- Origin: {state['origin']}\n"
            f"- Dates: {state['start_date']} to {state['end_date']}\n"
            f"- Travelers: {state.get('num_travelers', 1)}\n"
            f"- Budget: {state.get('budget', 'not specified')}\n"
            f"- Interests: {state.get('interests', 'general sightseeing')}\n\n"
            f"Research notes:\n{state.get('research_notes', '')}\n\n"
            f"Flight info:\n{state.get('flights_info', '')}\n\n"
            f"Hotel info:\n{state.get('hotels_info', '')}\n"
        ),
    )
    return {"itinerary": summary}
from langgraph.graph import StateGraph, END

# TripState and node functions already defined above in this notebook


def build_graph():
    """Sequential pipeline: research -> flights -> hotels -> itinerary"""
    graph = StateGraph(TripState)

    graph.add_node("research", research_node)
    graph.add_node("flights", flights_node)
    graph.add_node("hotels", hotels_node)
    graph.add_node("itinerary", itinerary_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "flights")
    graph.add_edge("flights", "hotels")
    graph.add_edge("hotels", "itinerary")
    graph.add_edge("itinerary", END)

    return graph.compile()


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{label}{suffix}: ").strip()
    return val or default


trip_input = {
    "destination": prompt("Destination"),
    "origin": prompt("Origin city"),
    "start_date": prompt("Start date (e.g. 2026-09-10)"),
    "end_date": prompt("End date (e.g. 2026-09-17)"),
    "num_travelers": prompt("Number of travelers", "1"),
    "budget": prompt("Budget (e.g. '2000 USD total')"),
    "interests": prompt("Interests (e.g. food, hiking, museums)"),
}

print("\nTrip input captured:")
trip_input

final_state = run_trip_planner(trip_input)
print("===================== ITINERARY =====================\n")
print(final_state.get("itinerary", "No itinerary generated."))
print("--- Research notes ---\n")
print(final_state.get("research_notes", ""))
print("--- Flight info ---\n")
print(final_state.get("flights_info", ""))

print("--- Hotel info ---\n")
print(final_state.get("hotels_info", ""))