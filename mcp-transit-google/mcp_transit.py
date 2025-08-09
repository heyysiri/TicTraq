import asyncio
import os
from datetime import datetime, timezone
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INVALID_PARAMS
from pydantic import BaseModel, Field

import httpx


# --- Load environment variables ---
load_dotenv()

AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert AUTH_TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert GOOGLE_MAPS_API_KEY is not None, "Please set GOOGLE_MAPS_API_KEY in your .env file"


# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="mcp-google-transit-client",
                scopes=["*"],
                expires_at=None,
            )
        return None


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


class TransitStep(BaseModel):
    vehicle: str
    line_name: str | None = None
    headsign: str | None = None
    num_stops: int | None = None
    departure_stop: str | None = None
    arrival_stop: str | None = None
    departure_time_local: str | None = None
    arrival_time_local: str | None = None


class TransitPlan(BaseModel):
    summary: str
    departure_time_local: str | None = None
    arrival_time_local: str | None = None
    duration_text: str | None = None
    fare_text: str | None = None
    steps: list[TransitStep]


async def call_google_directions(
    *,
    origin: str,
    destination: str,
    transit_mode: str | None,
    departure_time_epoch: int | None,
    arrival_time_epoch: int | None,
    language: str | None,
    region: str | None,
) -> dict:
    params: dict[str, str] = {
        "origin": origin,
        "destination": destination,
        "mode": "transit",
        "key": GOOGLE_MAPS_API_KEY,
    }
    if transit_mode:
        params["transit_mode"] = transit_mode
    if departure_time_epoch is not None:
        params["departure_time"] = str(departure_time_epoch)
    if arrival_time_epoch is not None:
        params["arrival_time"] = str(arrival_time_epoch)
    if language:
        params["language"] = language
    if region:
        params["region"] = region

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get("https://maps.googleapis.com/maps/api/directions/json", params=params)
        data = resp.json()
        if data.get("status") != "OK":
            message = data.get("error_message") or data.get("status") or "Unknown error"
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Google Directions error: {message}"))
        return data


def _format_epoch_to_iso_local(value: int) -> str:
    # Google returns seconds since epoch UTC; represent in ISO8601 local time string for readability
    dt = datetime.fromtimestamp(value)
    return dt.strftime("%Y-%m-%d %H:%M")


def extract_transit_plans(directions: dict) -> list[TransitPlan]:
    plans: list[TransitPlan] = []
    for route in directions.get("routes", []):
        for leg in route.get("legs", []):
            steps: list[TransitStep] = []

            dep_time = leg.get("departure_time", {}).get("value")
            arr_time = leg.get("arrival_time", {}).get("value")
            dep_text = leg.get("departure_time", {}).get("text")
            arr_text = leg.get("arrival_time", {}).get("text")
            duration_text = leg.get("duration", {}).get("text")

            fare_text = None
            if route.get("fare") and route["fare"].get("text"):
                fare_text = route["fare"]["text"]

            summary_parts: list[str] = []

            for step in leg.get("steps", []):
                travel_mode = step.get("travel_mode")
                if travel_mode == "WALKING":
                    summary_parts.append("Walk")
                    continue
                if travel_mode != "TRANSIT":
                    continue

                transit = step.get("transit_details", {})
                line = transit.get("line", {})
                vehicle = line.get("vehicle", {}).get("type", "TRANSIT")
                line_name = line.get("name") or line.get("short_name")
                headsign = transit.get("headsign")
                num_stops = transit.get("num_stops")
                departure_stop = transit.get("departure_stop", {}).get("name")
                arrival_stop = transit.get("arrival_stop", {}).get("name")
                dep_local = None
                arr_local = None
                if transit.get("departure_time", {}).get("value"):
                    dep_local = _format_epoch_to_iso_local(transit["departure_time"]["value"])
                if transit.get("arrival_time", {}).get("value"):
                    arr_local = _format_epoch_to_iso_local(transit["arrival_time"]["value"])

                steps.append(
                    TransitStep(
                        vehicle=vehicle,
                        line_name=line_name,
                        headsign=headsign,
                        num_stops=num_stops,
                        departure_stop=departure_stop,
                        arrival_stop=arrival_stop,
                        departure_time_local=dep_local,
                        arrival_time_local=arr_local,
                    )
                )

                tag = f"{vehicle}: {line_name or ''} → {headsign or ''}".strip()
                summary_parts.append(tag)

            summary = " › ".join(p for p in summary_parts if p)
            plans.append(
                TransitPlan(
                    summary=summary or "Transit plan",
                    departure_time_local=dep_text or (dep_time and _format_epoch_to_iso_local(dep_time)) or None,
                    arrival_time_local=arr_text or (arr_time and _format_epoch_to_iso_local(arr_time)) or None,
                    duration_text=duration_text,
                    fare_text=fare_text,
                    steps=steps,
                )
            )
    return plans


# --- MCP Server Setup ---
mcp = FastMCP(
    "Google Transit MCP Server",
    auth=SimpleBearerAuthProvider(AUTH_TOKEN),
)


@mcp.tool
async def validate() -> str:
    # Mirror the pattern from the bearer-token server; if MY_NUMBER is set, return it.
    return MY_NUMBER or "google-transit:ok"


TRANSIT_DESCRIPTION = RichToolDescription(
    description=(
        "Find public transit routes, next departures, and travel durations using Google Directions API (transit)."
    ),
    use_when=(
        "Use for queries like 'next bus from A to B', 'metro to X from Y', or 'when should I leave to catch the next train'."
    ),
)


@mcp.tool(description=TRANSIT_DESCRIPTION.model_dump_json())
async def transit_route(
    user_question: Annotated[
        str,
        Field(
            description=(
                "Freeform question, e.g., 'when is the next bus to JBS from Narayanguda' or 'metro to Hitec from Ameerpet'."
            )
        ),
    ],
    origin: Annotated[
        str | None,
        Field(description="Origin address or 'lat,lng'. Optional if it can be inferred from the question."),
    ] = None,
    destination: Annotated[
        str | None,
        Field(description="Destination address or 'lat,lng'. Optional if it can be inferred from the question."),
    ] = None,
    mode: Annotated[
        Literal["bus", "subway", "train", "tram", "rail", "metro"] | None,
        Field(description="Filter to a specific transit mode (bus, subway/metro, train, tram, rail)."),
    ] = None,
    depart_at_iso: Annotated[
        str | None,
        Field(description="Desired departure time in ISO8601 (local). If omitted, uses 'now'. Mutually exclusive with arrive_by_iso."),
    ] = None,
    arrive_by_iso: Annotated[
        str | None,
        Field(description="Desired arrival deadline in ISO8601 (local). Mutually exclusive with depart_at_iso."),
    ] = None,
    language: Annotated[
        str | None,
        Field(description="Response language, e.g., 'en', 'hi'."),
    ] = "en",
    region: Annotated[
        str | None,
        Field(description="Region bias, e.g., 'IN' for India."),
    ] = "IN",
    max_plans: Annotated[int, Field(description="Maximum number of plans to return")] = 3,
) -> dict:
    """
    Returns a concise set of upcoming public transit route options between origin and destination.
    """
    # Simple heuristic extraction if origin/destination missing
    q = user_question.lower()
    parsed_origin = origin
    parsed_destination = destination

    # Extract patterns like 'to X from Y' or 'from Y to X'
    if (parsed_origin is None or parsed_destination is None) and (" to " in q and " from " in q):
        try:
            before_to, after_to = q.split(" to ", 1)
            dest_candidate = after_to.split(" from ", 1)[0].strip()
            after_from = after_to.split(" from ", 1)[1].strip()
            # remaining after_from is origin
            origin_candidate = after_from.strip()
            parsed_destination = parsed_destination or dest_candidate
            parsed_origin = parsed_origin or origin_candidate
        except Exception:
            pass

    if parsed_origin is None or parsed_destination is None:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=(
                    "origin and destination are required. Provide them explicitly or in the question as 'to <dest> from <origin>'."
                ),
            )
        )

    # Mode normalization for Google 'transit_mode'
    transit_mode = None
    if mode:
        if mode == "metro":
            transit_mode = "subway"
        else:
            transit_mode = mode

    # Time handling
    if depart_at_iso and arrive_by_iso:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Specify only one of depart_at_iso or arrive_by_iso"))

    departure_time_epoch: int | None = None
    arrival_time_epoch: int | None = None
    if arrive_by_iso:
        # Interpret as local time string; let Python parse safely
        try:
            # naive -> local; then convert to epoch seconds
            naive = datetime.fromisoformat(arrive_by_iso)
            arrival_time_epoch = int(naive.replace(tzinfo=None).timestamp())
        except Exception as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Invalid arrive_by_iso: {e}"))
    elif depart_at_iso:
        try:
            naive = datetime.fromisoformat(depart_at_iso)
            departure_time_epoch = int(naive.replace(tzinfo=None).timestamp())
        except Exception as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Invalid depart_at_iso: {e}"))
    else:
        # default: now
        departure_time_epoch = int(datetime.now(timezone.utc).timestamp())

    directions = await call_google_directions(
        origin=parsed_origin,
        destination=parsed_destination,
        transit_mode=transit_mode,
        departure_time_epoch=departure_time_epoch,
        arrival_time_epoch=arrival_time_epoch,
        language=language,
        region=region,
    )

    plans = extract_transit_plans(directions)
    if not plans:
        return {"message": "No transit plans found"}

    # Trim
    plans = plans[: max(1, max_plans)]

    # Convert to serializable dict
    return {
        "origin": parsed_origin,
        "destination": parsed_destination,
        "plans": [p.model_dump() for p in plans],
    }


async def main():
    print("🚦 Starting Google Transit MCP server on http://0.0.0.0:8087")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8087)


if __name__ == "__main__":
    asyncio.run(main())


