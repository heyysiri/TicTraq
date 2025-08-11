import asyncio
import os
from datetime import datetime
from typing import Annotated, Literal, Any
from textwrap import dedent
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




def _routes_v2_origin_or_dest(text: str) -> dict[str, Any]:
    # Accept "lat,lng" or free-text address
    if "," in text:
        try:
            lat_str, lng_str = text.split(",", 1)
            return {
                "location": {
                    "latLng": {
                        "latitude": float(lat_str.strip()),
                        "longitude": float(lng_str.strip()),
                    }
                }
            }
        except Exception:
            pass
    return {"address": text}


async def call_google_routes_v2(
    *,
    origin: str,
    destination: str,
    mode: str | None,
    max_plans: int,
    depart_at_iso: str | None,
    language: str | None,
    region: str | None,
) -> dict:
    # Build body following user's sample, but map allowedTravelModes correctly
    body: dict[str, Any] = {
        "origin": _routes_v2_origin_or_dest(origin),
        "destination": _routes_v2_origin_or_dest(destination),
        "travelMode": "TRANSIT",
        "computeAlternativeRoutes": bool(max_plans and max_plans > 1),
        "transitPreferences": {
            "routingPreference": "LESS_WALKING",
        },
    }
    if depart_at_iso:
        body["departureTime"] = {"time": depart_at_iso}

    # Map mode → allowedTravelModes
    allowed: list[str] | None = None
    if mode:
        m = mode.lower()
        if m in ("metro", "subway"):
            allowed = ["SUBWAY"]
        elif m == "bus":
            allowed = ["BUS"]
        elif m == "train":
            allowed = ["TRAIN"]
        elif m == "tram":
            allowed = ["TRAM"]
        elif m == "rail":
            allowed = ["RAIL", "TRAIN"]
    if allowed:
        body["transitPreferences"]["allowedTravelModes"] = allowed

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        # Field mask to include transit details, distance/duration, and fare info
        "X-Goog-FieldMask": "routes.legs.steps.transitDetails,routes.duration,routes.distanceMeters,routes.travelAdvisory.transitFare",
    }
    params: dict[str, str] = {}
    if language:
        params["languageCode"] = language
    if region:
        params["regionCode"] = region

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            headers=headers,
            params=params,
            json=body,
        )
        data = resp.json()
        # Write raw response for debugging like user's workflow
        # try:
        #     with open("response.json", "w") as f:
        #         json.dump(data, f, indent=2)
        # except Exception:
        #     pass
        return data

def extract_transit_plans_from_routes(resp: dict) -> list[TransitPlan]:
    plans: list[TransitPlan] = []
    for route in (resp.get("routes") or []):
        legs = route.get("legs", []) or []
        steps_models: list[TransitStep] = []
        first_dep_iso: str | None = None
        last_arr_iso: str | None = None
        first_dep_local_text: str | None = None
        last_arr_local_text: str | None = None
        summary_parts: list[str] = []

        # Extract fare once per route (if present)
        fare_text: str | None = None
        try:
            fare = ((route.get("travelAdvisory") or {}).get("transitFare") or {})
            currency = fare.get("currencyCode")
            units = fare.get("units")
            nanos = fare.get("nanos")
            if units is not None or nanos is not None:
                # Format amount = units + nanos/1e9 with currency if available
                amount = (int(units or 0)) + (float(nanos or 0) / 1_000_000_000)
                if currency:
                    fare_text = f"{currency} {amount:,.2f}"
                else:
                    fare_text = f"{amount:,.2f}"
        except Exception:
            fare_text = None

        for leg in legs:
            for step in leg.get("steps", []) or []:
                # Detect transit steps by presence of transitDetails (travelMode may be omitted by field mask)
                td = step.get("transitDetails") or {}
                if not td:
                    continue
                sd = td.get("stopDetails", {}) or {}
                raw_dep_iso = sd.get("departureTime")
                raw_arr_iso = sd.get("arrivalTime")

                # Localized time strings (display as-is to the user)
                loc = td.get("localizedValues", {}) or {}
                dep_local_text = (
                    ((loc.get("departureTime") or {}).get("time") or {}).get("text")
                )
                arr_local_text = (
                    ((loc.get("arrivalTime") or {}).get("time") or {}).get("text")
                )

                # Capture first/last for overall plan fields
                if raw_dep_iso and first_dep_iso is None:
                    first_dep_iso = raw_dep_iso
                    first_dep_local_text = dep_local_text or first_dep_local_text
                if raw_arr_iso:
                    last_arr_iso = raw_arr_iso
                    last_arr_local_text = arr_local_text or last_arr_local_text

                # Transit metadata if available
                line_info = td.get("transitLine", {}) or {}
                vehicle_type = ((line_info.get("vehicle") or {}).get("type")) or "TRANSIT"
                line_name = line_info.get("nameShort") or line_info.get("name")
                headsign = td.get("headsign")
                num_stops = td.get("stopCount")
                departure_stop = (sd.get("departureStop") or {}).get("name")
                arrival_stop = (sd.get("arrivalStop") or {}).get("name")

                steps_models.append(
                    TransitStep(
                        vehicle=vehicle_type,
                        line_name=line_name,
                        headsign=headsign,
                        num_stops=num_stops,
                        departure_stop=departure_stop,
                        arrival_stop=arrival_stop,
                        # Use localized values directly for user display
                        departure_time_local=dep_local_text,
                        arrival_time_local=arr_local_text,
                    )
                )
                tag = f"{vehicle_type}: {line_name or ''} → {headsign or ''}".strip()
                summary_parts.append(tag)

        # Manual duration from first/last times
        duration_text: str | None = None
        if first_dep_iso and last_arr_iso:
            try:
                dt_dep = datetime.fromisoformat(first_dep_iso.replace("Z", "+00:00"))
                dt_arr = datetime.fromisoformat(last_arr_iso.replace("Z", "+00:00"))
                mins = round((dt_arr - dt_dep).total_seconds() / 60)
                duration_text = f"{mins} mins"
            except Exception:
                duration_text = None

        plans.append(
            TransitPlan(
                summary=" › ".join(p for p in summary_parts if p) or "Transit plan",
                # Prefer localized times for display at the plan level as well
                departure_time_local=first_dep_local_text,
                arrival_time_local=last_arr_local_text,
                duration_text=duration_text,
                fare_text=fare_text,
                steps=steps_models,
            )
        )
    return plans


# --- MCP Server Setup ---
mcp = FastMCP(
    "Instant Transit",
    auth=SimpleBearerAuthProvider(AUTH_TOKEN),
)


@mcp.tool
async def validate() -> str:
    # Mirror the pattern from the bearer-token server; if MY_NUMBER is set, return it.
    return MY_NUMBER or "google-transit:ok"


@mcp.tool
async def about() -> dict[str, str]:
    server_name = "Instant Transit details"
    server_description = dedent(
        """
        This MCP server helps discover public transit routes and upcoming departures between an origin and destination, with optional filtering by mode (bus,
        subway/metro), and preferred departure time. It also provides the fare details for the transit routes.

        Just ask "When is the next metro from X to Y" or "What is the cost of the bus from X to Y"
        and get instant answers!
        """
    ).strip()

    return {"name": server_name, "description": server_description}


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

    # If explicitly using metro, ensure queries target metro stations when users omit the phrase
    def _is_lat_lng(text: str | None) -> bool:
        if not text or "," not in text:
            return False
        try:
            lat_str, lng_str = text.split(",", 1)
            float(lat_str.strip())
            float(lng_str.strip())
            return True
        except Exception:
            return False

    def _ensure_metro_station_suffix(place: str) -> str:
        # Do not modify coordinates; only append if phrase is missing (case-insensitive)
        if _is_lat_lng(place):
            return place
        if "metro station" or "metro" in place.lower():
            return place
        return f"{place} metro station"

    adjusted_origin = parsed_origin
    adjusted_destination = parsed_destination
    if mode == "metro":
        if adjusted_origin:
            adjusted_origin = _ensure_metro_station_suffix(adjusted_origin)
        if adjusted_destination:
            adjusted_destination = _ensure_metro_station_suffix(adjusted_destination)

    # Prefer Routes v2 given user's request structure and times
    routes_v2 = await call_google_routes_v2(
        origin=adjusted_origin,
        destination=adjusted_destination,
        mode=mode,
        max_plans=max_plans,
        depart_at_iso=depart_at_iso,
        language=language,
        region=region,
    )
    plans = extract_transit_plans_from_routes(routes_v2)
    if not plans:
        return {"message": "No transit plans found"}

    # Trim
    plans = plans[: max(1, max_plans)]

    # Convert to serializable dict
    return {
        "origin": adjusted_origin,
        "destination": adjusted_destination,
        "plans": [p.model_dump() for p in plans],
    }


async def main():
    print("🚦 Starting Instant Transit MCP server on http://0.0.0.0:8087")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8087)


if __name__ == "__main__":
    asyncio.run(main())


