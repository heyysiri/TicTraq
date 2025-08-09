## Google Transit MCP Server -- TicTraq

A Model Context Protocol (MCP) server that returns upcoming public transit route options using Google Routes API v2 (Directions). It surfaces localized departure/arrival times and computes duration strictly as arrival minus departure.

### Key features
- **Localized times**: `departure_time_local` and `arrival_time_local` come from Google's `localizedValues` and are shown exactly as returned in the requested language/locale.
- **Accurate duration**: `duration_text` is computed as minutes between ISO `arrivalTime` and `departureTime` from the API; it does not use the API's generic `duration` label.
- **Mode filtering**: Optional `mode` parameter to target specific transit types; special handling for `metro` queries to improve results.
- **Robust parsing**: Detects transit steps via `transitDetails` so field masks don't omit critical fields.

## Prerequisites
- Python 3.11+
- A Google Maps Platform API key with access to Directions (Routes v2)

## Setup
1. Create and activate a virtual environment, then install dependencies (uses `uv`):

```bash
uv venv
uv sync
source .venv/bin/activate
```

2. Create a `.env` in the project root:

```bash
cp .env.example .env
```

3. Fill in the environment variables:

```env
AUTH_TOKEN=<your-auth-token>
GOOGLE_MAPS_API_KEY=<your-google-maps-api-key>
# Optional, used by the validate tool
MY_NUMBER=91<your-number>
```

## Run the server

```bash
python mcp-transit-google/mcp_transit.py
```

Expected output:

```
🚦 Starting Google Transit MCP server on http://0.0.0.0:8087
```

To expose it publicly (required for remote clients), use a tunnel like ngrok:

```bash
ngrok http 8087
```

Then connect your MCP client to `https://<your-ngrok-domain>/mcp` using your `AUTH_TOKEN`.

## Tools

### validate()
Simple health check. Returns `MY_NUMBER` if set, otherwise `google-transit:ok`.

### transit_route(...)
Finds upcoming transit route options. Parameters:

- `user_question` (str, required): Freeform text, e.g. "metro to Habsiguda from Parade Ground".
- `origin` (str | None): Address or `lat,lng`. Optional if inferable from `user_question`.
- `destination` (str | None): Address or `lat,lng`. Optional if inferable.
- `mode` (Literal["bus", "subway", "train", "tram", "rail", "metro"] | None): Filter to a specific mode.
- `depart_at_iso` (str | None): Desired departure time in ISO8601 local time. Mutually exclusive with `arrive_by_iso`.
- `arrive_by_iso` (str | None): Desired arrival deadline in ISO8601 local time. Mutually exclusive with `depart_at_iso`.
- `language` (str | None, default: "en"): Response language code.
- `region` (str | None, default: "IN"): Region bias code.
- `max_plans` (int, default: 3): Maximum number of plans to return.

Behavioral notes:
- If both `depart_at_iso` and `arrive_by_iso` are provided, the server returns an error.
- When `mode == "metro"` and the input is free-text (not coordinates), the server appends "metro station" to origin/destination if not already present, e.g. `"Habsiguda" → "Habsiguda metro station"`, to improve accuracy. Coordinates are left unchanged.

## Response schema

```json
{
  "origin": "Parade Ground metro station",
  "destination": "Habsiguda metro station",
  "plans": [
    {
      "summary": "SUBWAY: Blue Line → Nagole",
      "departure_time_local": "PM ৫.২৮", // from localizedValues (example locale)
      "arrival_time_local": "PM ৫.৩৮",   // from localizedValues (example locale)
      "duration_text": "10 mins",        // computed = arrival - departure
      "fare_text": null,
      "steps": [
        {
          "vehicle": "SUBWAY",
          "line_name": "Blue Line",
          "headsign": "Nagole",
          "num_stops": 5,
          "departure_stop": "Parade Ground",
          "arrival_stop": "Habsiguda",
          "departure_time_local": "PM ৫.২৮", // localized string
          "arrival_time_local": "PM ৫.৩৮"    // localized string
        }
      ]
    }
  ]
}
```

Field meanings:
- **departure_time_local / arrival_time_local**: Localized strings for display, taken from `transitDetails.localizedValues.(departureTime|arrivalTime).time.text`.
- **duration_text**: Minutes between ISO `stopDetails.departureTime` and `stopDetails.arrivalTime` (e.g., `"10 mins"`).
- **vehicle / line_name / headsign / num_stops / stops**: Taken from `transitDetails.transitLine`, `transitDetails.headsign`, and `transitDetails.stopCount/stopDetails`.

## Troubleshooting
- Ensure `.env` has valid `AUTH_TOKEN` and `GOOGLE_MAPS_API_KEY`.
- If you get "No transit plans found", try specifying `mode`, adjusting `origin`/`destination`, or checking that public transit is available for the route/time.
- If you need to debug the raw Google API response, you can temporarily write it to `response.json` by uncommenting the lines in `call_google_routes_v2` that dump the response.
- Network access to `routes.googleapis.com` must be allowed from your environment.

## Security
- Keep `AUTH_TOKEN` and `GOOGLE_MAPS_API_KEY` secret and out of version control.

## License
See `LICENSE` in the repository root.


