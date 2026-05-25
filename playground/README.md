# Playground

A FastAPI + React demo that wraps the [`causal-lift`](../) library in a web UI.
Useful for kicking the tires without writing Python, or as a hosted lead-gen
surface.

## Run locally

```bash
# 1. Install the library in editable mode from the repo root
pip install -e ..

# 2. Install playground backend deps
pip install -r backend/requirements.txt

# 3. Start the API
python -m uvicorn --app-dir backend main:app --port 8000 &

# 4. Start the React frontend (in another shell)
cd frontend && npm install && npm run dev

# 5. Open http://localhost:5173
```

The frontend is in `frontend/`, the FastAPI wrapper is in `backend/`.

## What's here vs the library

| Surface         | Lives in              | Purpose                                                  |
|-----------------|-----------------------|----------------------------------------------------------|
| `causal_lift`   | `../src/causal_lift/` | All analysis logic, CLI, dataclasses. Pip-installable.   |
| `backend/`      | this folder           | Thin HTTP wrapper. CSV upload, JSON serialization, CORS. |
| `frontend/`     | this folder           | React UI: file drop, dashboard, charts, CSV export.      |

Nothing in `playground/backend/` should contain analysis logic — if it does,
move it into `causal_lift` and import it back.
