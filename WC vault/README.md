# WC vault

Standalone Obsidian vault for the World Cup 2026 prediction system. Open this folder directly as a vault in Obsidian. It is separate from your main vault; nothing here is written into `The vault/`.

## Structure

- `Countries/` one note per team (snapshot, squad + availability, upcoming, your notes, change log)
- `Groups/` one note per group (standings probabilities, links to teams)
- `Matches/` one note per match (forecast, post-match, result vs prediction)
- `_forecast.md` tournament dashboard (champion ladder, stage probabilities)
- `_templates/` note templates used by the generator

## How facts flow (two-tier)

- AUTO blocks, between `<!-- WC26:AUTO:... START -->` and `... END -->`, are machine-written from the database on each run. Do not edit them; your edits there get replaced.
- HUMAN blocks (`## My read`, `## Manual overrides`) are yours. The generator never touches them. The `## Manual overrides` block is parsed back into the model, so what you write there influences predictions.

Probabilities are forecasts, not certainties. A single "most likely score" is usually only 8 to 12 percent likely. Read the W/D/L triple and top scorelines, and check the STALE banner.
