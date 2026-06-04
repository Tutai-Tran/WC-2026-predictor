from wc26 import db


def test_init_db_creates_schema(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    assert db.get_schema_version(conn) == db.SCHEMA_VERSION
    names = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for t in ["teams", "players", "matches", "results_ledger", "ratings",
              "predictions", "model_runs"]:
        assert t in names


def test_team_insert_and_unique(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    conn.execute("INSERT INTO teams (name, group_letter) VALUES (?, ?)", ("Brazil", "A"))
    conn.commit()
    row = conn.execute("SELECT name, group_letter FROM teams WHERE name='Brazil'").fetchone()
    assert row["group_letter"] == "A"
