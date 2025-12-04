import sqlite3, os
from datetime import datetime

DB_PATH = 'outputs/frc_events.db'

class DB:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS robots (
                track_id INTEGER PRIMARY KEY,
                team_name TEXT,
                score INTEGER DEFAULT 0,
                last_event_time TEXT
            )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                event_name TEXT,
                points INTEGER,
                frame INTEGER,
                timestamp TEXT
            )""")
        self.conn.commit()

    def add_or_update_robot(self, track_id, team_name=None):
        cur = self.conn.cursor()
        cur.execute('SELECT track_id FROM robots WHERE track_id=?', (track_id,))
        if cur.fetchone() is None:
            cur.execute('INSERT INTO robots(track_id, team_name, score) VALUES(?,?,?)', (track_id, team_name, 0))
        else:
            if team_name:
                cur.execute('UPDATE robots SET team_name=? WHERE track_id=?', (team_name, track_id))
        self.conn.commit()

    def add_event(self, track_id, event_name, points, frame):
        cur = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute('INSERT INTO events(track_id, event_name, points, frame, timestamp) VALUES(?,?,?,?,?)',
                    (track_id, event_name, points, frame, now))
        cur.execute('UPDATE robots SET score = score + ?, last_event_time=? WHERE track_id=?',
                    (points, now, track_id))
        self.conn.commit()

    def get_scoreboard(self, limit=20):
        cur = self.conn.cursor()
        cur.execute('SELECT track_id, team_name, score FROM robots ORDER BY score DESC LIMIT ?', (limit,))
        return cur.fetchall()
