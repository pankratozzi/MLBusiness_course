import sqlite3


def create_db(db_name: str = 'news.sqlite'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS news_db')
    c.execute('CREATE TABLE news_db (id INTEGER PRIMARY KEY, \
                                     news_text TEXT, \
                                     label INTEGER NOT NULL, \
                                     score DECIMAL(1,8) NOT NULL DEFAULT 0, \
                                     comment VARCHAR(200), \
                                     created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()


def insert_data(db='news.sqlite', **data):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('INSERT INTO news_db (news_text, label, score) VALUES \
              (?, ?, ?)', (data.get('text'), data.get('label'), data.get('score')))
    conn.commit()
    conn.close()


def inspect_data(ix: int, db='news.sqlite') -> list:
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT * FROM news_db WHERE id = ?', (ix,))
    results = c.fetchall()
    conn.close()
    return results[0]


def watch_all(db='news.sqlite'):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT * FROM news_db')
    results = c.fetchall()
    conn.close()
    return results


def delete_data(ix: int, db='news.sqlite'):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('DELETE FROM news_db WHERE id = ?', (ix,))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    pass
