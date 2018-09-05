import sqlite3
import csv

class Error(Exception):
    pass

class OperationalError(Error):
    pass

try:
    conn = sqlite3.connect("sqlite3.db")
    cur = conn.cursor()
    try:
        cur.execute('create table supermarket(sname, item, item_weight, price, available)')
    except Exception as e:
        # print(e)
        pass
    sql = 'insert into supermarket values (?, ?, ?, ?, ?)'
    try:
        with open('morrisons.csv', 'r') as f:
            csvfile = csv.reader(f)
            for row in csvfile:
                cur.execute(sql, row)
    except Exception:
        pass
    else:
        conn.commit()
        cur.execute('select * from supermarket')
        for item in cur.fetchall():
            print(item)

except Exception as e:
    print("Error:", e)

finally:
    conn.close()
