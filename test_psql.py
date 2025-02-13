import psycopg2

# Database credentials (replace with your actual credentials)
DB_CONFIG = {
    "dbname": "url",
    "user": "postgres",
    "password": "arnavmodi",
    "host": "localhost",
    "port": "5432"
}

def test_psql_store():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create test table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_test (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                label BOOLEAN NOT NULL
            );
        """)
        conn.commit()
        
        # Insert test data
        test_url = "http://example.com"
        test_label = True  # Assume phishing
        cursor.execute("INSERT INTO feedback_test (url, label) VALUES (%s, %s)", (test_url, test_label))
        conn.commit()
        
        # Fetch stored data
        cursor.execute("SELECT * FROM feedback_test;")
        rows = cursor.fetchall()
        
        print("Stored Data:")
        for row in rows:
            print(row)
        
        # Close connection
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_psql_store()
