import sqlite3

# Connect to the database
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Create users table with password hashing in mind
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- Store hashed password
    details TEXT
);
''')

# Create appliances table with foreign key and power rating column
cursor.execute('''
CREATE TABLE IF NOT EXISTS appliances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    institution_type TEXT NOT NULL,
    historical_units INTEGER NOT NULL,
    previous_bill REAL NOT NULL,
    appliance_name TEXT NOT NULL,
    avg_hours_used INTEGER NOT NULL,
    power_rating INTEGER NOT NULL,  -- New column for appliance wattage
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);
''')

# Commit changes and close connection
conn.commit()
conn.close()
