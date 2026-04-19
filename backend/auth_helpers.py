import bcrypt
from pymongo import MongoClient
from .config import MONGODB_URI

# Initialize MongoDB client
client = MongoClient(MONGODB_URI)
db = client.get_database("test") # Using 'test' as seen in Node code or default
users_collection = db.users

def hash_password(password):
    """Hashes a password for storing it in the database."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)

def check_password(password, hashed_password):
    """Checks if a password matches its hash."""
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def find_user_by_email(email):
    """Finds a user by their email address."""
    return users_collection.find_one({"email": email})

def create_user(name, email, password, role='user'):
    """Creates a new user record in the database."""
    hashed = hash_password(password)
    user_data = {
        "name": name,
        "email": email,
        "password": hashed,
        "role": role
    }
    result = users_collection.insert_one(user_data)
    user_data["_id"] = str(result.inserted_id)
    # Don't return password in user data
    del user_data["password"]
    return user_data

def format_user_response(user_dict):
    """Formats the user dictionary for response (removes sensitive data)."""
    if not user_dict:
        return None
    return {
        "_id": str(user_dict["_id"]),
        "name": user_dict.get("name"),
        "email": user_dict.get("email"),
        "role": user_dict.get("role", "user")
    }
