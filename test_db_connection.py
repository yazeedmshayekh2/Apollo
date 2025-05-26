from pymongo import MongoClient
import sys

def test_mongodb_connection():
    try:
        # Connect to MongoDB
        print("Attempting to connect to MongoDB...")
        client = MongoClient('mongodb://localhost:27017/')
        
        # Test the connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        # Create/Get database and collection
        db = client['face_verification']
        collection = db['face_features']
        
        # Test collection operations
        print("\nTesting collection operations...")
        
        # Insert a test document
        test_doc = {
            'user_id': 'test_user',
            'features': [0.1, 0.2, 0.3]  # Sample features
        }
        
        # Insert and verify
        result = collection.insert_one(test_doc)
        print(f"Inserted test document with ID: {result.inserted_id}")
        
        # Retrieve and verify
        retrieved_doc = collection.find_one({'user_id': 'test_user'})
        print(f"Retrieved document: {retrieved_doc}")
        
        # Clean up test data
        collection.delete_one({'user_id': 'test_user'})
        print("Test document cleaned up")
        
        print("\nDatabase connection and operations test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        print("\nPlease make sure MongoDB is running on your system.")
        print("You can start MongoDB using:")
        print("sudo service mongod start  # For Ubuntu/Debian")
        print("or")
        print("mongod  # If installed locally")
        return False

if __name__ == "__main__":
    test_mongodb_connection() 