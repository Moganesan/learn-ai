# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Task 1: Data Collection and Preparation
# Dummy knowledge base for C programming language
knowledge_base = {
    "What is a variable in C?": "A variable is a container that holds a value.",
    "How to declare a variable in C?": "You can declare a variable using the syntax: data_type variable_name;",
    "Can you write a c program?":"printf('Hello World')"
    # Add more C programming-related questions and answers to the knowledge_base
}

# Preprocess the data
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]

    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in filtered_text]

    return " ".join(stemmed_text)


# Preprocess the knowledge base
preprocessed_knowledge_base = {preprocess_text(question): answer for question, answer in knowledge_base.items()}

# Task 2: Building the Intent Classifier
# Create training data for the intent classifier
training_data = [
    ("What is a variable in C?", "definition"),
    ("How to declare a variable in C?", "example"),
    ("Can you write a c program?","Code")
    # Add more training data for different intents
]

# Extract features (i.e., convert text to numerical vectors)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([preprocess_text(question) for question, _ in training_data])

# Create the intent classifier (using a simple Naive Bayes classifier)
intent_classifier = MultinomialNB()
y_train = [intent for _, intent in training_data]
intent_classifier.fit(X_train, y_train)


# Task 3: Designing the Response Generator
def generate_response(user_query):
    # Preprocess user query
    preprocessed_query = preprocess_text(user_query)

    # Convert user query to vector using the same vectorizer used in training
    query_vector = vectorizer.transform([preprocessed_query])

    # Predict the intent of the user query
    intent = intent_classifier.predict(query_vector)[0]

    # Find the most relevant response from the knowledge base
    relevant_response = knowledge_base.get(user_query)  # Direct match in the original query
    if not relevant_response:
        preprocessed_query_vector = vectorizer.transform([preprocessed_query])
        similarity_scores = [vectorizer.transform([preprocess_text(k)]) for k in knowledge_base.keys()]
        similarity_scores = [max(score.sum(axis=1)) for score in similarity_scores]
        max_similarity_score = max(similarity_scores)
        most_relevant_question = list(knowledge_base.keys())[similarity_scores.index(max_similarity_score)]
        relevant_response = knowledge_base[most_relevant_question]

    return relevant_response


# Task 4: Interactive Chatbot Interface
def chatbot_interface():
    print("Chatbot: Hi! I'm here to teach you about C programming. How can I help you?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = generate_response(user_input)
        print("Chatbot:", response)


# Task 5: Testing and Improving the Chatbot
# Test the chatbot with various C programming-related queries
if __name__ == "__main__":
    chatbot_interface()
