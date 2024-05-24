import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

# Initialize or load the dataset
dataset_path = 'dataset.csv'
if os.path.exists(dataset_path):
    data = pd.read_csv(dataset_path)
else:
    data = pd.DataFrame(columns=['move1', 'move2', 'outcome'])

# Function to play the Dragon-Tiger game
def dragon_tiger_game(model):
    choices = ['T', 'D']
    pattern = ''

    while True:
        user_choice = input("Enter your choice (T for Tiger, D for Dragon, or 'q' to exit): ").upper()
        if user_choice == 'Q':
            break

        computer_choice = predict_next_choice(pattern, model, choices)
        print("Computer's choice:", computer_choice)

        outcome = 'win' if user_choice == computer_choice else 'lose'
        new_entry = pd.DataFrame([[user_choice, computer_choice, outcome]], columns=['move1', 'move2', 'outcome'])
        global data
        data = pd.concat([data, new_entry], ignore_index=True)
        pattern += user_choice + computer_choice

    data.to_csv(dataset_path, index=False)
    print("Game ended. Dataset saved.")

# Function to update the dataset
def update_dataset():
    move1 = input("Enter move1 (T or D): ").upper()
    move2 = input("Enter move2 (T or D): ").upper()
    outcome = 'win' if move1 == move2 else 'lose'

    new_entry = pd.DataFrame([[move1, move2, outcome]], columns=['move1', 'move2', 'outcome'])
    global data
    data = pd.concat([data, new_entry], ignore_index=True)
    data.to_csv(dataset_path, index=False)
    print("Dataset updated and saved.")

# Function to train a machine learning model
def train_model():
    global data

    if len(data) < 10:  # Require more data for a more robust model
        print("Not enough data to train. Please update the dataset.")
        return None

    X = pd.get_dummies(data[['move1', 'move2']], columns=['move1', 'move2'], drop_first=True)
    y = (data['outcome'] == 'win').astype(int)  # Convert outcomes to binary labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    dump(clf, 'random_forest_model.joblib')
    print("Model trained and saved.")
    return clf

# Function to load an existing model
def load_model():
    if os.path.exists('random_forest_model.joblib'):
        model = load('random_forest_model.joblib')
        print("Model loaded successfully.")
        return model
    else:
        print("No existing model found. Please train a new model.")
        return None

# Function to predict the next choice using the trained model
def predict_next_choice(pattern, model, choices):
    if len(pattern) >= 2:
        sequence_to_predict = pattern[-2:]
        new_data = pd.get_dummies(
            pd.DataFrame([list(sequence_to_predict)], columns=['move1', 'move2']),
            columns=['move1', 'move2'], drop_first=True
        )
        new_data = new_data.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(new_data)
        return choices[prediction[0]]
    else:
        return random.choice(choices)

# Main loop for user interaction
while True:
    action = input("Do you want to (1) update the dataset, (2) play the game, or (3) train the model? Enter 'exit' to end: ")

    if action == '1':
        update_dataset()
    elif action == '2':
        model = load_model()
        if model is not None:
            dragon_tiger_game(model)
    elif action == '3':
        train_model()
    elif action.lower() == 'exit':
        break
    else:
        print("Invalid option. Please enter '1', '2', '3', or 'exit'.")
