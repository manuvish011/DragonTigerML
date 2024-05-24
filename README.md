# Dragon-Tiger Game with Machine Learning

This project implements the Dragon-Tiger game, where users can play against a computer that predicts the next move using a machine learning model. The project includes functionalities for updating the dataset, training the model, and playing the game. The model used is a Random Forest Classifier.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Dragon-Tiger is a simple two-choice game where players choose either Dragon (D) or Tiger (T). The computer predicts the player's choice based on previous moves using a trained machine learning model.

## Features
- Play Dragon-Tiger game with a computer opponent
- Update the game dataset with new moves and outcomes
- Train a Random Forest model using the collected dataset
- Predict the next move using the trained model

## Requirements
- Python 3.x
- Pandas
- Scikit-learn
- Joblib

## Setup
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/dragon-tiger-game.git
    cd dragon-tiger-game
    ```

2. Install the required packages:
    ```sh
    pip install pandas scikit-learn joblib
    ```

3. If you have an existing dataset, place it in the project directory as `dataset.csv`. If not, a new dataset will be created when you start playing the game.

## Usage
Run the script and follow the prompts to update the dataset, train the model, or play the game:
```sh
python dragon_tiger_game.py
```

You will be presented with options to:
1. Update the dataset with new moves and outcomes.
2. Play the Dragon-Tiger game.
3. Train the machine learning model.

### Update Dataset
To manually update the dataset:
1. Select option `1`.
2. Enter `move1` (T or D).
3. Enter `move2` (T or D).
The outcome will be automatically determined and the dataset will be saved.

### Play the Game
To play the game:
1. Select option `2`.
2. Enter your choice (T for Tiger, D for Dragon, or 'q' to exit).

### Train the Model
To train the model:
1. Select option `3`.
2. The model will be trained using the current dataset and the accuracy will be displayed.

## How It Works
- The dataset stores pairs of moves and outcomes.
- The model is trained using a Random Forest Classifier.
- During gameplay, the model predicts the computer's next move based on the player's previous moves.

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss any changes.
