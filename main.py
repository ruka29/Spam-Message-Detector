from src.train import train_and_evaluate
from src.detect import detect_anomalies

if __name__ == "__main__":
    train_and_evaluate()

    user_message = input("Enter a message to check if it's anomalous or not: ")
    test_message = [user_message]

    results = detect_anomalies(test_message)

    if results[0] == 1:
        print("The message is classified as *anomalous*.")
    else:
        print("The message is classified as *normal*.")
