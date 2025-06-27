import time


def wait_and_print():
    """Waits for 5 seconds and then prints a message."""
    print("Starting countdown...")
    time.sleep(5)  # Wait for 5 seconds
    print("5 seconds have passed!")


if __name__ == "__main__":
    wait_and_print()
