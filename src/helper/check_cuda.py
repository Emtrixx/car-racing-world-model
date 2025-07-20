import torch


def check_cuda():
    """
    Checks for CUDA availability and prints GPU details if available.
    """
    if torch.cuda.is_available():
        print("✅ CUDA is available!")

        # Get the number of available GPUs
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {device_count}")

        # Print details for each GPU
        for i in range(device_count):
            print(f"--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability(i)}")

    else:
        print("❌ CUDA is not available on this system.")


if __name__ == "__main__":
    check_cuda()
