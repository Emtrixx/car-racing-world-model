from openevolve import OpenEvolve
import asyncio

# Initialize the system
evolve = OpenEvolve(
    # initial_program_path="../train_ppo_sb3.py",
    initial_program_path="../test.py",
    evaluation_file="evaluator.py",
    config_path="config.yaml"
)


async def main():
    # Run the evolution
    best_program = await evolve.run(iterations=10)
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
