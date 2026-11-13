import asyncio
from random import randint

import dotenv
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

GCD_PROMPT = """
Compute the greatest common divisor (GCD) of two integers, A and B, using Euclid's algorithm.

Follow these steps:

Divide the larger number by the smaller number and compute the remainder.

While the remainder is not zero, replace the larger number with the smaller number, replace the smaller number with the remainder, and compute the new remainder.

When the remainder is zero, the GCD is the smaller number in the final division.

Execute the algorithm exactly as specified.

Return the final GCD as an integer.
"""

COPRIME_PAIRS = [(98581, 57185), (5381434291, 6535021284), (877311774988717, 100932996856453), (44605089451835675074, 39800203950254457483), (8317997619091563331637861, 8504991781521025589549324), (908917784777063186834584128825, 743370568296920787988931131561), (22454777762994318630658917357231529, 45972558163471731115766646282032523), (5963713098286115603031119660227492758014, 9897494185414577522246222599304248162503), (648358054941840304825726021198496021875953779, 412000183102303760001001698084896072709872256), (44732255746494267271602705439976888628871126366423, 63787612252400921304360834468287369424236035954331)]



def gcd(a: int, b: int) -> tuple[int, int]:
    """Return the nonnegative GCD and the number of remainder steps."""
    a, b = abs(a), abs(b)
    n_iter = 0

    while b:
        a, b = b, a % b
        n_iter += 1

    return a, n_iter


def generate_coprime_pairs() -> list[tuple[int, int]]:
    """Return one coprime pair per digit count from 5 to 50 in steps of 5."""
    pairs = []
    for digits in range(5, 51, 5):
        lower = 10 ** (digits - 1)
        upper = 10 ** digits - 1

        while True:
            a = randint(lower, upper)
            b = randint(lower, upper)
            if gcd(a, b)[0] == 1:
                pairs.append((a, b))
                break

    return pairs


async def gcd_llm(a: int, b: int) -> tuple[int, str]:
    model = "openai/gpt-6.1-sol"
    reasoning_effort = "medium"
    reasoning_summary = "detailed"

    prompt = f"{GCD_PROMPT.strip()}\n\nFirst number: {a}\nSecond number: {b}"

    pydantic_ai.BANNER_ENABLED = False
    agent = Agent(
        f"openrouter:{model}",
        output_type=int,
    )
    model_settings = OpenRouterModelSettings(openrouter_provider={"require_parameters": True})
    model_settings["extra_body"] = {
        "reasoning": {"effort": reasoning_effort, "summary": reasoning_summary}
    }
    response = await agent.run(prompt, model_settings=model_settings)
    return response.output, (response.response.thinking or "").strip()


async def run_test_cases(
    test_cases: list[tuple[int, int]], *, print_reasoning: bool = True, n_evals: int = 1
) -> None:
    if n_evals < 1:
        raise ValueError("n_evals must be at least 1")

    for index, (a, b) in enumerate(test_cases, start=1):
        print(f"========== Test case {index} ==========")
        print(f"a = {a}\nb = {b}")

        local_gcd, _ = gcd(a, b)
        print("\n--- Local GCD ---")
        print(local_gcd)

        evaluations = await asyncio.gather(*(gcd_llm(a, b) for _ in range(n_evals)))
        for eval_index, (llm_gcd, reasoning_text) in enumerate(evaluations, start=1):
            print(f"\n--- LLM GCD ({eval_index}/{n_evals}) ---")
            print(llm_gcd)

            if print_reasoning:
                print("\nReasoning trace:")
                print(reasoning_text or "No reasoning text returned.")

        correct_results = sum(llm_gcd == local_gcd for llm_gcd, _ in evaluations)
        accuracy = correct_results / n_evals
        print(f"\nLLM accuracy: {correct_results}/{n_evals} ({accuracy:.2%})")
        print()


async def main():
    dotenv.load_dotenv()

    for expected_gcd in (1, 4, 7, 16, 47, 64, 113, 150):
        print(f"========== Expected GCD: {expected_gcd} ==========")
        test_cases = [(a * expected_gcd, b * expected_gcd) for a, b in COPRIME_PAIRS]
        await run_test_cases(test_cases, print_reasoning=True, n_evals=5)


if __name__ == "__main__":
    asyncio.run(main())
