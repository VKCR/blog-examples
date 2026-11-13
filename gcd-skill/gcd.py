from random import randint

import dotenv
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


def gcd_llm(a: int, b: int) -> tuple[int, str]:
    model = "openai/gpt-6-sol"
    reasoning_effort = "medium"
    reasoning_summary = "detailed"

    prompt = f"{GCD_PROMPT.strip()}\n\nFirst number: {a}\nSecond number: {b}"

    agent = Agent(
        f"openrouter:{model}",
        output_type=int,
    )
    model_settings = OpenRouterModelSettings(openrouter_provider={"require_parameters": True})
    model_settings["extra_body"] = {
        "reasoning": {"effort": reasoning_effort, "summary": reasoning_summary}
    }
    response = agent.run_sync(prompt, model_settings=model_settings)
    return response.output, (response.response.thinking or "").strip()


def run_test_cases(test_cases: list[tuple[int, int]], *, print_reasoning: bool = True) -> None:
    for index, (a, b) in enumerate(test_cases, start=1):
        print(f"========== Test case {index} ==========")
        print(f"a = {a}\nb = {b}")

        print("\n--- Local GCD ---")
        print(gcd(a, b)[0])

        print("\n--- LLM GCD ---")
        llm_gcd, reasoning_text = gcd_llm(a, b)
        print(llm_gcd)

        if print_reasoning:
            print("\n--- Reasoning trace ---")
            print(reasoning_text or "No reasoning text returned.")
        print()


def main():
    dotenv.load_dotenv()
    run_test_cases(COPRIME_PAIRS, print_reasoning=False)


if __name__ == "__main__":
    main()
