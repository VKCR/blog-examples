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


def gcd(a: int, b: int) -> tuple[int, int]:
    """Return the nonnegative GCD and the number of remainder steps."""
    a, b = abs(a), abs(b)
    n_iter = 0

    while b:
        a, b = b, a % b
        n_iter += 1

    return a, n_iter


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
    test_cases = [
        (1239819283702104981928739812734120398012980980980121 * 228, 12391802380109822251293801209381029830192765043 * 228),
        # (123981928370210498192873981273412039801298098098012, 1239180238010982225129380120938102983019276504),
        # (12398192837021049819287398127341203984, 123918023801098222512938012093816),
        # (283746591824, 567493183664),
        # (19898984984984, 69894984984),
        #(129879871209 * 131, 1209812123098 * 131),
    ]
    run_test_cases(test_cases)


if __name__ == "__main__":
    main()
