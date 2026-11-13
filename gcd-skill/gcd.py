import os

import dotenv
import openrouter

GCD_PROMPT = """
Compute the greatest common divisor (GCD) of two integers, A and B, using Euclid's algorithm.

Follow these steps:

Divide the larger number by the smaller number and compute the remainder.

While the remainder is not zero, replace the larger number with the smaller number, replace the smaller number with the remainder, and compute the new remainder.

When the remainder is zero, the GCD is the smaller number in the final division.

Execute the algorithm exactly as specified.

Output only the final GCD as a single integer. Do not include explanations, intermediate steps, labels, or any other text.
"""


def gcd(a: int, b: int) -> tuple[int, int]:
    """Return the nonnegative GCD and the number of remainder steps."""
    a, b = abs(a), abs(b)
    n_iter = 0

    while b:
        a, b = b, a % b
        n_iter += 1

    return a, n_iter


def send_chat(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model=None,
    reasoning_effort=None,
    reasoning_summary=None,
):
    """Send a user prompt using arguments or environment settings."""
    model = model or os.getenv("OPENROUTER_MODEL")
    if not model:
        raise ValueError("Pass a model or set OPENROUTER_MODEL.")

    reasoning = {}
    if reasoning_effort:
        reasoning["effort"] = reasoning_effort
    if reasoning_summary:
        reasoning["summary"] = reasoning_summary
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    with openrouter.OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"]) as client:
        return client.chat.send(messages=messages, model=model, reasoning=reasoning or None)


def main():
    dotenv.load_dotenv()
    examples = (
        (1239819283702104981928739812734120398012980980980121 * 7, 12391802380109822251293801209381029830192765043 * 7),
        # (123981928370210498192873981273412039801298098098012, 1239180238010982225129380120938102983019276504),
        # (12398192837021049819287398127341203984, 123918023801098222512938012093816),
        # (283746591824, 567493183664),
        # (19898984984984, 69894984984),
        #(129879871209 * 131, 1209812123098 * 131),
    )
    for index, (a, b) in enumerate(examples):
        if index > 0:
            print("----------")
        print(f"GCD({a}, {b})")
        print(f"Local GCD: {gcd(a, b)[0]}")
        response = send_chat(
            f"{GCD_PROMPT.strip()}\n\nFirst number: {a}\nSecond number: {b}",
            model="openai/gpt-6-sol",
            reasoning_effort="medium",
            reasoning_summary="detailed",
        )
        message = response.choices[0].message.model_dump(by_alias=True)
        summaries = [
            detail["summary"]
            for detail in message.get("reasoning_details") or []
            if detail.get("type") == "reasoning.summary" and detail.get("summary")
        ]
        reasoning_text = "\n".join(summaries) or message.get("reasoning")
        if reasoning_text:
            print(f"Reasoning:\n{reasoning_text}")
        print(f"OpenRouter answer:\n{message['content']}\n")


if __name__ == "__main__":
    main()
