import os

from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, Field

from utils import build_dictionary, get_prompt


class DictionaryGameResult(BaseModel):
    words: list[str] = Field(min_length=101, max_length=101)


def llm_dict_game(model: str = "openai/gpt-6-sol") -> DictionaryGameResult:
    """Run the dictionary game with an LLM and return its word sequence."""

    with OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"]) as client:
        response = client.chat.send(
            model=model,
            messages=[{"role": "user", "content": get_prompt()}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "dictionary_game_result",
                    "strict": True,
                    "schema": DictionaryGameResult.model_json_schema(),
                },
            },
            provider={"require_parameters": True},
        )

    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("The LLM returned no text content")

    return DictionaryGameResult.model_validate_json(content)


def local_dict_game() -> DictionaryGameResult:
    word_dict = build_dictionary()

    # 1. Starting from an initial word
    current_word = 'aquatically'
    words = [current_word]

    # 2. Start the index at the length of the initial word
    i = len(current_word)

    # 3. Do this for n_iter iterations

    n_iter = 100

    for _ in range(n_iter):
        # 4. Lookup the list of words starting with the last letter of the current word
        last_letter = current_word[-1]

        word_list = word_dict[last_letter]

        # 5. Print the i-th word of that list
        # Since i increments indefinitely, we use modulo 100 (i % 100) to safely index into the 100-word list
        new_word = word_list[i % 100]
        words.append(new_word)

        # 6. Increment i by the lengths of the previous and newly selected words
        i += len(current_word) + len(new_word)

        # 7. This becomes the new current word
        current_word = new_word

    return DictionaryGameResult(words=words)


def print_dictionary_game_result(result: DictionaryGameResult) -> None:
    """Print a dictionary game result with its iteration details."""
    initial_word = result.words[0]
    previous_word = initial_word
    i = len(initial_word)

    print("\nDictionary game : \n")
    print(f"Initial word: {initial_word} (Starting index i: {i})")
    print("-" * 50)

    for iteration, word in enumerate(result.words[1:], start=1):
        print(
            f"Iteration {iteration:3}: {word:<15} "
            f"(Used index {i} % 100 = {i % 100})"
        )
        i += len(previous_word) + len(word)
        previous_word = word


def compare_outputs(
        first: DictionaryGameResult,
        second: DictionaryGameResult,
) -> None:
    """Print the first index and words at which two game results diverge."""
    if len(first.words) != len(second.words):
        print("Size mismatch")
        return

    for index, (first_word, second_word) in enumerate(
            zip(first.words, second.words)
    ):
        if first_word != second_word:
            print(
                f"First divergence at index {index}: "
                f"{first_word!r} != {second_word!r}"
            )
            return

    print("The dictionary game results are identical.")


if __name__ == "__main__":
    load_dotenv()

    local_res = local_dict_game()
    print_dictionary_game_result(local_res)

    llm_res = llm_dict_game()
    print_dictionary_game_result(llm_res)

    compare_outputs(llm_res, local_res)

