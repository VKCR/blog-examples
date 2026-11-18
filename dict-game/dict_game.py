from statistics import median

import pydantic_ai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from utils import NUM_ITERATIONS, WORDS_PER_LETTER, build_dictionary, get_prompt

pydantic_ai.BANNER_ENABLED = False


class DictionaryGameResult(BaseModel):
    words: list[str] = Field(
        min_length=NUM_ITERATIONS + 1,
        max_length=NUM_ITERATIONS + 1,
    )


def llm_dict_game(
    model: str = "openai/gpt-6.1-sol", reasoning: str = "medium"
) -> tuple[DictionaryGameResult, str]:
    """Return the LLM's word sequence and any thinking text it provides."""

    agent = Agent(
        f"openrouter:{model}",
        output_type=DictionaryGameResult,
        model_settings=OpenRouterModelSettings(
            openrouter_provider={"require_parameters": True},
            extra_body={"reasoning": {"effort": reasoning, "summary": "detailed"}},
        ),
    )
    result = agent.run_sync(get_prompt())
    return result.output, (result.response.thinking or "").strip()


def local_dict_game() -> DictionaryGameResult:
    word_dict = build_dictionary()

    # 1. Starting from an initial word
    current_word = 'aquatically'
    words = [current_word]

    # 2. Start the index at the length of the initial word
    i = len(current_word)

    # 3. Run the configured number of iterations
    for _ in range(NUM_ITERATIONS):
        # 4. Lookup the list of words starting with the last letter of the current word
        last_letter = current_word[-1]

        word_list = word_dict[last_letter]

        # 5. Use modulo to wrap the index within the word list
        new_word = word_list[i % WORDS_PER_LETTER]
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
            f"(Used index {i} % {WORDS_PER_LETTER} = {i % WORDS_PER_LETTER})"
        )
        i += len(previous_word) + len(word)
        previous_word = word


def compare_outputs(
    first: DictionaryGameResult,
    second: DictionaryGameResult,
) -> int:
    """Return the first divergence index, or the length for a full match."""
    if len(first.words) != len(second.words):
        raise ValueError("Dictionary game results have different sizes.")

    for index, (first_word, second_word) in enumerate(zip(first.words, second.words)):
        if first_word != second_word:
            return index

    return len(first.words)


def print_difference(
    first: DictionaryGameResult,
    second: DictionaryGameResult,
) -> None:
    """Print whether two game results match and their first divergence if any."""
    index = compare_outputs(first, second)
    if index == len(first.words):
        print("The dictionary game results are identical.")
    else:
        print(
            f"First divergence at index {index}: "
            f"{first.words[index]!r} != {second.words[index]!r}"
        )


def median_first_divergence(
    model: str,
    reference: DictionaryGameResult,
    verbose: bool = True,
) -> int:
    """Return the median first divergence index across five runs of the model."""
    divergence_indices = []
    for run in range(1, 6):
        print(f"\nModel {model}, run {run}/5")
        result, reasoning_trace = llm_dict_game(model=model)
        divergence_indices.append(compare_outputs(result, reference))
        if verbose:
            print_dictionary_game_result(result)
            print_difference(result, reference)
            print("LLM reasoning trace")
            print(reasoning_trace)

    return median(divergence_indices)


def main() -> None:
    """Evaluate four models against the local game and print their median divergences."""
    local_res = local_dict_game()
    print("Local dictionary game")
    print_dictionary_game_result(local_res)

    models = [
        "openai/gpt-6.1-sol",
        # "openai/gpt-6-sol",
        # "openai/gpt-6-terra",
        # "openai/gpt-6-luna",
    ]
    results = []
    for model in models:
        median_index = median_first_divergence(model, local_res)
        results.append((model, median_index))

    print("\nExperiment results")
    for model, median_index in results:
        print(f"- {model}: median first divergence = {median_index}")


if __name__ == "__main__":
    load_dotenv()
    main()

