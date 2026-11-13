from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from utils import NUM_ITERATIONS, WORDS_PER_LETTER, build_dictionary, get_prompt


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

    llm_res, llm_reasoning_trace = llm_dict_game()
    print_dictionary_game_result(llm_res)

    compare_outputs(llm_res, local_res)

    print("LLM reasoning trace : ")
    print(llm_reasoning_trace)

