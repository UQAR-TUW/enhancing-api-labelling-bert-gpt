

def create_prompt(relevant_words: list[str],
                  target_labels: list[str]) -> list[dict]:
    """
    Creates a prompt for the classifier using relevant words and target labels
    :param relevant_words: Relevant words extracted from OpenAPI documents
    :param target_labels: Labels used as choices for the classifier
    :return: System and User prompt for the classifier
    """

    template = [
        {
            "role": "system",
            "content": "You will receive keywords extracted from OpenAPI documents.\nYour task is to classify the document into one of the following categories: {target_labels}.\nRespond only with the category name."
        },
        {
            "role": "user",
            "content": "{relevant_words}"
        }
    ]

    template[0]["content"] = template[0]["content"].format(target_labels=", ".join(target_labels))
    template[1]["content"] = ", ".join(relevant_words[:50])

    return template
