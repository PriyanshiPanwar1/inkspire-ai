from rag import retrieve_context

def generate_content(
    prompt,
    tone,
    content_type,
    call_llm,
    use_rag,
    add_caption=False,
    add_hashtags=False,
    add_hook=False,
    add_keywords=False
):
    from rag import retrieve_context

    context = retrieve_context(prompt) if use_rag else ""

    extras = ""

    if add_hook:
        extras += "\nHook: (attention-grabbing opening line)"

    if add_caption:
        extras += "\nCaption: (short engaging caption)"

    if add_hashtags:
        extras += "\nHashtags: (trending and relevant hashtags)"

    if add_keywords:
        extras += "\nKeywords: (SEO keywords)"

    structured_prompt = f"""
    You are a professional content creator.

    Use this context if relevant:
    {context}

    Create a {content_type} in a {tone} tone.

    Follow this format strictly:

    Title:
    Content:
    CTA:
    {extras}

    Topic: {prompt}
    """

    return call_llm(structured_prompt)