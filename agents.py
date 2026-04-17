from rag import retrieve_context

# ---- RESEARCH AGENT ----
def research_agent(prompt, call_llm, use_rag):
    context = retrieve_context(prompt) if use_rag else ""

    research_prompt = f"""
    You are a research expert.

    Topic: {prompt}

    Context:
    {context}

    Provide key insights, facts, and important points.
    """

    return call_llm(research_prompt)


# ---- WRITER AGENT ----
def writer_agent(research, tone, content_type, call_llm):
    writer_prompt = f"""
    You are a professional content writer.

    Write a {content_type} in a {tone} tone.

    Use this research:
    {research}

    Format:
    Title:
    Content:
    CTA:
    """

    return call_llm(writer_prompt)


# ---- EDITOR AGENT ----
def editor_agent(content, call_llm):
    return call_llm(f"""
    Improve clarity, grammar, and flow:

    {content}
    """)


# ---- SEO AGENT ----
def seo_agent(content, call_llm):
    return call_llm(f"""
    Optimize this for SEO.
    Add headings and keywords:

    {content}
    """)


# ---- WORKFLOW ----
def run_workflow(prompt, tone, content_type, call_llm, use_rag, show_steps):
    research = research_agent(prompt, call_llm, use_rag)
    draft = writer_agent(research, tone, content_type, call_llm)
    edited = editor_agent(draft, call_llm)
    final = seo_agent(edited, call_llm)

    if show_steps:
        return f"""
## 🔍 Research
{research}

## ✍️ Draft
{draft}

## 📝 Edited
{edited}

## 🚀 Final Output
{final}
"""
    return final