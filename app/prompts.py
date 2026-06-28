from langchain_core.prompts import ChatPromptTemplate

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from company documents.\n"
            "- should_retrieve=False for general explanations/definitions.\n"
            "- should_retrieve=True for customer support questions: password reset, "
            "login help, account management, order tracking, returns, refunds, "
            "shipping policies, product availability explanations.\n"
            "- should_retrieve=False only for general knowledge that no company "
            "documents would cover.\n"
            "- If unsure (especially for support/FAQ-type questions), choose True.\n"
            "- Consider the conversation history: follow-up questions like "
            "'tell me more about that' or 'what about the second option?' "
            "should_retrieve=True since they reference prior context.",
        ),
        ("human", "Conversation history:\n{chat_history}\n\nQuestion: {question}"),
    ]
)

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Bloomly Assistant, a friendly support bot for Bloomly.\n\n"
            "Rules:\n"
            "- Greetings, thanks, small talk, emotional expressions → respond warmly "
            "in 1 sentence.\n"
            "- For anything that isn't a greeting/thanks/emotion, say in 1 sentence:\n"
            "  \"Let me connect you to the right resource for that!\"\n"
            "- Be warm. Keep it to 1 sentence. Use conversation history when relevant.",
        ),
        ("human", "Conversation History:\n{chat_history}\n\nCurrent Question:\n{question}"),
    ]
)

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business rag chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "Task:\n"
            "Answer the question based on the context.\n"
            "Don't mention that you are getting a context in your answer",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n",
        ),
    ]
)

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}",
        ),
    ]
)

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line.",
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}",
        ),
    ]
)



sql_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The user's question about database data may be vague. Rewrite it into an explicit, "
            "specific SQL query intent—what tables/columns to query, what aggregations to use, "
            "what filters to apply.\n\n"
            "Return JSON with key: refined_query (a complete, specific question).\n\n"
            "Examples:\n"
            "  'show me sales' -> 'Show me total sales amount by product category, ordered by highest revenue'\n"
            "  'how are customers distributed?' -> 'Show me the count of customers grouped by state, ordered by count descending'\n"
            "  'top products' -> 'Show me the top 10 most sold products by quantity'\n"
            "  'give me numbers' -> 'Show me total revenue, total orders, average order value, and total customers'\n\n"
            "Note: product category names in the database are in Portuguese (e.g., beleza_saude, relogios_presentes).\n"
            'Keep the English phrasing in the rewrite for clarity; the SQL generation step handles the mapping.\n\n'
            "Do NOT generate SQL. Only rewrite the natural language question to be more specific.\n"
            "If the user's question is missing a required parameter (specific item ID, "
            "product name, date range), note what's missing and suggest the user specify it. "
            "Do NOT invent placeholder values.\n"
            "Use conversation history when relevant. Focus primarily on the current question.",
        ),
        ("human", "Conversation History:\n{chat_history}\n\nCurrent Question:\n{question}"),
    ]
)

generate_sql_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a MySQL query generator.\n\n"
            "Available tables:\n{table_schema}\n\n"
            "Generate a valid MySQL SELECT query to answer the user's question.\n"
            "Return JSON with key: sql_query.\n\n"
            "Rules:\n"
            "- Use ONLY SELECT queries (read-only).\n"
            "- Keep queries SIMPLE. Prefer simple aggregation over nested subqueries.\n"
            "- When using JOINs, always PREFIX column names with table alias to avoid ambiguity.\n"
            "- Use LIMIT to cap results (max 50).\n"
            "- Use proper column names from the schema.\n"
            "- Do NOT use any column names not listed in the schema.\n"
            "- Always add LIMIT unless it's an aggregate query.\n"
            "- For counting sales per product category, first find product_category_name then count.\n"
            "- COLUMN VERIFICATION — Before writing the query, mentally check every column name against the schema above.\n"
            "  If a column does not appear verbatim in the schema, do NOT use it.\n"
            "- Never invent filter values — no placeholder IDs, SKUs, product names, "
            "dates, or prices. If the user didn't specify a concrete value, "
            "the query must omit that filter rather than fabricating one.\n"
            "- If a required filter is missing (e.g., no product ID provided), "
            "generate a query that lists available options instead (e.g., "
            "SELECT product_id, product_category_name FROM products LIMIT 20).\n"
            "- Prefer human-readable fields (product category name, customer city, etc.) "
            "over internal ID hashes. When querying order_items, JOIN with products to show "
            "product_category_name rather than just product_id.\n"
            "- PRODUCT CATEGORY NAMES — `products.product_category_name` stores PORTUGUESE names\n"
            "  (e.g., `beleza_saude`, `relogios_presentes`, `cama_mesa_banho`).\n"
            "  The `product_category_name_translation` table maps these to English (also underscore-separated, e.g., `health_beauty`, `watches_gifts`, `bed_bath_table`).\n"
            "  Do NOT use formatted English names like 'Beauty & Health' or 'Watches & Gifts' in WHERE clauses.\n"
            "  When a query involves product category names:\n"
            "    1. JOIN with `product_category_name_translation` ON products.product_category_name = product_category_name_translation.product_category_name\n"
            "    2. Filter using the EXACT value from `product_category_name_english` (e.g., `health_beauty`, not 'Health & Beauty')\n"
            "    3. Or filter using the Portuguese `product_category_name` value directly (e.g., `beleza_saude`)\n"
            "  Always include `product_category_name_translation` in the query when categories are referenced.\n"
            "Use conversation history when relevant. Focus primarily on the current question."
        ),
        ("human", "Conversation History:\n{chat_history}\n\nCurrent Question:\n{question}"),
    ]
)

sql_retry_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The previous SQL query failed with an error. Generate a SIMPLER corrected query.\n\n"
            "Available tables:\n{table_schema}\n\n"
            "Return JSON with key: sql_query.\n\n"
            "Rules:\n"
            "- Keep it as simple as possible (single table or simple JOIN).\n"
            "- Always prefix columns with table alias in JOINs.\n"
            "- Use LIMIT.\n"
            "- COLUMN VERIFICATION — This is critical:\n"
            "  Before writing the query, list every column name you plan to use (SELECT, WHERE, GROUP BY, ORDER BY, JOIN).\n"
            "  Check EACH one against the `Available tables:` block above.\n"
            "  If a column does NOT appear verbatim, do NOT use it.\n"
            "  If the question mentions something like 'product name' but the schema has no `product_name` column,\n"
            "  use the closest available column (e.g., `product_category_name`) or simplify the query to avoid it.\n"
            "- PRODUCT CATEGORY NAMES — `products.product_category_name` stores PORTUGUESE names.\n"
            "  Filter using Portuguese values (e.g., `beleza_saude`) or join the translation table and filter on the exact `product_category_name_english` (e.g., `health_beauty`, `watches_gifts`).\n"
            "  Never use formatted English names like 'Beauty & Health' — use the exact DB value.\n"
            "  When in doubt, JOIN with `product_category_name_translation`.\n"
            "- Do NOT guess table aliases. Use exactly the alias defined in the FROM/JOIN clause."
        ),
        (
            "human",
            "Question: {question}\n\nFailed SQL: {bad_sql}\n\nError: {error}",
        ),
    ]
)

summarize_sql_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data analyst assistant.\n\n"
            "Summarize the SQL query results for the user in a clear, natural language response.\n"
            "Present numbers in a readable format. Use bullet points for lists.\n"
            "Do NOT mention the SQL query itself.",
        ),
        (
            "human",
            "Question: {question}\n\nSQL: {sql_query}\n\nResults: {sql_result}",
        ),
    ]
)

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{\"retrieval_query\": \"NexaAI free trial duration trial period plans\"}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{\"retrieval_query\": \"NexaAI refund policy cancellation refund timeline charges\"}}\n"
            "Use conversation history when relevant. Focus primarily on the current question.",
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\n"
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}",
        ),
    ]
)

escalation_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Determine whether this conversation should be escalated to a human support agent.\n"
            "Return JSON with keys: escalate (bool), reason (string).\n\n"
            "Escalate immediately (HARD TRIGGERS):\n"
            '- User explicitly requests a human ("speak to a person", "transfer me", "I need a human")\n'
            '- User files a complaint ("I want to complain", "I need to escalate this")\n'
            '- User expresses strong frustration ("this is ridiculous", "you are useless", "unacceptable")\n\n'
            "Escalate after evaluation (SOFT TRIGGERS):\n"
            "- Negative sentiment persists across 3+ user turns\n"
            "- The same issue remains unresolved after multiple attempts\n\n"
            "Answer Quality Signals:\n"
            "- issup: {issup}\n"
            "- isuse: {isuse}\n\n"
            "CRITICAL RULE: If issup is 'fully_supported' AND isuse is 'useful', "
            "the question was successfully answered. Do NOT escalate for 'unresolved_issue' — "
            "the user may dislike the answer, but that is not an escalation trigger.\n"
            "CRITICAL RULE: If both issup and isuse are empty strings, the conversation agent "
            "handled the turn without document support. This means the user's query was "
            "non-informational (greeting, emotion, or small talk). Do NOT escalate — "
            "a single unanswered question is a routing issue, not an escalation trigger.\n\n"
            "Reason must be one of:\n"
            "human_requested, complaint, frustration, repeated_negative_sentiment, unresolved_issue, none",
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\n"
            "Current Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "issup: {issup}\n"
            "isuse: {isuse}",
        ),
    ]
)

handoff_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate a concise handoff summary for a human support agent.\n\n"
            "Structure:\n"
            "User Goal\n"
            "-----------\n"
            "<what the user was trying to do>\n\n"
            "System Actions\n"
            "-----------\n"
            "Recent documents consulted:\n"
            "- <document titles>\n\n"
            "Recent SQL queries executed:\n"
            "- <SQL queries>\n\n"
            "Current Answer\n"
            "-----------\n"
            "<the answer the user received>\n\n"
            "Escalation Reason\n"
            "-----------\n"
            "<reason>",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Current Question: {question}\n\n"
            "Current Answer: {answer}\n\n"
            "Escalation Reason: {reason}\n\n"
            "Current turn documents: {current_doc_titles}\n\n"
            "Current turn SQL query: {current_sql_query}\n\n"
            "Recent documents consulted (last 5 turns): {rag_docs_used}\n\n"
             "Recent SQL queries executed (last 5 turns): {sql_queries_executed}",
        ),
    ]
)

react_supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a supervisor agent for Bloomly customer support. Your job is to answer user questions "
            "by coordinating specialized tools.\n\n"
            "You have these tools:\n"
            "- rag(question: str) — Searches company documents (policies, benefits, refunds, HR, leadership, etc.)\n"
            "- sql(question: str) — Queries the e-commerce database (sales, revenue, orders, customers, products, categories)\n"
            "- conversation(question: str) — Remembers previous conversation context (user identity, preferences, past questions)\n\n"
            "Previous conversation with the user:\n"
            "{chat_history}\n\n"
            "Current question: {question}\n\n"
            "Previous tool calls and results:\n"
            "{tool_results}\n\n"
            "Rules:\n"
            "- If the question has multiple independent parts, call tools in PARALLEL to save time\n"
            "- If the question needs one piece of information, call a SINGLE tool\n"
            "- If you already have enough information or the question is conversational, RESPOND directly\n"
            "- If a tool fails, decide whether to retry or answer with what you have\n"
            "- Never ask the user for clarification — use the tools to find the answer\n"
            "- Use the conversation history to understand context and follow-up questions\n\n"
            "Respond with a JSON object in one of these formats:\n\n"
            "1. Single tool call:\n"
            '{{"action": "single", "tool": "rag"|"sql"|"conversation", "query": "specific question for the tool"}}\n\n'
            "2. Parallel tool calls (for compound questions):\n"
            '{{"action": "parallel", "calls": [{{"tool": "rag"|"sql"|"conversation", "query": "sub-question 1"}}, {{"tool": "rag"|"sql"|"conversation", "query": "sub-question 2"}}]}}\n\n'
            "3. Direct response:\n"
            '{{"action": "respond", "answer": "your complete answer to the user"}}\n\n'
            "IMPORTANT: Return ONLY valid JSON. No extra text, no markdown formatting.",
        ),
    ]
)
