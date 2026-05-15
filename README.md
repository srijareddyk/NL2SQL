# NL2SQL
Natural Language to SQL Query Generator with RAG

**Title:** Natural Language to SQL Query Generator 

This project proposes the development of an intelligent Natural Language to SQL (Text-to-SQL) system that converts user questions written in plain English into executable SQL queries. The primary goal of the project is to make database querying more accessible to non-technical users while improving query accuracy on both known and unseen database schemas through the integration of Retrieval-Augmented Generation (RAG).

**Text Source:** The system will be trained using the WikiSQL dataset, which contains natural language questions paired with SQL queries and corresponding table schemas. In addition to the training dataset, the proposed system will support user-uploaded CSV and SQLite files to provide real-time schema context during inference.  

**Model Architecture:** The core model architecture will use T5-Base (220 million parameters), fine-tuned using LoRA (Low-Rank Adaptation) for parameter-efficient training. T5 is an encoder-decoder transformer model that is well suited for sequence-to-sequence tasks such as text-to-SQL translation. The encoder will process the natural language query along with the retrieved schema context through 12 transformer layers with multi-head self-attention mechanisms. The decoder will then generate SQL queries autoregressively using cross-attention to the encoder outputs. Model training will use cross-entropy loss with the AdamW optimizer, a learning rate of 5e-5, and early stopping based on execution accuracy.

**Extra Criteria:**

- RAG (Retrieval-Augmented Generation): When users upload a table, the system extracts the schema (column names, types, sample values) and stores it in a vector database. For each query, the system retrieves relevant schema information and similar example queries, then constructs an augmented prompt with this context before passing to the model. This provides the model with table-specific information that wasn't in the training data.

- Chatbot GUI: The interface will allow users to upload database files, ask questions in natural language, view generated SQL queries, and display query results in formatted tables. The chatbot will support multi-turn conversations, maintain query history, suggest common schema-based queries, validate generated SQL before execution, and provide explanations for SQL execution errors to improve usability and user interaction.
