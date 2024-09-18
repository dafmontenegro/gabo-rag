# Gabo RAG Tutorial
**'Gabo'** is a **RAG (Retrieval-Augmented Generation)** system designed to enhance the capabilities of **LLMs (Large Language Models)** such as **'Llama 3.1'** or **'Phi 3.5**'. This project honors Colombian author **Gabriel García Márquez** by marking the tenth anniversary of his death, creating a specialized assistant to answer questions about his work, and using new technologies to further reveal his literary legacy.

- [1. Tools and Technologies](#1-tools-and-technologies)
- [2. How to run Ollama in Google Colab?](#2-how-to-run-ollama-in-google-colab)
  - [2.1 Ollama installation](#21-ollama-installation)
  - [2.2 Run 'ollama serve'](#22-run-ollama-serve)
  - [2.3 Run 'ollama pull \<model\_name\>'](#23-run-ollama-pull-model_name)
- [3. Exploring LLMs](#3-exploring-llms)
- [References](#references)

## Author

- **Daniel Felipe Montenegro** [GitHub](https://github.com/dafmontenegro) | [Blog](https://www.youtube.com/@MiAmigoMelquiades) | [X](https://x.com/dafmontenegro)

## 1. Tools and Technologies

- [**Ollama**](https://ollama.com/): Running models ([Llama 3.1 ](https://ollama.com/library/llama3.1)and [Phi 3.5](https://ollama.com/library/phi3.5)) and embeddings ([Nomic](https://ollama.com/library/nomic-embed-text))
- [**LangChain**](https://python.langchain.com/docs/introduction/): Framework and web scraping tool
- [**Chroma**](https://docs.trychroma.com/): Vector database

> A special thanks to ['Ciudad Seva (Casa digital del escritor Luis López Nieves)'](https://ciudadseva.com/quienes-somos/), from which the texts used in this project were extracted and where a comprehensive [Spanish Digital Library](https://ciudadseva.com/biblioteca/) is available.

## 2. How to run Ollama in Google Colab?

### 2.1 Ollama installation
For this, we simply go to the [Ollama downloads page](https://ollama.com/download/linux) and select **Linux**. The command is as follows `curl -fsSL https://ollama.com/install.sh | sh`

```bash
!curl -fsSL https://ollama.com/install.sh | sh
```

---

### 2.2 Run 'ollama serve'
If you run ollama serve, you will encounter the issue where you cannot execute subsequent cells and your script will remain stuck in that cell indefinitely. To resolve this, you simply need to run the following command:

```bash
!nohup ollama serve > ollama_serve.log 2>&1 &
```

After running this command, it is advisable to wait a reasonable amount of time for it to execute before running the next command, so you can add something like:

```python
import time
time.sleep(3)
```

---

### 2.3 Run 'ollama pull <model_name>'
For this project, we will use [Llama 3.1 ](https://ollama.com/library/llama3.1)and [Phi 3.5](https://ollama.com/library/phi3.5), so we perform the respective pulls of the models.

```bash
!ollama pull llama3.1
!ollama pull phi3.5
```

## 3. Exploring LLMs
Now that we have our LLMs, it's time to test them with what will be our control question.

```python
test_message = "¿Cuántos hijos tiene la señora vieja del cuento Algo muy grave va a suceder en este pueblo?"
# EN:"How many children does the old woman in the story 'Something Very Serious Is Going to Happen in This Town' have?"
```

> 'Gabo' will be designed to function in Spanish, as it was Gabriel García Márquez's native language and his literary work is also in this language.

The information is found at the beginning of [the story,](https://ciudadseva.com/texto/algo-muy-grave-va-a-suceder-en-este-pueblo/) so we expect it to be something that can be answered if it has the necessary information.

```text
ES
Fragmento inicial de 'Algo muy grave va a suceder en este pueblo' de Gabriel García Márquez.
"Imagínese usted un pueblo muy pequeño donde hay una señora vieja que tiene dos hijos, uno de 17 y una hija de 14... "

EN
Initial excerpt from 'Something Very Serious Is Going to Happen in This Town' by Gabriel García Márquez:
"Imagine a very small town where there is an old woman who has two children, a 17-year-old son and a 14-year-old daughter..."
```

Before we can invoke the LLM, we need to install LangChain. [1]

```bash
!pip install -qU langchain_community
```

Now we create the models.

```python
from langchain_community.llms import Ollama

llm_llama = Ollama(model="llama3.1")

llm_phi = Ollama(model="phi3.5")
```

Invoke Llama 3.1

```python
llm_llama.invoke(test_message)
```

Invoke Phi 3.5

```python
llm_phi.invoke(test_message)
```

> At this stage, none of the models are expected to be capable of answering the question correctly, and they might even hallucinate while attempting to provide a response. To address this, we will begin constructing our RAG in the next section.

## References
[1] **Ollama. (s. f.). ollama/docs/tutorials/langchainpy.md at main · ollama/ollama. GitHub.** https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md