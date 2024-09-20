# Gabo RAG Tutorial
**'Gabo'** is a **RAG (Retrieval-Augmented Generation)** system designed to enhance the capabilities of **LLMs (Large Language Models)** such as **'Llama 3.1'** or **'Phi 3.5**'. This project honors Colombian author **Gabriel García Márquez** by marking the tenth anniversary of his death, creating a specialized assistant to answer questions about his work, and using new technologies to further reveal his literary legacy.

[**Repository**](https://github.com/dafmontenegro/gabo-rag) | [**Python Notebook**](https://github.com/dafmontenegro/gabo-rag/blob/master/gabo_rag.ipynb)

- [1. Tools and Technologies](#1-tools-and-technologies)
- [2. How to run Ollama in Google Colab?](#2-how-to-run-ollama-in-google-colab)
  - [2.1 Ollama installation](#21-ollama-installation)
  - [2.2 Run 'ollama serve'](#22-run-ollama-serve)
  - [2.3 Run 'ollama pull \<model\_name\>'](#23-run-ollama-pull-model_name)
- [3. Exploring LLMs](#3-exploring-llms)
- [4. Data Extraction and Preparation](#4-data-extraction-and-preparation)
  - [4.1 Web Scraping and Chunking](#41-web-scraping-and-chunking)
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
For this, we simply go to the [Ollama downloads page](https://ollama.com/download/linux) and select **Linux**. The command is as follows

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

> At this stage, none of the models are expected to be capable of answering the question correctly, and they might even hallucinate while attempting to provide a response. To address this, we will begin constructing our **RAG** in the next section.

## 4. Data Extraction and Preparation
To collect the information that our **RAG** will use, we will perform **Web Scraping** of the section dedicated to [Gabriel Garcia Marquez](https://ciudadseva.com/autor/gabriel-garcia-marquez/) in the **Ciudad Seva web site**.

### 4.1 Web Scraping and Chunking
The first step will be to save the list of sources we will extract from the website into a variable.

```python
base_urls = ["https://ciudadseva.com/autor/gabriel-garcia-marquez/cuentos/",
             "https://ciudadseva.com/autor/gabriel-garcia-marquez/opiniones/",
             "https://ciudadseva.com/autor/gabriel-garcia-marquez/otrostextos/"]
```

Now we will create a function to collect all the links that lead to the texts. If we look at the HTML structure, we will notice that the information we're looking for is inside an `<article>` element with the class `status-publish`. Then, we simply extract the `href` attributes from the `<li>` elements inside the `<a>` tags.

```python
from langchain.document_loaders import WebBaseLoader

def get_urls(url):
    article = WebBaseLoader(url).scrape().find("article", "status-publish")
    lis = article.find_all("li", "text-center")
    return [li.find("a").get("href") for li in lis]
```
Let's see how many texts by the writer we can gather.

```python
gabo_urls = []

for base_url in base_urls:
    gabo_urls.extend(get_urls(base_url))

len(gabo_urls)
```

```text
OUTPUT: 51
```

Now that we have the URLs of the texts to feed our **RAG**, we just need to perform web scraping directly from the content of the stories. For that, we will build a function that follows a logic very similar to the previous function, which will initially give us the **raw text**, along with the **reference information** about what we are obtaining (the information found in `<header>`).

```python
def ciudad_seva_loader(url):
    article = WebBaseLoader(url).scrape().find("article", "status-publish")
    title = " ".join(article.find("header").get_text().split())
    article.find("header").decompose()
    texts = (" ".join(article.get_text().split())).split(". ")
    return [f"Fragmento {i+1}/{len(texts)} de '{title}': '{text}'" for i, text in enumerate(texts)]
```

There are indeed many ways to perform chunking, several of which are discussed in **"5 Levels of Text Splitting"** [2]. The most interesting idea for me about how to split texts, and what I believe fits best in this project, is **Semantic Splitting**. So, following that idea, we will ensure that the function divides all the texts by their periods, thus generating **semantic fragments in Spanish**.

> Tests were performed on the **Semantic Similarity** [3] offered by **Langchain**, but the results were worse. In this case, there is no need to do something extremely sophisticated, when the simplest and practically obvious solution is the best.

## References
[1] **Ollama. (s. f.). ollama/docs/tutorials/langchainpy.md at main · ollama/ollama. GitHub.** https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md

[2] **FullStackRetrieval-Com. (s. f.). RetrievalTutorials/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb at main · FullStackRetrieval-com/RetrievalTutorials. GitHub.** https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

[3] How to split text based on semantic similarity | 🦜️🔗 LangChain. (s. f.). https://python.langchain.com/docs/how_to/semantic-chunker/