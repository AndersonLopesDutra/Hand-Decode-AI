# 🎨 Hand-Decode AI

![Logo do Projeto](web_app/static/images/logo.jpg)

O Hand-Decode AI é um projeto de Aplicação web MLOps full-stack para treinar e testar um modelo de Deep Learning (PyTorch CNN) para reconhecimento de dígitos em tempo real, com feedback de treino via stream, resposta por voz e servido por Flask.

<img width="1219" height="601" alt="image" src="https://github.com/user-attachments/assets/17d900fd-df35-4812-aaf8-8c877d11e07d" />

### ✨ Funcionalidades Principais

* **Interface de Teste (Inferência):** Desenhe um dígito (0-9) em um canvas e receba a previsão do modelo em tempo real.
* **Interface de Treinamento (MLOps):** Inicie um novo ciclo de treinamento do modelo diretamente pela interface web.
* **Feedback em Tempo Real:** Acompanhe o log de treinamento (baixa de dataset, progresso por época, acurácia final) através de um stream de dados (SSE).
* **Detecção de Hardware:** A interface identifica se o servidor possui uma GPU NVIDIA (CUDA) e habilita/desabilita opções de treino dinamicamente.
* **Pré-processamento Inteligente:** O sistema não apenas redimensiona a imagem, mas usa a biblioteca PIL para inverter, encontrar, cortar e centralizar o dígito desenhado, aumentando drasticamente a acurácia.
* **Feedback por Voz (TTS):** O navegador utiliza a API de Fala (Web Speech API) para "falar" o número previsto (ex: "Número 3") em português.

---

### 🧠 O Modelo (Deep Learning)

O cérebro por trás do projeto é uma **Rede Neural Convolucional (CNN)**, a arquitetura padrão-ouro para tarefas de visão computacional.

* **Arquitetura:** A CNN é construída em PyTorch e consiste em:
    1.  2 Camadas Convolucionais (`Conv2d`) com ativação `ReLU` (32 e 64 filtros).
    2.  1 Camada de `MaxPool2d` para reduzir a dimensionalidade.
    3.  Camadas de `Dropout` para regularização (prevenir overfitting).
    4.  2 Camadas Lineares (`Linear`) que resultam em 10 saídas (uma para cada dígito).
* **Dataset:** O modelo é treinado no clássico dataset **MNIST**.
* **Data Augmentation (Opção 1):** Para tornar o modelo mais robusto a diferentes estilos de escrita, aplicamos as seguintes transformações aleatórias durante o treino:
    * **Rotação** Aleatória (±10°)
    * **Translação** Aleatória (±10% no eixo X/Y)
    * **Shear / Cisalhamento** Aleatório (±10°)

---

### 🛠️ Pilha Tecnológica (Tech Stack)

Este projeto foi construído com as seguintes tecnologias:

* **Linguagem:** ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
* **IDE:** ![Visual Studio Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)
* **Backend (Servidor Web):** ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
* **Deep Learning:** ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
* **Processamento de Imagem:** `Pillow (PIL)`
* **Frontend (UI):**
    * ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
    * ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
    * ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) (com Canvas, Fetch API, EventSource e **Web Speech API**)

---

### 🚀 Como Executar Localmente

1.  **Clone o repositório:**
    ```bash
    git clone https://[URL-DO-SEU-REPOSITORIO-GIT]
    cd Hand-Decode AI
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
3.  **Instale as dependências:**
    ```bash
    # Para treino em CPU (ou GPU AMD/Intel)
    pip install torch torchvision flask pillow
    # Para treino em GPU (NVIDIA)
    # Verifique a versão correta do CUDA no site do PyTorch
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    pip install flask pillow
    ```
4.  **Execute o servidor web:**
    ```bash
    cd web_app
    python app_web.py
    ```

5.  **Acesse:** Abra seu navegador e vá para `http://127.0.0.1:5000`.



