# 🧠 Reconhecimento de Expressões Faciais com CNNs e SHAP

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

**Reconhecimento de Expressões Faciais com Redes Neurais Convolucionais: Uma Abordagem Interpretável sobre a Base FER2013**

Este repositório contém o código-fonte e os resultados do projeto desenvolvido para a disciplina de **Sistemas Inteligentes 1** (UTFPR). O projeto consiste em um sistema de classificação de emoções faciais (raiva, felicidade, neutro, tristeza e surpresa) utilizando uma CNN compacta treinada do zero, combinada com uma análise de interpretabilidade via SHAP para entender as decisões do modelo.

---

## 📖 Visão Geral

A expressão facial é um dos canais mais ricos da comunicação humana não verbal. Este projeto visa dotar máquinas da capacidade de reconhecer essas expressões, abrindo caminho para sistemas mais empáticos e responsivos.

**Diferenciais do Projeto:**
- **Modelo Leve:** Apenas **3,5 milhões** de parâmetros (~13 MB), comparado a modelos como VGG16 (mais de 130 milhões).
- **Alta Performance:** Acurácia Top-1 de **72,13%**, superando a concordância média entre anotadores humanos (~65%).
- **Interpretabilidade:** Uso da técnica **SHAP** para visualizar quais regiões da face influenciam a decisão do modelo, transformando a "caixa-preta" em uma ferramenta compreensível.

---

## 📊 Base de Dados

Utilizamos a base [**FER2013**](https://www.kaggle.com/datasets/msambare/fer2013), organizada em 7 classes de emoção. Para reduzir ambiguidades e melhorar o foco, removemos as classes *disgust* e *fear*, trabalhando com as **5 classes** restantes:

| Classe       | Amostras (Treino) | Amostras (Teste) |
|--------------|-------------------|------------------|
| Angry        | 3.995                 | 958              |
| Happy        | 7.215                 | 1.774            |
| Neutral      | 4.965                 | 1.233            |
| Sad          | 4.830                 | 1.247            |
| Surprise     | 3.171                 | 831              |

---

## 🏗️ Arquitetura da Rede

A CNN é composta por três blocos convolucionais seguidos por uma camada densa. Cada bloco utiliza Batch Normalization e Dropout para regularização, garantindo estabilidade no treinamento e prevenindo overfitting.

| Bloco       | Camadas                                      | Saída        |
|-------------|----------------------------------------------|--------------|
| Entrada     | Imagem 48×48×1 (escala de cinza)             | 48×48×1      |
| Conv Block 1| 2×Conv 3×3 (64) + BN + MaxPool + Dropout(0.25) | 24×24×64    |
| Conv Block 2| 2×Conv 3×3 (128) + BN + MaxPool + Dropout(0.30)| 12×12×128   |
| Conv Block 3| 2×Conv 3×3 (256) + BN + MaxPool + Dropout(0.40)| 6×6×256     |
| Densa       | Flatten + Dense 256 + BN + Dropout(0.50)     | 256          |
| Saída       | Dense 5 + Softmax                            | 5 classes    |

---

## 📈 Resultados

### Métricas Globais
- **Top-1 Accuracy:** **72,13%**
- **Top-2 Accuracy:** **88,66%**

### Métricas por Classe

| Classe   | Precisão | Revocação | F1-Score | Suporte |
|----------|----------|-----------|----------|---------|
| angry    | 0.649    | 0.619     | 0.634    | 958     |
| happy    | 0.864    | 0.856     | **0.860**| 1.774   |
| neutral  | 0.601    | 0.674     | 0.635    | 1.233   |
| sad      | 0.626    | 0.574     | **0.599**| 1.247   |
| surprise | 0.830    | 0.844     | 0.837    | 831     |

### Análise de Erros
A matriz de confusão revela um **"triângulo de confusão"** entre as classes *angry*, *sad* e *neutral*, indicando que a similaridade visual entre essas emoções de valência negativa é o principal fator limitante do desempenho.

---

## 🔍 Interpretabilidade com SHAP

A principal inovação deste projeto é a aplicação da técnica **SHAP (SHapley Additive exPlanations)** para interpretar as decisões do modelo.

**Metodologia:**
- Utilizamos o `GradientExplainer` com um *background* de 100 imagens do conjunto de treino.
- A análise foi focada nas amostras classificadas incorretamente.

**Principais Descobertas:**
- Para a classe *angry* classificada erroneamente como *sad* ou *neutral*, os mapas SHAP mostraram que o modelo foca nas mesmas regiões (boca e sobrancelhas) para ambas as classes.
- Isso confirma que os erros não são aleatórios ou baseados em artefatos de imagem, mas sim decorrentes da **ambiguidade visual inerente** às expressões faciais.

<div align="center">
  <img width="1458" height="511" alt="shap_exemplo" src="https://github.com/user-attachments/assets/14b7711b-2cd9-4c81-ad8c-0baa13987887" />
  <br>
  <em>Figura: Mapa SHAP para uma amostra de 'angry' classificada como 'sad'. A sobreposição das regiões em destaque evidencia a ambiguidade visual.</em>
</div>

---

## 🚀 Como Executar

Este projeto foi desenvolvido em **Google Colab** para facilitar a reprodução. Siga os passos abaixo:

1.  **Clone o repositório** ou baixe o notebook (`ProjetoSI.ipynb`).
2.  Faça o upload do notebook para o **Google Colab**.
3.  Coloque o arquivo `archive.zip` (FER2013) em uma pasta no seu Google Drive (ex: `MeuDrive/datasets/`).
4.  Execute as células sequencialmente.

**Dependências Principais:**
- Python 3.9+
- TensorFlow 2.15+
- OpenCV
- SHAP
- Matplotlib, Seaborn, Pandas

---

## 🔮 Trabalhos Futuros

- Realizar um ajuste mais fino da taxa de aprendizado e aumentar o número de épocas para verificar se a tendência de queda da perda de validação se mantém.
- Incluir as classes *disgust* e *fear* utilizando técnicas mais robustas de balanceamento.
- Aplicar *Transfer Learning* (e.g., MobileNetV2) para melhorar o desempenho nas classes de menor acurácia.
- Adaptar o sistema para inferência em **tempo real** a partir de vídeo, integrando a detecção facial já prototipada.

---

## 👥 Autores

- **Caio Vinícius Maciel Delgado** - [LinkedIn](https://www.linkedin.com/in/caio-vin%C3%ADcius-maciel-delgado-55730b31b/) | [GitHub](https://github.com/CaioVDelgado)
- **Guilherme Pança Franco** - [LinkedIn](https://www.linkedin.com/in/guilherme-panca-franco-60a80a360/?skipRedirect=true) | [GitHub](https://github.com/Gu1-Fr4nc0)
- **João Pedro Garcia Bronharo** - [LinkedIn](https://www.linkedin.com/in/jo%C3%A3o-pedro-garcia-210b0a321/) | [GitHub](https://github.com/seu-github)

*Projeto desenvolvido para a disciplina de **Sistemas Inteligentes 1** - Engenharia de Computação - Universidade Tecnológica Federal do Paraná (UTFPR).*

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 📚 Referências

- Goodfellow, I. J., et al. (2013). *Challenges in representation learning: A report on three machine learning contests*.
- Pramerdorfer, C., & Kampel, M. (2016). *Facial Expression Recognition using Convolutional Neural Networks: State of the Art*.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP).

---

💡 **Dica:** Não se esqueça de substituir os links do `assets/shap_example.png` pela imagem real que você gerou, e os placeholders de perfil para os seus links reais do LinkedIn/GitHub.
