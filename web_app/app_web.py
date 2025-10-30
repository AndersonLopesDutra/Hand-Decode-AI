import torch
import torch.nn as nn  # Módulos básicos do PyTorch (camadas, funções de perda)
import torch.nn.functional as F  # Funções de ativação (ReLU, Softmax)
import torch.optim as optim  # Otimizadores (Adam, SGD)
from torchvision import datasets, transforms  # Para baixar/transformar o dataset MNIST
from torch.utils.data import DataLoader  # Para carregar os dados em lotes
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from PIL import Image, ImageOps
import io
import base64  # Para decodificar a imagem enviada pelo JS
import re  # Para limpar o cabeçalho da imagem Base64
import sys  # Para sair do script se houver erro
import threading  # Para rodar o treino sem travar o app
import time  # Para o stream de logs
import os  # Para achar o caminho do modelo


# --- 1. Definição da Arquitetura da Rede (A CNN) ---
class RedeNeural(nn.Module):
    def __init__(self):
        super(RedeNeural, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Correção já aplicada
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)  # 10 saídas (dígitos 0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# --- 2. Variáveis Globais e Configurações ---
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '../modelo_salvo')
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_mnist.pth")
DATA_DIR = os.path.join(BASE_DIR, '../data')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

training_in_progress = False  # Flag para bloquear a previsão
training_log_stream = []  # Fila de mensagens de log

# Instancia o modelo na CPU (para inferência)
device = torch.device("cpu")
modelo = RedeNeural().to(device)


# --- 3. Funções de Carregamento e Pré-processamento ---

# Função helper para carregar o modelo na variável 'modelo' global
def carregar_modelo_global():
    global modelo
    try:
        modelo.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Modelo de produção carregado com sucesso: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Aviso: Arquivo do modelo não encontrado em {MODEL_PATH}. O modelo precisa ser treinado.")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
    modelo.eval()  # Coloca o modelo em modo de avaliação


# Transformações para o app (pré-processamento "inteligente")
transformador_app = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# --- 4. LÓGICA DE TREINAMENTO ---

# Função helper para enviar logs
def log_message(message):
    print(message)  # Imprime no console do servidor
    training_log_stream.append(message)  # Adiciona à fila para o navegador


# A função principal de treino que rodará em uma 'thread'
def run_training_job(device_choice):
    global training_in_progress
    global modelo

    training_in_progress = True  # BLOQUEIA a previsão

    try:
        log_message("--- INICIANDO TRABALHO DE TREINAMENTO ---")

        # 4.1. Seleção de Dispositivo
        device_train = None
        if device_choice == 'auto':
            device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choice == 'cpu':
            device_train = torch.device("cpu")
        elif device_choice == 'cuda':
            # Esta é a verificação que falhou (corretamente)
            if not torch.cuda.is_available():
                log_message("ERRO: 'cuda' solicitado, mas não está disponível. Usando 'cpu'.")
                device_train = torch.device("cpu")
            else:
                device_train = torch.device("cuda")
        log_message(f"Usando dispositivo de treino: {device_train}")

        # 4.2. Preparação de Dados (com Opção 1: Shear)
        log_message("Preparando Data Augmentation (Opção 1)...")
        transformador_teste = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transformador_treino = transforms.Compose([
            transforms.RandomRotation(degrees=10),  # Gira
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),  # Desloca e Inclina
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        log_message(f"Baixando e preparando dataset MNIST (na pasta {DATA_DIR})...")
        dataset_treino = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transformador_treino)
        dataset_teste = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transformador_teste)
        train_loader = DataLoader(dataset_treino, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(dataset_teste, batch_size=1000, shuffle=False)
        log_message("Dataset pronto.")

        # 4.3. O Treinamento
        modelo_treino = RedeNeural().to(device_train)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(modelo_treino.parameters(), lr=0.001)
        num_epocas = 5

        for epoca in range(num_epocas):
            log_message(f"\n--- Época {epoca + 1}/{num_epocas} ---")
            modelo_treino.train()
            total_batches = len(train_loader)

            for batch_idx, (dados, alvos) in enumerate(train_loader):
                dados, alvos = dados.to(device_train), alvos.to(device_train)
                optimizer.zero_grad()
                saida = modelo_treino(dados)
                perda = criterion(saida, alvos)
                perda.backward()
                optimizer.step()

                # Envia log a cada 10%
                if (batch_idx + 1) % (total_batches // 10) == 0:
                    progresso = (batch_idx + 1) / total_batches * 100
                    log_message(f"  Época {epoca + 1}: {progresso:.0f}%... Perda atual: {perda.item():.4f}")

        log_message("\nTreinamento concluído. Iniciando avaliação...")

        # 4.4. Avaliação
        modelo_treino.eval()
        perda_teste = 0.0
        acertos = 0
        with torch.no_grad():
            for dados, alvos in test_loader:
                dados, alvos = dados.to(device_train), alvos.to(device_train)
                saida = modelo_treino(dados)
                perda_teste += criterion(saida, alvos).item()
                predicao = saida.argmax(dim=1, keepdim=True)
                acertos += predicao.eq(alvos.view_as(predicao)).sum().item()

        perda_teste /= len(test_loader)
        acuracia_geral = 100. * acertos / len(test_loader.dataset)

        log_message("\n--- RELATÓRIO DE AVALIAÇÃO ---")
        log_message(f"  Acurácia Geral: {acuracia_geral:.2f}% ({int(acertos)}/{len(test_loader.dataset)})")
        log_message(f"  Perda média:    {perda_teste:.4f}")

        # 4.5. Salvamento e Recarregamento
        log_message(f"\nSalvando novo modelo em: {MODEL_PATH}")
        torch.save(modelo_treino.state_dict(), MODEL_PATH)

        log_message("Recarregando modelo de produção (na CPU)...")
        carregar_modelo_global()  # Recarrega o 'modelo' global (na CPU)

        log_message("\n--- TRABALHO DE TREINAMENTO CONCLUÍDO ---")

    except Exception as e:
        log_message(f"\n--- ERRO CRÍTICO NO TREINO: {e} ---")
    finally:
        training_in_progress = False  # DESBLOQUEIA a previsão


# --- 5. Configuração do Servidor Web (Flask) ---

app = Flask(__name__)  # Instancia o aplicativo Flask


# --- *** MUDANÇA: Rota principal agora verifica o CUDA *** ---
@app.route('/')
def index():
    # 'torch.cuda.is_available()' verifica se há uma GPU NVIDIA compatível
    cuda_available = torch.cuda.is_available()

    # Passa a variável 'cuda_available' (True/False) para o HTML
    return render_template('index.html', cuda_available=cuda_available)


@app.route('/predict', methods=['POST'])
def predict():
    if training_in_progress:
        return jsonify({
            'success': False,
            'error': "Modelo está treinando. Tente em alguns minutos."
        }), 503

    try:
        data = request.get_json()
        img_str = re.sub('^data:image/.+;base64,', '', data['image_data'])
        img_bytes = base64.b64decode(img_str)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('L')

        # Pré-processamento "Inteligente" (Centralização)
        imagem_invertida = ImageOps.invert(img_pil)
        bbox = imagem_invertida.getbbox()
        if not bbox:
            return jsonify({'success': True, 'prediction': 'N/A', 'confidence': 0})
        img_cortada = imagem_invertida.crop(bbox)
        l_maior = max(img_cortada.size)
        padding = 20
        img_quadrada = Image.new('L', (l_maior + padding * 2, l_maior + padding * 2), 0)
        pos_x = (img_quadrada.width - img_cortada.width) // 2
        pos_y = (img_quadrada.height - img_cortada.height) // 2
        img_quadrada.paste(img_cortada, (pos_x, pos_y))

        imagem_redimensionada = img_quadrada.resize((28, 28), Image.Resampling.LANCZOS)
        imagem_tensor = transformador_app(imagem_redimensionada)
        imagem_batch = imagem_tensor.unsqueeze(0).to(device)

        # Previsão
        with torch.no_grad():
            saida = modelo(imagem_batch)
            probabilidade = torch.exp(saida)
            previsao = saida.argmax(dim=1).item()
            confianca = probabilidade.max().item() * 100

        return jsonify({
            'success': True,
            'prediction': previsao,
            'confidence': confianca
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def start_train():
    global training_in_progress
    if training_in_progress:
        return jsonify({'message': 'Treinamento já em andamento.'}), 409

    data = request.get_json()
    device_choice = data.get('device', 'auto')
    training_log_stream.clear()

    # Inicia a função de treino em uma nova thread
    train_thread = threading.Thread(target=run_training_job, args=[device_choice])
    train_thread.start()

    return jsonify({'message': 'Treinamento iniciado!'})


@app.route('/train-status-stream')
def train_status_stream():
    # Esta função "generator" envia dados de log
    def generator():
        global training_in_progress
        global training_log_stream
        last_log_count = 0

        while training_in_progress:
            if len(training_log_stream) > last_log_count:
                new_messages = training_log_stream[last_log_count:]
                last_log_count = len(training_log_stream)
                for msg in new_messages:
                    yield f"data: {msg}\n\n"  # Formato SSE
            yield "event: ping\n\n"
            time.sleep(0.5)

        # Envia as últimas mensagens
        if len(training_log_stream) > last_log_count:
            for msg in training_log_stream[last_log_count:]:
                yield f"data: {msg}\n\n"

        yield "data: TRAINING_COMPLETE\n\n"  # Mensagem de fim

    return Response(stream_with_context(generator()), mimetype='text/event-stream')


# --- 6. Execução do Servidor ---
if __name__ == '__main__':
    carregar_modelo_global()  # Carrega o modelo ao iniciar
    app.run(debug=False, threaded=True, port=5000)