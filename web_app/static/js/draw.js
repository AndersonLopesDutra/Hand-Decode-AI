// Espera a página carregar inteira antes de rodar
window.addEventListener('load', () => {

    // --- 1. Seleciona os Elementos (TESTE) ---
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d'); // O "pincel" 2D
    const btnPredict = document.getElementById('btnPredict');
    const btnClear = document.getElementById('btnClear');
    const resultText = document.getElementById('resultText');
    const canvasOverlay = document.getElementById('canvas-overlay'); // Camada de bloqueio

    // --- 2. Seleciona os Elementos (TREINO) ---
    const btnTrain = document.getElementById('btnTrain');
    const deviceSelector = document.getElementById('deviceSelector');
    const trainLog = document.getElementById('train-log');

    // --- 3. Seleciona Elementos do Modal "Leia-me" ---
    const btnReadme = document.getElementById('btnReadme');
    const modalOverlay = document.getElementById('readmeModal');
    const modalClose = document.getElementById('modalClose');

    // Variáveis de estado do canvas
    let desenhando = false;
    let ultimoX = 0;
    let ultimoY = 0;

    // --- 4. Função de Text-to-Speech (TTS) ---
    function falarTexto(texto) {
        // Verifica se o navegador suporta a Web Speech API
        if ('speechSynthesis' in window) {
            // Limpa a fila de falas (se houver alguma pendente)
            window.speechSynthesis.cancel();

            // Cria um novo objeto "fala"
            const utterance = new SpeechSynthesisUtterance(texto);

            // Define o idioma para Português do Brasil
            utterance.lang = 'pt-BR';

            // Opcional: Ajustar a velocidade e o tom
            utterance.rate = 1.0; // Velocidade (0.1 a 10)
            utterance.pitch = 1.0; // Tom (0 a 2)

            // Manda o navegador falar
            window.speechSynthesis.speak(utterance);
        } else {
            console.log("Seu navegador não suporta a Web Speech API.");
        }
    }

    // --- 5. Funções do Canvas (Desenho) ---

    function limparCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultText.innerHTML = "Desenhe um número (0-9) acima";
        resultText.className = '';
    }
    limparCanvas(); // Limpa na inicialização

    function comecarDesenho(e) {
        desenhando = true;
        [ultimoX, ultimoY] = [e.offsetX, e.offsetY];
    }

    function desenhar(e) {
        if (!desenhando) return;
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 12;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(ultimoX, ultimoY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [ultimoX, ultimoY] = [e.offsetX, e.offsetY];
    }

    function pararDesenho() {
        desenhando = false;
    }

    // Eventos do mouse no canvas
    canvas.addEventListener('mousedown', comecarDesenho);
    canvas.addEventListener('mousemove', desenhar);
    canvas.addEventListener('mouseup', pararDesenho);
    canvas.addEventListener('mouseout', pararDesenho);

    // Botão de Limpar
    btnClear.addEventListener('click', limparCanvas);

    // --- 6. Função de Previsão (TESTE) ---

    async function handlePredictClick() {
        btnPredict.disabled = true;
        btnPredict.innerHTML = "Prevendo...";
        btnPredict.classList.add('is-loading');

        const imageData = canvas.toDataURL('image/png');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: imageData })
            });

            const data = await response.json();

            if (data.success) {
                const confianca = data.confidence.toFixed(2);

                // Texto para exibir na tela
                const textoResultado = `Previsão: ${data.prediction} (Confiança: ${confianca}%)`;
                resultText.innerHTML = textoResultado;
                resultText.className = 'success';

                // --- *** MUDANÇA: Texto da fala alterado *** ---
                // Adicionamos "Número " antes do dígito
                const textoParaFalar = `Número ${data.prediction}`;
                falarTexto(textoParaFalar);
                // --- Fim da mudança ---

            } else {
                resultText.innerHTML = data.error || "Erro na previsão.";
                resultText.className = 'error';
                falarTexto(data.error || "Erro na previsão.");
            }
        } catch (error) {
            resultText.innerHTML = "Erro de conexão com o servidor.";
            resultText.className = 'error';
            console.error("Erro de Rede:", error);
            falarTexto("Erro de conexão");
        }

        // Reativa o botão
        btnPredict.disabled = false;
        btnPredict.innerHTML = "Realizar Previsão";
        btnPredict.classList.remove('is-loading');
    }
    btnPredict.addEventListener('click', handlePredictClick);


    // --- 7. Função de Treinamento (TREINO) ---

    let sseConnection = null; // Guarda a conexão de stream

    async function handleTrainClick() {
        btnTrain.disabled = true;
        btnTrain.innerHTML = "Treinamento em Andamento...";

        // BLOQUEIA A INTERFACE DE TESTE
        canvasOverlay.classList.add('disabled');
        btnPredict.disabled = true;
        btnClear.disabled = true;

        trainLog.textContent = 'Iniciando conexão com o servidor...\n';
        const device = deviceSelector.value;

        try {
            // 1. Envia o comando para iniciar o treino
            const trainResponse = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device: device })
            });

            if (!trainResponse.ok) {
                const errorData = await trainResponse.json();
                throw new Error(errorData.message || "Não foi possível iniciar o treino.");
            }

            trainLog.textContent += 'Treinamento iniciado. Aguardando logs...\n';

            // 2. Abre a conexão de stream (SSE)
            sseConnection = new EventSource('/train-status-stream');

            // 3. O que fazer quando uma mensagem (log) chega
            sseConnection.onmessage = (event) => {
                if (event.data === "TRAINING_COMPLETE") {
                    trainLog.textContent += "\n--- MODELO TREINADO E RECARREGADO ---";
                    trainLog.scrollTop = trainLog.scrollHeight;
                    stopTrainingStream(); // Encerra tudo
                } else {
                    trainLog.textContent += event.data + '\n';
                    trainLog.scrollTop = trainLog.scrollHeight;
                }
            };

            // 4. O que fazer se o stream falhar
            sseConnection.onerror = (error) => {
                trainLog.textContent += "\n--- ERRO NO STREAM DE LOGS. Conexão perdida. ---";
                console.error("Erro no EventSource:", error);
                stopTrainingStream();
            };

        } catch (error) {
            trainLog.textContent += `\nErro ao iniciar: ${error.message}`;
            stopTrainingStream(); // Desbloqueia a UI se falhar
        }
    }

    // Função para limpar e desbloquear tudo
    function stopTrainingStream() {
        if (sseConnection) {
            sseConnection.close(); // Fecha a conexão
            sseConnection = null;
        }

        // Desbloqueia a UI
        btnTrain.disabled = false;
        btnTrain.innerHTML = "Iniciar Treinamento";
        canvasOverlay.classList.remove('disabled');
        btnPredict.disabled = false;
        btnClear.disabled = false;
    }

    btnTrain.addEventListener('click', handleTrainClick);


    // --- 8. Lógica do Modal "Leia-me" ---

    function abrirModal() {
        modalOverlay.classList.remove('hidden'); // Mostra o modal
    }

    function fecharModal() {
        modalOverlay.classList.add('hidden'); // Esconde o modal
    }

    // Conecta os botões às funções
    btnReadme.addEventListener('click', abrirModal);
    modalClose.addEventListener('click', fecharModal);

    // Fecha o modal se o usuário clicar no fundo (overlay)
    modalOverlay.addEventListener('click', (event) => {
        if (event.target === modalOverlay) {
            fecharModal();
        }
    });

});