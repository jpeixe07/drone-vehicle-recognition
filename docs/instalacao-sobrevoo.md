# Sobrevoo — instalação e execução local

O Sobrevoo é a interface web do pipeline de reconhecimento de veículos. O mesmo processo Python serve a página e a API: não é necessário iniciar um frontend separado. O upload executa a inferência real com YOLO e EfficientNet e consulta o índice de referências.

## Requisitos

- Git: https://git-scm.com/downloads
- Python **3.11 de 64 bits**, recomendado neste guia. O pacote aceita Python >=3.10 e <3.13; **Python 3.13 não é compatível** com a configuração atual.
- Navegador atualizado. VS Code é opcional; os comandos podem ser executados no PowerShell.
- Internet para clonar o projeto, instalar dependências e baixar os pesos na preparação inicial.
- Espaço livre para o ambiente Python, bibliotecas e pesos (reserve alguns GB). O mínimo de RAM ainda não foi medido.
- GPU não é obrigatória: este guia usa CPU. O tempo de análise depende do computador e da imagem.

Execute cada comando separadamente e aguarde o término antes do próximo. Não copie o prefixo `PS ...>` nem as mensagens de saída do terminal. Se houver erro, resolva-o antes de continuar.

## 1. Conferir o Python no Windows

```powershell
py --list
```

Se o Python 3.11 não aparecer, instale-o pelo instalador oficial em https://www.python.org/downloads/windows/ ou pelo Windows Package Manager:

```powershell
winget install -e --id Python.Python.3.11
```

Feche e reabra o terminal após a instalação. Você pode manter outras versões instaladas.

```powershell
py -3.11 --version
```

## 2. Baixar a branch com a interface

```powershell
git clone --branch feat/sobrevoo-web https://github.com/jpeixe07/drone-vehicle-recognition.git
cd drone-vehicle-recognition
```

Se já clonou o repositório, entre na pasta existente e use:

```powershell
git fetch origin
git switch feat/sobrevoo-web
git pull --ff-only
```

Se o Git informar alterações locais ou divergências, não as descarte: revise-as antes de atualizar.

## 3. Criar o ambiente e instalar as dependências

Na pasta que contém `pyproject.toml`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
```

O caminho completo do Python da `.venv` dispensa ativar o ambiente e alterar a política de execução do PowerShell. A instalação pode demorar. Aguarde o comando terminar sem erros e o prompt retornar.

## 4. Preparar os pesos e criar o índice

```powershell
.\.venv\Scripts\python.exe -m vehicle_recognition prepare --device cpu
.\.venv\Scripts\python.exe -m vehicle_recognition index --device cpu
```

O primeiro comando prepara os pesos. O segundo calcula as representações da galeria e grava `artifacts/reference_indexes/efficientnet_b3.npz`. A galeria usada na validação inicial tinha 83 referências; esse número pode mudar com atualizações.

Não é necessário repetir esses passos a cada inicialização. Reconstrua o índice quando alterar a galeria ou a configuração do extrator. As fotos reais do conjunto de avaliação não vêm no clone e não são necessárias para analisar uma imagem própria. Credenciais do Roboflow só são necessárias se você optar por baixar o conjunto externo; veja o README.

## 5. Iniciar o Sobrevoo

```powershell
.\.venv\Scripts\python.exe -m vehicle_recognition.server --port 8000
```

Quando aparecer `Sobrevoo: http://127.0.0.1:8000`, abra **http://127.0.0.1:8000** no navegador.

Mantenha o terminal aberto durante o uso. É normal o prompt não retornar enquanto o servidor está rodando. Para encerrar, pressione **Ctrl + C**.

Nas próximas vezes, basta entrar na pasta do projeto e executar novamente o comando do servidor. Não é preciso reinstalar tudo.

## 6. Testar a análise

1. Envie ou arraste uma imagem JPG ou PNG de até 20 MiB.
2. Aguarde o processamento.
3. Se houver detecções, clique nas caixas para ver recorte, confiança da detecção, similaridade visual e candidatos.

Abrir a página confirma apenas que o servidor web iniciou. Receber o resultado de um upload sem erro confirma a execução da inferência.

“Modelo desconhecido” significa que o backend rejeitou a identificação pelo critério configurado; não significa falha de instalação. “Sem veículos detectados” é outro resultado possível. Similaridade visual e confiança da detecção são medidas diferentes. A interface não inventa categorias militar/civil.

As imagens em `docs/images/vehicles` são ilustrações para demonstração de execução. As imagens em `data/reference_dataset` fazem parte da galeria e não são uma avaliação independente. Para avaliar o uso real, utilize imagens aéreas completas externas à galeria.

## Problemas comuns

| Mensagem ou sintoma | Como resolver |
| --- | --- |
| `No suitable Python runtime found` | Instale Python 3.11 e confira `py --list`. Ter apenas Python 3.13 não basta. |
| `.venv` ou Python não encontrado | Confira se está na pasta do projeto e se concluiu a criação do ambiente. |
| `No module named vehicle_recognition` | Execute a instalação `-m pip install -e .` usando o Python da `.venv`, na raiz do repositório. |
| `invalid int value: '8000.\\.venv...'` | O comando foi colado duas vezes na mesma linha. Execute apenas uma cópia do comando de inicialização. |
| Porta 8000 ocupada / `WinError 10048` | Encerre a instância anterior ou inicie com `--port 8001` e abra http://127.0.0.1:8001. |
| Navegador não conecta | Confira se o terminal do servidor continua aberto e se está usando a porta indicada nele. |
| “Falha na análise” | Veja o erro no terminal. Confira a conclusão de `prepare` e `index`; envie o erro completo para diagnóstico. |
| Índice ausente ou incompatível | Execute `index --device cpu` com a mesma configuração de extrator usada pelo servidor. |
| Arquivo inválido ou muito grande | Use uma imagem válida JPG/PNG de até 20 MiB. |

## Linux (CPU)

Com Git, Python 3.11 e o módulo venv disponíveis, na raiz do repositório:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
.venv/bin/python -m vehicle_recognition prepare --device cpu
.venv/bin/python -m vehicle_recognition index --device cpu
.venv/bin/python -m vehicle_recognition.server --port 8000
```

## API e testes

`POST /api/predict` recebe os bytes da imagem diretamente no corpo da requisição, **não multipart/form-data**. Retorna o relatório real `predictions.json`, acrescido de `acceptance_threshold`. A própria interface faz essa chamada.

Respostas: 200 para análise concluída, 400 para imagem inválida, 413 para corpo vazio ou acima do limite e 503 para falha na inferência. Cada upload usa uma pasta temporária isolada. As requisições são processadas em série.

Para executar os testes no Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O servidor atual escuta somente em `127.0.0.1`: é destinado ao uso local. Hospedagem pública exige adaptação e configuração próprias. A CLI existente continua disponível; consulte o [README](../README.md) para outros comandos e configurações.
