# Transcrição de Vídeo

Este projeto transcreve vídeos usando a API da OpenAI e salva as transcrições no Google Drive.

## Configuração do Ambiente

1. **Criar ambiente virtual**:
```bash
python -m venv .venv
```

2. **Ativar o ambiente virtual**:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. **Instalar dependências**:
```bash
pip install -r requirements.txt
```

4. **Configurar variáveis de ambiente**:
- Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
```
OPENAI_API_KEY=sua_chave_api
GOOGLE_DRIVE_FOLDERS_TO_MONITOR=id_da_pasta1,id_da_pasta2
GOOGLE_DRIVE_RESULTS_FOLDER_ID=id_da_pasta_resultados
```

5. **Configurar credenciais do Google**:
- Coloque o arquivo `service-account-key.json` na raiz do projeto

## Executando o Projeto

1. **Ativar o ambiente virtual** (se ainda não estiver ativo):
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. **Executar o script**:
```bash
python video_transcriber.py
```

## Estrutura de Diretórios

- `.venv/` - Ambiente virtual Python
- `temp/` - Arquivos temporários
- `outputs/` - Arquivos de saída
- `.env` - Variáveis de ambiente
- `service-account-key.json` - Credenciais do Google
- `requirements.txt` - Dependências do projeto
- `video_transcriber.py` - Script principal

## Funcionalidades

- Monitora uma pasta específica no Google Drive
- Baixa vídeos automaticamente
- Divide vídeos grandes em partes de até 20MB
- Transcreve o áudio usando a API Whisper da OpenAI
- Gera resumos usando GPT-3.5
- Salva transcrições e resumos em uma pasta separada no Google Drive
- Organiza resultados por nome do cliente/vendedor

## Requisitos

- Python 3.9 ou superior
- FFmpeg instalado no sistema
- Conta no Google Cloud com API do Drive habilitada
- Conta na OpenAI com API key

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd [NOME_DO_DIRETÓRIO]
```

2. Instale o FFmpeg:
   - Windows: 
     ```powershell
     winget install ffmpeg
     ```
   - Linux:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - macOS:
     ```bash
     brew install ffmpeg
     ```

## Estrutura de Arquivos

```
.
├── video_transcriber.py    # Script principal
├── requirements.txt        # Dependências
├── .env                   # Configurações
├── service-account-key.json # Chave do Google Drive
├── temp/                  # Arquivos temporários
└── outputs/              # Arquivos de saída
```

## Formato dos Arquivos

Os vídeos devem seguir o formato:
```
Nome do Cliente - Descrição.mp4
```

Exemplo:
```
João Silva - Reunião de Vendas.mp4
```

## Limitações

- Tamanho máximo de cada parte do vídeo: 20MB
- Formato de vídeo suportado: MP4
- Necessário conexão com internet
- Necessário FFmpeg instalado

## Solução de Problemas

1. Erro de autenticação do Google Drive:
   - Verifique se o arquivo `service-account-key.json` está correto
   - Verifique se a API do Drive está habilitada

2. Erro de FFmpeg:
   - Verifique se o FFmpeg está instalado
   - Execute `ffmpeg -version` para confirmar

3. Erro de API da OpenAI:
   - Verifique se a chave API está correta no `.env`
   - Verifique se tem créditos suficientes

## Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. 