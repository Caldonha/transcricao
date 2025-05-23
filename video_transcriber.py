import os
import tempfile
import shutil
import json
import math
from pathlib import Path
from typing import List, Set, Dict
import time
from tqdm import tqdm
import re
import subprocess
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import datetime

# Imports do Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

from moviepy.editor import VideoFileClip, AudioFileClip
import openai
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações
SCOPES = ['https://www.googleapis.com/auth/drive']  # Escopo completo para criar pastas

# Lista de IDs de pastas a monitorar no Google Drive
FOLDERS_TO_MONITOR = os.getenv('GOOGLE_DRIVE_FOLDERS_TO_MONITOR', '').split(',')
RESULTS_FOLDER_ID = os.getenv('GOOGLE_DRIVE_RESULTS_FOLDER_ID')  # Pasta para resultados

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB em bytes
CHUNK_SIZE_MB = 20  # Tamanho de cada pedaço em MB
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024  # Convertendo MB para bytes
TEMP_DIR = Path('temp')
OUTPUT_DIR = Path('outputs')
PROCESSED_IDS_FILE = Path('processed_video_ids.json')
TOTAL_COST_FILE = Path('total_cost.json')  # Arquivo para armazenar o custo total acumulado

# Configurar OpenAI
openai.api_key = OPENAI_API_KEY

# Classe para gerenciar o custo total acumulado
class TotalCostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.last_updated = ""
        self.load()
    
    def load(self):
        """Carrega o custo total acumulado do arquivo"""
        if TOTAL_COST_FILE.exists():
            try:
                with open(TOTAL_COST_FILE, 'r') as f:
                    data = json.load(f)
                    self.total_cost = data.get('total_cost', 0.0)
                    self.last_updated = data.get('last_updated', "")
                print(f"💰 Custo total acumulado carregado: ${self.total_cost:.4f}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar custo total: {str(e)}")
    
    def save(self):
        """Salva o custo total acumulado no arquivo"""
        try:
            with open(TOTAL_COST_FILE, 'w') as f:
                data = {
                    'total_cost': self.total_cost,
                    'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                json.dump(data, f)
        except Exception as e:
            print(f"⚠️ Erro ao salvar custo total: {str(e)}")
    
    def add_cost(self, cost):
        """Adiciona um novo custo ao total acumulado"""
        self.total_cost += cost
        self.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()
    
    def get_total(self):
        """Retorna o custo total acumulado"""
        return self.total_cost
    
    def get_summary(self):
        """Retorna um resumo do custo total"""
        return {
            'total_cost': self.total_cost,
            'last_updated': self.last_updated
        }

class TranscriptionCounter:
    def __init__(self):
        self.total_seconds = 0
        self.total_files = 0
        
    def add_transcription(self, duration_seconds):
        self.total_seconds += duration_seconds
        self.total_files += 1
        
    def get_minutes(self):
        return self.total_seconds / 60
        
    def estimate_cost(self):
        # Preço por minuto do GPT-4o Transcribe conforme documentação oficial mais recente
        # Atualizado para $0.006 por minuto
        GPT4O_TRANSCRIBE_PRICE_PER_MINUTE = 0.006  # $0.006 por minuto
        return self.get_minutes() * GPT4O_TRANSCRIBE_PRICE_PER_MINUTE
        
    def get_summary(self):
        return {
            'total_minutes': self.get_minutes(),
            'total_files': self.total_files,
            'estimated_cost': self.estimate_cost()
        }
        
    def reset(self):
        """Reinicia o contador para um novo vídeo"""
        self.total_seconds = 0
        self.total_files = 0

class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    def add_usage(self, response):
        if hasattr(response, 'usage'):
            self.total_tokens += response.usage.total_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            
    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'estimated_cost': self.estimate_cost()
        }
        
    def estimate_cost(self):
        # Preços atualizados para o GPT-4.1 conforme documentação mais recente (2025)
        # $0.01 por 1K tokens de entrada e $0.03 por 1K tokens de saída
        GPT41_PROMPT_PRICE = 0.01  # $0.01 por 1K tokens de prompt
        GPT41_COMPLETION_PRICE = 0.03  # $0.03 por 1K tokens de completion
        
        prompt_cost = (self.prompt_tokens / 1000) * GPT41_PROMPT_PRICE
        completion_cost = (self.completion_tokens / 1000) * GPT41_COMPLETION_PRICE
        
        return prompt_cost + completion_cost
        
    def reset(self):
        """Reinicia o contador para um novo vídeo"""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

# Criar instâncias globais dos contadores
token_counter = TokenCounter()
transcription_counter = TranscriptionCounter()
total_cost_tracker = TotalCostTracker()

def authenticate_google_drive() -> build:
    """Autentica com o Google Drive usando Service Account."""
    try:
        # Verificar se estamos no Railway
        if os.getenv('RAILWAY_ENVIRONMENT'):
            # Usar credenciais do Railway
            credentials_json = os.getenv('GOOGLE_CREDENTIALS')
            if not credentials_json:
                raise ValueError("GOOGLE_CREDENTIALS não encontrado nas variáveis de ambiente")
            
            # Criar arquivo temporário com as credenciais
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(credentials_json)
                temp_file_path = temp_file.name
            
            credentials = service_account.Credentials.from_service_account_file(
                temp_file_path,
                scopes=SCOPES
            )
            
            # Remover arquivo temporário
            os.unlink(temp_file_path)
        else:
            # Usar arquivo local
            credentials = service_account.Credentials.from_service_account_file(
                'service-account-key.json',
                scopes=SCOPES
            )
        
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        print(f"Erro ao autenticar com o Google Drive: {str(e)}")
        raise

def load_processed_ids() -> Set[str]:
    """Carrega os IDs dos vídeos já processados."""
    if PROCESSED_IDS_FILE.exists():
        with open(PROCESSED_IDS_FILE, 'r') as f:
            try:
                # Carregar como lista JSON e converter para conjunto
                id_list = json.load(f)
                return set(id_list)
            except json.JSONDecodeError as e:
                print(f"⚠️ Erro ao decodificar arquivo de IDs processados: {str(e)}")
                return set()
    return set()

def save_processed_ids(processed_ids: Set[str]):
    """Salva os IDs dos vídeos já processados como um array JSON."""
    try:
        # Converter conjunto para lista ordenada antes de salvar
        id_list = sorted(list(processed_ids))
        
        with open(PROCESSED_IDS_FILE, 'w') as f:
            # Formatar JSON com indentação para maior legibilidade
            json.dump(id_list, f, indent=2)
            
        print(f"✅ {len(processed_ids)} IDs de vídeos processados salvos com sucesso")
    except Exception as e:
        print(f"❌ Erro ao salvar IDs de vídeos processados: {str(e)}")

def get_video_files(service: build, folder_id: str) -> List[dict]:
    """Obtém lista de arquivos de vídeo na pasta especificada."""
    query = f"'{folder_id}' in parents and mimeType contains 'video/'"
    results = service.files().list(q=query, fields="files(id, name, size)").execute()
    return results.get('files', [])

def download_file(service: build, file_id: str, file_name: str) -> str:
    """Baixa um arquivo do Google Drive para uma pasta temporária."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    
    # Limpar o nome do arquivo para evitar problemas com caracteres especiais
    safe_filename = ''.join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in file_name)
    print(f"💾 Nome original: {file_name}")
    print(f"💾 Nome seguro para download: {safe_filename}")
    
    with tqdm(total=100, desc=f"Baixando {file_name}") as pbar:
        done = False
        while not done:
            status, done = downloader.next_chunk()
            pbar.update(int(status.progress() * 100))
    
    temp_path = TEMP_DIR / safe_filename
    with open(temp_path, 'wb') as f:
        f.write(fh.getvalue())
    
    return str(temp_path)

def create_folder_if_not_exists(service: build, folder_name: str, parent_id: str) -> str:
    """Cria uma pasta no Google Drive se ela não existir e retorna o ID."""
    # Verificar se a pasta já existe
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    
    # Se a pasta já existe, retorna o ID dela
    if response['files']:
        return response['files'][0]['id']
    
    # Se não existe, cria a pasta
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')

def upload_file_to_drive(service: build, file_path: str, file_name: str, folder_id: str) -> str:
    """Faz upload de um arquivo para o Google Drive e retorna o ID."""
    try:
        print(f"🔄 Iniciando upload do arquivo: {file_name}")
        print(f"🔄 Caminho do arquivo: {file_path}")
        print(f"🔄 Pasta de destino ID: {folder_id}")
        
        # Verificar se o arquivo existe e tem conteúdo
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"Arquivo vazio: {file_path}")
            
        print(f"✅ Arquivo verificado: {file_size / 1024:.2f} KB")
        
        # Verificar se a pasta existe e obter seu nome
        try:
            folder_info = service.files().get(fileId=folder_id, fields="name").execute()
            folder_name = folder_info.get("name", "Pasta desconhecida")
            print(f"✅ Pasta de destino verificada: {folder_name} (ID: {folder_id})")
        except Exception as e:
            print(f"⚠️ Erro ao verificar pasta: {str(e)}")
            raise ValueError(f"Pasta não encontrada: {folder_id}")
        
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,parents'
        ).execute()
        
        file_id = file.get('id')
        
        # Verificar se o arquivo foi criado na pasta correta
        parents = file.get('parents', [])
        if folder_id not in parents:
            print(f"⚠️ AVISO: O arquivo pode ter sido criado em uma pasta diferente da esperada")
            print(f"⚠️ Pasta esperada: {folder_id}")
            print(f"⚠️ Pastas do arquivo: {parents}")
        
        print(f"✅ Upload concluído: {file_name} (ID: {file_id}) na pasta {folder_name}")
        return file_id
        
    except Exception as e:
        print(f"❌ Erro no upload do arquivo {file_name}: {str(e)}")
        raise

def get_video_size_duration(video_path: str) -> tuple:
    """Retorna o tamanho do arquivo em bytes e a duração em segundos."""
    file_size = os.path.getsize(video_path)
    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()
    return file_size, duration

def estimate_time_for_size(duration: float, total_size: int, target_size: int) -> float:
    """Estima a duração de tempo para um tamanho específico de arquivo."""
    if total_size == 0:
        return 0
    return (duration * target_size) / total_size

def split_video_by_size(video_path: str, chunk_duration: int = 600) -> List[str]:
    """Divide um vídeo em partes de no máximo 15MB cada."""
    print(f"🔄 Iniciando divisão do vídeo: {video_path}")
    
    # Verificar se o arquivo existe
    if not os.path.exists(video_path):
        print(f"❌ Arquivo de vídeo não encontrado: {video_path}")
        return []
    
    try:
        # Obter duração total e tamanho
        video = VideoFileClip(video_path)
        total_duration = video.duration
        total_size = os.path.getsize(video_path)
        
        print(f"⏱️ Duração total: {total_duration:.2f} segundos")
        print(f"📊 Tamanho total: {total_size / (1024*1024):.2f}MB")
        
        # Se o vídeo for menor que 15MB, retornar o próprio arquivo
        if total_size <= 15 * 1024 * 1024:  # 15MB em bytes
            print("✅ Vídeo menor que 15MB, não precisa dividir")
            return [video_path]
        
        # Calcular quantos chunks precisamos baseado no tamanho
        num_chunks = math.ceil(total_size / (15 * 1024 * 1024))  # Dividir em partes de 15MB
        print(f"📝 Dividindo em {num_chunks} partes de no máximo 15MB")
        
        parts = []
        
        for i in tqdm(range(num_chunks), desc="Dividindo vídeo"):
            # Calcular duração aproximada para cada parte
            part_duration = total_duration / num_chunks
            start_time = i * part_duration
            end_time = min((i + 1) * part_duration, total_duration)
            
            part_path = TEMP_DIR / f"part_{i+1}_of_{num_chunks}.mp4"
            print(f"\n🔄 Criando parte {i+1}: {part_path}")
            
            try:
                # Extrair subclip
                subclip = video.subclip(start_time, end_time)
                
                # Configurar parâmetros de compressão para garantir tamanho máximo
                subclip.write_videofile(
                    str(part_path),
                    codec='libx264',
                    audio_codec='aac',
                    bitrate='1000k',  # Taxa de bits mais baixa para garantir tamanho menor
                    preset='ultrafast',  # Compressão mais rápida
                    threads=4,
                    ffmpeg_params=[
                        '-maxrate', '1000k',
                        '-bufsize', '2000k',
                        '-crf', '28'  # Taxa de compressão mais alta
                    ]
                )
                
                # Verificar se o arquivo foi criado
                if not os.path.exists(part_path):
                    print(f"⚠️ Parte {i+1} não foi criada: {part_path}")
                    continue
                
                # Verificar tamanho real
                part_size = os.path.getsize(part_path)
                print(f"📊 Tamanho da parte {i+1}: {part_size / (1024*1024):.2f}MB")
                
                # Se ainda estiver muito grande, tentar comprimir mais
                if part_size > 15 * 1024 * 1024:  # 15MB em bytes
                    print(f"⚠️ Parte {i+1} ainda está muito grande, comprimindo mais...")
                    compressed_path = TEMP_DIR / f"compressed_part_{i+1}_of_{num_chunks}.mp4"
                    
                    try:
                        # Comprimir mais usando MoviePy
                        subclip.write_videofile(
                            str(compressed_path),
                            codec='libx264',
                            audio_codec='aac',
                            bitrate='500k',  # Taxa de bits ainda mais baixa
                            preset='ultrafast',
                            threads=4,
                            ffmpeg_params=[
                                '-maxrate', '500k',
                                '-bufsize', '1000k',
                                '-crf', '32'  # Taxa de compressão ainda mais alta
                            ]
                        )
                        
                        # Verificar se o arquivo comprimido foi criado
                        if not os.path.exists(compressed_path):
                            print(f"⚠️ Arquivo comprimido não foi criado: {compressed_path}")
                            parts.append(part_path)
                            continue
                        
                        # Remover arquivo original e usar o comprimido
                        if os.path.exists(part_path):
                            os.remove(part_path)
                        part_path = compressed_path
                        
                        # Verificar tamanho final
                        part_size = os.path.getsize(part_path)
                        print(f"✅ Parte {i+1} comprimida: {part_size / (1024*1024):.2f}MB")
                        
                        # Se ainda estiver muito grande, dividir em mais partes
                        if part_size > 15 * 1024 * 1024:
                            print(f"⚠️ Parte {i+1} ainda está muito grande, dividindo em subpartes...")
                            subparts = split_video_by_size(str(part_path), chunk_duration // 2)
                            parts.extend(subparts)
                            continue
                            
                    except Exception as e:
                        print(f"⚠️ Erro ao comprimir parte {i+1}: {e}")
                        # Continuar com o arquivo original se a compressão falhar
                
                parts.append(str(part_path))
                print(f"✅ Parte {i+1} criada com sucesso!")
                
            except Exception as e:
                print(f"❌ Erro ao criar parte {i+1}: {str(e)}")
        
        video.close()
        print(f"\n✅ Total de partes criadas: {len(parts)}")
        return parts
        
    except Exception as e:
        print(f"❌ Erro ao dividir vídeo: {str(e)}")
        return []

def get_audio_duration(audio_path: str) -> float:
    """Obtém a duração de um arquivo de áudio em segundos."""
    cmd = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ], capture_output=True, text=True, check=True)
    
    return float(cmd.stdout.strip())

def split_audio(audio_path: str, chunk_duration: int = 600) -> List[str]:
    """Divide um arquivo de áudio em partes de duração específica."""
    print(f"🔄 Iniciando divisão do áudio: {audio_path}")
    
    # Verificar se o arquivo de áudio existe
    if not os.path.exists(audio_path):
        print(f"❌ Arquivo de áudio não encontrado: {audio_path}")
        return []
    
    try:
        duration = get_audio_duration(audio_path)
        print(f"⏱️ Duração total do áudio: {duration:.2f} segundos")
        
        num_chunks = math.ceil(duration / chunk_duration)
        print(f"📝 Dividindo em {num_chunks} partes de {chunk_duration} segundos")
        
        chunks = []
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            chunk_path = TEMP_DIR / f"chunk_{i}.wav"
            print(f"🔄 Criando chunk {i+1}: {chunk_path}")
            
            try:
                # Usar MoviePy para extrair o trecho de áudio
                audio = AudioFileClip(audio_path)
                chunk = audio.subclip(start_time, end_time)
                chunk.write_audiofile(
                    str(chunk_path),
                    codec='pcm_s16le',
                    fps=16000,
                    nbytes=2
                )
                chunk.close()
                audio.close()
                
                # Verificar se o chunk foi criado
                if not os.path.exists(chunk_path):
                    print(f"⚠️ Chunk {i+1} não foi criado: {chunk_path}")
                    continue
                    
                print(f"✅ Chunk {i+1} criado: {chunk_path}")
                chunks.append(str(chunk_path))
                
            except Exception as e:
                print(f"❌ Erro ao criar chunk {i+1}: {str(e)}")
        
        print(f"✅ Total de chunks criados: {len(chunks)}")
        return chunks
        
    except Exception as e:
        print(f"❌ Erro ao dividir áudio: {str(e)}")
        return []

def transcribe_audio(video_path: str) -> str:
    """Transcreve um arquivo de vídeo usando a API da OpenAI."""
    print(f"📝 Iniciando transcrição do arquivo: {video_path}")
    
    try:
        # Verificar se o vídeo existe antes de tentar processar
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"O arquivo de vídeo não existe: {video_path}")
        
        print(f"✅ Arquivo de vídeo encontrado: {video_path}")
        
        # Obter duração do vídeo
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        
        print(f"⏱️ Duração do vídeo: {duration:.2f} segundos")
        
        # Registrar duração para cálculo de custo
        transcription_counter.add_transcription(duration)
        
        chunk_duration = 600  # 10 minutos em segundos
        
        # Se o vídeo for menor que 10 minutos, transcreve direto
        if duration <= chunk_duration:
            print("📝 Transcrevendo vídeo completo (menor que 10 minutos)...")
            with open(video_path, 'rb') as video_file:
                transcript = openai.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=video_file
                )
            print(f"✅ Transcrição concluída: {len(transcript.text)} caracteres")
            return transcript.text
        
        # Dividir o vídeo em partes
        print("🔄 Dividindo vídeo em partes...")
        chunks = split_video_by_size(video_path, chunk_duration)
        print(f"✅ Vídeo dividido em {len(chunks)} partes")
        transcripts = []
        
        for i, chunk_path in enumerate(chunks):
            print(f"📝 Transcrevendo parte {i+1} de {len(chunks)}...")
            
            # Verificar se o chunk existe
            if not os.path.exists(chunk_path):
                print(f"⚠️ Chunk {i+1} não encontrado: {chunk_path}")
                transcripts.append(f"[Chunk {i+1} não encontrado]")
                continue
                
            try:
                # Transcrever o chunk
                with open(chunk_path, 'rb') as video_file:
                    transcript = openai.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=video_file
                    )
                transcripts.append(transcript.text)
                print(f"✅ Parte {i+1} transcrita com sucesso! ({len(transcript.text)} caracteres)")
                
            except Exception as e:
                print(f"❌ Erro ao transcrever parte {i+1}: {str(e)}")
                # Se der erro, tenta dividir em partes ainda menores
                if "token limit" in str(e).lower():
                    print("⚠️ Limite de tokens atingido. Dividindo em partes menores...")
                    smaller_chunks = split_video_by_size(chunk_path, chunk_duration // 2)
                    
                    for j, small_chunk in enumerate(smaller_chunks):
                        # Verificar se o small_chunk existe
                        if not os.path.exists(small_chunk):
                            print(f"⚠️ Subchunk {j+1} não encontrado: {small_chunk}")
                            transcripts.append(f"[Subchunk {j+1} não encontrado]")
                            continue
                            
                        try:
                            with open(small_chunk, 'rb') as video_file:
                                transcript = openai.audio.transcriptions.create(
                                    model="gpt-4o-transcribe",
                                    file=video_file
                                )
                            transcripts.append(transcript.text)
                            print(f"✅ Subparte {j+1} transcrita com sucesso! ({len(transcript.text)} caracteres)")
                        except Exception as e2:
                            print(f"❌ Erro ao transcrever subparte {j+1}: {str(e2)}")
                            transcripts.append(f"[Erro na transcrição da parte {i+1}.{j+1}]")
                        
                        # Limpar arquivo temporário
                        if os.path.exists(small_chunk):
                            os.remove(small_chunk)
                else:
                    transcripts.append(f"[Erro na transcrição da parte {i+1}]")
            
            # Limpar arquivo temporário
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
                print(f"🧹 Parte {i+1} removida")
        
        # Juntar todas as transcrições
        print(f"\n🔄 Juntando {len(transcripts)} transcrições...")
        
        # Verificar se todas as transcrições foram obtidas
        total_parts = len(chunks)
        successful_parts = sum(1 for t in transcripts if not t.startswith("["))
        if successful_parts < total_parts:
            print(f"⚠️ Atenção: Apenas {successful_parts} de {total_parts} partes foram transcritas com sucesso.")
        
        full_transcript = " ".join(transcripts)
        print(f"✅ Transcrições juntadas ({len(full_transcript)} caracteres)")
        return full_transcript
        
    except Exception as e:
        print(f"❌ Erro ao processar vídeo: {e}")
        raise

def extract_client_name_from_transcript(transcript: str) -> str:
    """Tenta extrair o nome do cliente da transcrição usando IA."""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em extrair informações de textos."},
                {"role": "user", "content": f"Esta é a transcrição de uma reunião de vendas. Por favor, identifique e extraia APENAS o nome do cliente/prospect mencionado na conversa. Se houver múltiplos nomes, extraia o nome que parece ser do cliente principal. Se não conseguir identificar com certeza, retorne 'Cliente Não Identificado'.\n\n{transcript[:2000]}"}
            ],
            max_tokens=50
        )
        client_name = response.choices[0].message.content.strip()
        
        # Limpar o resultado para ter apenas o nome
        client_name = re.sub(r'^[^a-zA-ZÀ-ÿ]*|[^a-zA-ZÀ-ÿ ]*$', '', client_name)
        client_name = re.sub(r'O nome do cliente é |Cliente: |Nome: ', '', client_name)
        
        # Se o resultado for muito longo ou vazio, usar valor padrão
        if len(client_name) > 50 or len(client_name) < 2:
            return "Cliente Não Identificado"
            
        return client_name
    except Exception as e:
        print(f"Erro ao extrair nome do cliente: {e}")
        return "Cliente Não Identificado"

def extract_client_name_from_filename(filename: str) -> str:
    """Extrai o nome do cliente do nome do arquivo no formato 'ClaxClub - Nome do cliente'."""
    try:
        # Verificar se o nome do arquivo segue o padrão ClaxClub - Nome do cliente
        if " - " not in filename:
            return "Cliente Não Identificado"
        
        # Extrair parte após "ClaxClub - "
        if filename.lower().startswith("claxclub"):
            parts = filename.split(" - ", 1)
            if len(parts) >= 2:
                # Extrai a parte após "ClaxClub - "
                client_part = parts[1]
                
                # Se houver mais hífens, pode ser data ou outras informações, pegar só a primeira parte
                if " - " in client_part:
                    client_part = client_part.split(" - ")[0]
                
                # Se houver parênteses, extrair o que está antes
                if "(" in client_part:
                    client_part = client_part.split("(")[0].strip()
                
                return client_part
        
        return "Cliente Não Identificado"
    except Exception as e:
        print(f"Erro ao extrair nome do cliente do nome do arquivo: {e}")
        return "Cliente Não Identificado"

def generate_sales_analysis(transcript: str) -> str:
    """Gera uma análise detalhada da reunião de vendas."""
    max_retries = 3
    retry_delay = 5  # segundos
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Gerando análise da reunião... (Tentativa {attempt + 1}/{max_retries})")
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em criar resumos de reuniões de forma clara, bem estruturada e profissional. Sua análise deve considerar o contexto da conversa, sem julgamentos inadequados. Respeite estratégias de vendas consultivas e firmes, como as ensinadas por nomes como Flávio Augusto, Joel Jota e Caio Carneiro."},
                    {"role": "user", "content": f"""Sua tarefa é gerar um resumo organizado no seguinte formato:

🧩 Segmento:
[Descreva o segmento de atuação do cliente de forma resumida]

🧠 Modelo de negócios:
[Explique brevemente como o cliente atua, detalhando produtos/serviços principais e o público-alvo]

💰 Faturamento anual:
[Informe o faturamento anual ou a projeção, se mencionado]

📊 Margens:
[Descreva a margem média de lucro, se disponível]

🎯 Ticket médio:
[Informe o ticket médio por tipo de produto ou serviço]

🔁 Processo comercial:
[Resuma como é o processo de vendas/comercial hoje — equipe, canais, desafios]

👥 Número de colaboradores:
[Quantidade de colaboradores internos e externos]

🩹 O que ele precisa? Qual sua dor?
[Liste de forma objetiva os principais desafios e necessidades que o cliente relatou]

🤝 Relação prévia com a empresa ou ecossistema:
[Explique se ele já conhece, já é cliente, participa de outros programas ou tem admiração pelo ecossistema]

📈 Probabilidade de fechamento:
[Avalie se a chance de fechamento é alta, média ou baixa, com justificativa baseada no comportamento do cliente na reunião]

❌ O que poderia ser feito de forma diferente na abordagem comercial:
[Em vez de criticar o vendedor, identifique oportunidades de melhoria com foco em adaptação ao perfil do cliente e condução mais estratégica — principalmente quando o lead já expressou seus objetivos. Considere também o tempo limitado da reunião.]

⏱️ Sentimento temporal:
[Analise o sentimento do cliente ao longo da reunião, destacando momentos em que demonstrou maior animação ou resistência. Considere a limitação de tempo e possíveis interrupções.]

👤 Nome do cliente:
[Identifique o nome completo do cliente/prospect mencionado na conversa. Se não for possível identificar com certeza, escreva "Cliente Não Identificado"]

⚠️ Observações técnicas:
• Sempre separe claramente as falas entre vendedor e cliente, mesmo quando a transcrição estiver com formatação ambígua.
• Caso os dados de ROI estejam incompletos ou imprecisos, evite gerar análises conclusivas e sinalize essa limitação no relatório.

Após isso, gere também um resumo ainda mais enxuto, com apenas frases curtas e objetivas, focando nos seguintes pontos:
• Segmento
• Conhecimento prévio sobre a empresa
• Faturamento e margens
• Número de colaboradores
• Público-alvo
• Principais dores e necessidades
• Impressão sobre o interesse e próxima ação

Importante:
• Seja objetivo, profissional e direto.
• Use emojis para organizar visualmente (como no exemplo acima).
• Não invente dados que não foram citados.
• Considere o tempo real da reunião: priorize os pontos discutidos com mais profundidade.
• Seja coerente com o estilo de abordagem consultiva e firme.
• Separe claramente o que foi falado pelo vendedor e pelo lead.

Transcrição:
{transcript}"""}
                ],
                max_tokens=2000,
                timeout=30
            )
            
            # Adicionar uso de tokens ao contador
            token_counter.add_usage(response)
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"❌ Erro ao gerar análise (Tentativa {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Aguardando {retry_delay} segundos antes de tentar novamente...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Todas as tentativas falharam")
                return f"""# Análise da Reunião (GERAÇÃO FALHOU)

Não foi possível gerar a análise completa após {max_retries} tentativas.
Último erro: {str(e)}

## Resumo Básico
A transcrição contém aproximadamente {len(transcript)} caracteres.

## Recomendação
Por favor, verifique sua conexão com a internet e tente novamente mais tarde.
Se o problema persistir, entre em contato com o suporte técnico.
"""

def extract_client_name_from_analysis(analysis: str) -> str:
    """Extrai o nome do cliente da análise de vendas."""
    try:
        # Procurar pelo padrão "👤 Nome do cliente:" ou similar na análise
        name_pattern = re.search(r'(?:👤|Nome do cliente:)[^\n]*?([A-Za-zÀ-ÿ\s]{2,50})', analysis, re.IGNORECASE)
        if name_pattern:
            client_name = name_pattern.group(1).strip()
            # Limpar o resultado
            client_name = re.sub(r'^\s*[:-]\s*', '', client_name)
            client_name = re.sub(r'Cliente Não Identificado|Não foi possível identificar|Nome não identificado', 'Cliente Não Identificado', client_name, re.IGNORECASE)
            
            # Se for muito curto ou longo, usar valor padrão
            if len(client_name) < 2 or len(client_name) > 50 or client_name.lower() == "não identificado":
                return "Cliente Não Identificado"
                
            return client_name
            
        # Se não encontrar o padrão específico, verificar menções gerais ao cliente
        possible_names = re.findall(r'cliente[:\s]+([A-Za-zÀ-ÿ\s]{2,50})', analysis, re.IGNORECASE)
        if possible_names:
            for name in possible_names:
                clean_name = name.strip()
                if len(clean_name) >= 2 and len(clean_name) <= 50:
                    return clean_name
        
        return "Cliente Não Identificado"
        
    except Exception as e:
        print(f"Erro ao extrair nome do cliente da análise: {e}")
        return "Cliente Não Identificado"

def generate_roi_report(transcript: str, analysis: str) -> str:
    """Gera um relatório de ROI baseado na transcrição e análise da reunião."""
    max_retries = 3
    retry_delay = 5  # segundos
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Gerando relatório de ROI... (Tentativa {attempt + 1}/{max_retries})")
            
            # Obter a data atual para o relatório
            current_date = time.strftime("%d/%m/%Y")
            
            # Extrair informações importantes da análise
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Primeiro, extrair dados essenciais da análise para o relatório de ROI
            print("🔄 Extraindo dados para relatório de ROI...")
            extract_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em extrair informações de negócios de textos."},
                    {"role": "user", "content": f"""Extraia da análise a seguir as informações relevantes para criar um relatório de ROI para o ClaxClub. 
                    Forneça os dados no formato JSON com os seguintes campos:
                    
                    {{
                        "nome_empresario": "Nome completo do cliente/empresário",
                        "faturamento": "Faturamento anual ou mensal com valor numérico",
                        "ticket_medio": "Ticket médio de venda com valor numérico",
                        "margem": "Margem atual em porcentagem",
                        "produto_servico": "Produto ou serviço principal",
                        "regiao_atuacao": "Região onde atua (estado, país, cidade)",
                        "dores": ["Lista das principais dores mencionadas"],
                        "potencial_crescimento": "Alto, médio ou baixo, com justificativa",
                        "sonho_visao": "Sonho ou visão de futuro mencionada",
                        "interesses_mentoria": ["Lista de interesses na mentoria"],
                        "nivel_autoridade": "Alto, médio ou baixo no mercado",
                        "abertura_mudanca": "Alta, média ou baixa"
                    }}
                    
                    Se alguma informação não estiver disponível, use "não informado". 
                    
                    Análise:
                    {analysis}
                    
                    Transcrição (use apenas se necessário para complementar):
                    {transcript[:3000]}"""
                    }
                ],
                max_tokens=1000,
                timeout=30
            )
            
            # Adicionar uso de tokens ao contador
            token_counter.add_usage(extract_response)
            
            # Extrair o JSON da resposta
            try:
                extracted_data = json.loads(extract_response.choices[0].message.content)
                print("✅ Dados para relatório de ROI extraídos com sucesso")
            except json.JSONDecodeError:
                print("⚠️ Erro ao decodificar JSON, tentando extrair manualmente...")
                # Tentar extrair manualmente se o formato não for perfeitamente JSON
                content = extract_response.choices[0].message.content
                # Encontrar o início e fim do JSON na resposta
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    try:
                        extracted_data = json.loads(content[start:end])
                        print("✅ Dados extraídos manualmente com sucesso")
                    except:
                        print("❌ Falha na extração manual")
                        extracted_data = {
                            "nome_empresario": "Não identificado",
                            "faturamento": "não informado",
                            "ticket_medio": "não informado",
                            "margem": "não informado",
                            "produto_servico": "não informado",
                            "regiao_atuacao": "não informado",
                            "dores": ["não informado"],
                            "potencial_crescimento": "não informado",
                            "sonho_visao": "não informado",
                            "interesses_mentoria": ["não informado"],
                            "nivel_autoridade": "não informado",
                            "abertura_mudanca": "não informado"
                        }
                else:
                    extracted_data = {
                        "nome_empresario": "Não identificado",
                        "faturamento": "não informado",
                        "ticket_medio": "não informado",
                        "margem": "não informado",
                        "produto_servico": "não informado",
                        "regiao_atuacao": "não informado",
                        "dores": ["não informado"],
                        "potencial_crescimento": "não informado",
                        "sonho_visao": "não informado",
                        "interesses_mentoria": ["não informado"],
                        "nivel_autoridade": "não informado",
                        "abertura_mudanca": "não informado"
                    }
            
            # Preparar os dados para o prompt
            nome = extracted_data.get("nome_empresario", "Não identificado")
            faturamento = extracted_data.get("faturamento", "não informado")
            ticket_medio = extracted_data.get("ticket_medio", "não informado")
            margem = extracted_data.get("margem", "não informado")
            produto_servico = extracted_data.get("produto_servico", "não informado")
            regiao_atuacao = extracted_data.get("regiao_atuacao", "não informado")
            dores = ", ".join(extracted_data.get("dores", ["não informado"]))
            potencial_crescimento = extracted_data.get("potencial_crescimento", "não informado")
            sonho_visao = extracted_data.get("sonho_visao", "não informado")
            interesses_mentoria = ", ".join(extracted_data.get("interesses_mentoria", ["não informado"]))
            nivel_autoridade = extracted_data.get("nivel_autoridade", "não informado")
            abertura_mudanca = extracted_data.get("abertura_mudanca", "não informado")
            
            # Gerar o relatório de ROI com base nos dados extraídos
            print("🔄 Gerando documento de ROI com os dados extraídos...")
            roi_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": """✅ Prompt para criação de Agente IA - Documento Pós-Reunião Comercial (ClaxClub)
Você é um agente especialista em análise de negócios e vendas, com foco em gerar um relatório personalizado para empresários logo após uma reunião comercial. Seu objetivo é mostrar, com dados e projeções claras, como o ingresso no ClaxClub (clube de mentoria com Flávio Augusto, Joel Jota e Caio Carneiro) pode gerar retorno financeiro (ROI tangível) e autoridade de mercado (ROI intangível)."""},
                    {"role": "user", "content": f"""Crie o documento de ROI para o cliente {nome}. 
Faturamento atual: {faturamento}
Ticket médio atual de venda: {ticket_medio}
Margem atual: {margem}
Produto/serviço principal: {produto_servico}
Região de atuação: {regiao_atuacao}
Maiores dores reveladas na reunião: {dores}
Potencial de crescimento: {potencial_crescimento}
Sonho ou visão de futuro mencionada: {sonho_visao}
Interesses na mentoria: {interesses_mentoria}
Nível de autoridade atual no mercado: {nivel_autoridade}
Grau de abertura à mudança: {abertura_mudanca}

✍️ Instruções de saída (output)
Gere um documento com os seguintes blocos, sempre usando linguagem direta, personalizada e profissional:

1.⁠ ⁠Introdução Personalizada
Apresente uma visão sobre o momento atual do cliente e o que foi percebido na reunião. Reconheça conquistas e valide dores.

2.⁠ ⁠Cenário Atual vs. Possível com o Clax
Simule crescimento de faturamento, ticket médio e margem, mostrando o impacto direto mesmo sem depender de novos clientes. Inclua também um cenário com aumento de demanda. Use tabelas se possível.

3.⁠ ⁠ROI Tangível
Simule projeções com números claros (lucro, retorno do investimento)

Multiplicador do investimento no ClaxClub - caso não tenha na conversa o valor do club, considere R$ 400.000,00.

4.⁠ ⁠ROI Intangível
Liste os ganhos não financeiros, como:

Autoridade por associação aos mentores

Networking com empresários relevantes

Estruturação do negócio e do time

Acesso a treinamentos e capacitação

Possibilidade de virar mentor e ganhar equity

5.⁠ ⁠Forma de pagamento
Apresente a proposta como um fluxo inteligente de investimento (parcelas escalonadas, etc.)

6.⁠ ⁠Conclusão personalizada
Reforce que o empresário está exatamente na fase ideal para o ClaxClub, e que essa é uma decisão estratégica, não apenas emocional. Termine com uma frase forte, como:

"Você tem muito a ganhar. E mais ainda, muito a contribuir."

O documento deve seguir exatamente essa estrutura de tópicos e ter cabeçalho com data atual ({current_date})."""
                    }
                ],
                max_tokens=3000,
                timeout=30
            )
            
            # Adicionar uso de tokens ao contador
            token_counter.add_usage(roi_response)
            
            # Formatar o relatório final
            roi_content = roi_response.choices[0].message.content
            
            # Verificar se há cabeçalho, caso contrário, adicionar
            if "Relatório de ROI" not in roi_content and "ROI - ClaxClub" not in roi_content:
                cabeçalho = f"""**Relatório de ROI - Proposta ClaxClub**
Cliente: {nome}
Produto/Serviço: {produto_servico}
Data: {current_date}

---

"""
                roi_content = cabeçalho + roi_content
            
            print("✅ Relatório de ROI gerado com sucesso")
            return roi_content
                
        except Exception as e:
            print(f"❌ Erro ao gerar relatório de ROI (Tentativa {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Aguardando {retry_delay} segundos antes de tentar novamente...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Todas as tentativas falharam")
                current_date = time.strftime("%d/%m/%Y")
                return f"""**Relatório de ROI - Proposta ClaxClub**
Cliente: [NOME EMPRESÁRIO/A NÃO INFORMADO]
Produto/Serviço: [PRODUTO/SERVIÇO NÃO INFORMADO]
Data: {current_date}

---

# ❌ Relatório de ROI (GERAÇÃO FALHOU)

Não foi possível gerar o relatório de ROI após {max_retries} tentativas.
Último erro: {str(e)}

## 🔧 Recomendação
Por favor, verifique sua conexão com a internet e tente novamente mais tarde.
Se o problema persistir, entre em contato com o suporte técnico.
"""

def identify_speakers(transcript: str, meeting_info: Dict) -> str:
    """
    Identifica os diferentes falantes na transcrição e formata o texto com nomes.
    
    Args:
        transcript: A transcrição original
        meeting_info: Informações sobre a reunião, incluindo vendedor e cliente
    
    Returns:
        Transcrição formatada com identificação dos falantes
    """
    max_retries = 3
    retry_delay = 5  # segundos
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Identificando falantes na transcrição... (Tentativa {attempt + 1}/{max_retries})")
            
            # Sempre usar "Vendedor" para o vendedor 
            seller_name = "Vendedor"
            
            # Para o cliente, usar o nome real ou "Cliente" se não identificado
            client_name = meeting_info.get('client_name', 'Cliente')
            if client_name == "Cliente Não Identificado":
                client_name = "Cliente"
            
            print(f"🔍 Identificando falas entre: Vendedor e {client_name}")
            print(f"📊 Tamanho da transcrição: {len(transcript)} caracteres")
            
            # Processar a transcrição em uma única chamada, sem dividir
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em análise de conversas e identificação de falantes em transcrições."},
                    {"role": "user", "content": f"""Identifique os diferentes falantes nesta transcrição e reformate-a para que cada fala seja precedida pelo nome do falante seguido de dois pontos.

Informações importantes:
- O VENDEDOR deve ser SEMPRE identificado apenas como "Vendedor" (nunca use o nome real do vendedor)
- O CLIENTE deve ser identificado como "{client_name}"
- Podem existir outros participantes na conversa - identifique-os como "Participante 1", "Participante 2", etc.
- Mantenha a sequência exata das falas e todo o conteúdo original
- NÃO resuma, divida ou altere o conteúdo das falas
- NÃO adicione introduções ou explicações
- MANTENHA todas as falas na íntegra, sem quebrar em partes menores
- IMPORTANTE: Certifique-se de sempre alternar corretamente entre Vendedor e {client_name}, analisando o contexto da conversa para identificar quem está falando

Exemplo do formato desejado:
Vendedor: Olá, tudo bem com você? Obrigado por reservar um tempo para nossa conversa hoje.
{client_name}: Tudo bem sim, obrigado por me atender.
Vendedor: Então, como eu estava explicando por e-mail, a nossa solução...

Transcrição:
{transcript}"""}
                ],
                max_tokens=4000,
                timeout=180  # Aumentar timeout para transcrições longas
            )
            
            # Adicionar uso de tokens ao contador
            token_counter.add_usage(response)
            
            formatted_transcript = response.choices[0].message.content.strip()
            
            # Garantir que a formatação está correta
            if not re.search(r'^[A-Za-zÀ-ÿ\s]+:', formatted_transcript, re.MULTILINE):
                # Se não tiver o formato esperado (nome: fala), tentar formatar manualmente
                print("🔄 Ajustando formato da transcrição...")
                segments = re.split(r'\n\n+|\n(?=[A-Z])|(?<=[.!?])\s+(?=[A-Z])', transcript)
                formatted_lines = []
                for i, segment in enumerate(segments):
                    if segment.strip():
                        speaker = "Vendedor" if i % 2 == 0 else client_name
                        formatted_lines.append(f"{speaker}: {segment.strip()}")
                
                formatted_transcript = "\n".join(formatted_lines)
            
            # Verificar se não há confusão entre Vendedor e Cliente
            real_seller_name = meeting_info.get('seller_name', '')
            if real_seller_name and real_seller_name in formatted_transcript:
                # Substituir o nome real do vendedor por "Vendedor"
                formatted_transcript = re.sub(
                    rf"{re.escape(real_seller_name)}:", 
                    "Vendedor:", 
                    formatted_transcript
                )
            
            # Remover espaços duplos entre as falas para economizar tokens
            formatted_transcript = re.sub(r'\n\n+', '\n', formatted_transcript)
            
            print(f"✅ Falantes identificados com sucesso!")
            return formatted_transcript
                
        except openai.RateLimitError as e:
            print(f"⚠️ Erro de limite de tokens (Tentativa {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"⏳ Aguardando {retry_delay} segundos antes de tentar novamente...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Para erros de limite de tokens, usar abordagem simplificada
                print("⚠️ Usando abordagem simplificada devido a limitações de tokens")
                return simple_speaker_identification(transcript, meeting_info)
        except Exception as e:
            print(f"❌ Erro ao identificar falantes (Tentativa {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Aguardando {retry_delay} segundos antes de tentar novamente...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Todas as tentativas falharam")
                return simple_speaker_identification(transcript, meeting_info)

def simple_speaker_identification(transcript: str, meeting_info: Dict) -> str:
    """
    Implementação simplificada para identificar falantes quando o método principal falha.
    
    Este método é mais eficiente para transcrições grandes, pois não depende de modelos grandes
    de linguagem, mas é menos preciso na identificação dos falantes.
    """
    print("🔄 Usando método alternativo para identificação de falantes...")
    
    # Para o cliente, usar o nome real ou "Cliente" se não identificado
    client_name = meeting_info.get('client_name', 'Cliente') 
    if client_name == "Cliente Não Identificado":
        client_name = "Cliente"
    
    # Dividir por parágrafos ou frases completas
    segments = re.split(r'\n\n+|\n(?=[A-Z])|(?<=[.!?])\s+(?=[A-Z])', transcript)
    formatted_lines = []
    
    # Padrões para detectar perguntas típicas de vendedor
    vendedor_patterns = [
        r'(?i)como posso (ajudar|auxiliar)',
        r'(?i)qual (é|seria) (o seu|seu)',
        r'(?i)gostaria de (saber|entender)',
        r'(?i)você (já|tem|possui)',
        r'(?i)poderia me (contar|falar)',
        r'(?i)(vamos|podemos) (conversar|falar)',
        r'(?i)(obrigado|agradeço)',
        r'(?i)bem(-|\s)vindo',
        r'(?i)prazer em (conhecer|falar)',
        r'(?i)então.{1,30}(como|qual)'
    ]
    
    # Tente identificar quem começa a conversa - normalmente é o vendedor
    current_speaker = "Vendedor"
    
    for i, segment in enumerate(segments):
        if not segment.strip():
            continue
            
        # Detectar padrões de fala do vendedor
        is_vendedor = False
        if i == 0:  # Primeira fala geralmente é do vendedor
            is_vendedor = True
        else:
            # Verificar padrões de fala típicos de vendedor
            for pattern in vendedor_patterns:
                if re.search(pattern, segment):
                    is_vendedor = True
                    break
        
        # Alternar falantes se não for o vendedor
        if not is_vendedor:
            current_speaker = client_name if current_speaker == "Vendedor" else "Vendedor"
        
        # Adicionar fala formatada (sem negrito)
        formatted_lines.append(f"{current_speaker}: {segment.strip()}")
        
        # Alternar para próxima fala
        current_speaker = client_name if current_speaker == "Vendedor" else "Vendedor"
    
    # Juntar com quebras de linha simples, sem espaços extras
    return "\n".join(formatted_lines)

def format_transcript_for_output(transcript: str, meeting_info: Dict) -> str:
    """Formata a transcrição para um formato mais estruturado para salvar."""
    timestamp = time.strftime("%d/%m/%Y %H:%M")
    
    # Verificar o tamanho da transcrição
    transcript_length = len(transcript)
    print(f"📊 Tamanho da transcrição original: {transcript_length} caracteres")
    
    # Usar a transcrição original sem identificação de falantes
    print("ℹ️ Mantendo transcrição original sem identificação de falantes...")
    formatted_transcript = transcript
    
    formatted_output = f"""# TRANSCRIÇÃO COMPLETA DA REUNIÃO

Data de processamento: {timestamp}

---

{formatted_transcript}

---
"""
    return formatted_output

def save_results(service: build, transcript: str, analysis: str, meeting_info: Dict, original_folder_id: str):
    """
    Salva a transcrição e a análise em uma pasta específica criada dentro da pasta original do vídeo.
    Para cada vídeo, cria uma pasta dedicada para seus arquivos de transcrição e análise.
    """
    try:
        # Extrair informações
        original_filename = meeting_info.get('original_filename', '')
        client_name = meeting_info.get('client_name', 'Cliente Não Identificado')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        date_formatted = time.strftime("%d-%m-%Y")
        
        print(f"💾 Iniciando salvamento dos resultados...")
        print(f"💾 Pasta original ID: {original_folder_id}")
        
        # Verificar a pasta original
        try:
            folder_info = service.files().get(fileId=original_folder_id, fields="name").execute()
            original_folder_name = folder_info.get("name", "Pasta desconhecida")
            print(f"💾 Pasta original: {original_folder_name} (ID: {original_folder_id})")
        except Exception as e:
            print(f"⚠️ Erro ao verificar pasta original: {str(e)}")
            original_folder_name = "Pasta desconhecida"
        
        # Criar nome para a pasta específica com o nome do arquivo
        video_folder_name = f"Transcrição - {original_filename}"
        print(f"💾 Criando pasta para o arquivo: {video_folder_name}")
        
        # Criar pasta específica para o vídeo dentro da pasta original
        folder_metadata = {
            'name': video_folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [original_folder_id]
        }
        
        try:
            video_folder = service.files().create(body=folder_metadata, fields='id,name').execute()
            video_folder_id = video_folder.get('id')
            print(f"✅ Pasta criada: {video_folder_name} (ID: {video_folder_id})")
        except Exception as e:
            print(f"❌ Erro ao criar pasta para o vídeo: {str(e)}")
            raise
        
        print(f"💾 Tamanho da transcrição: {len(transcript)} caracteres")
        print(f"💾 Tamanho da análise: {len(analysis)} caracteres")
        
        # Formatar transcrição e análise
        formatted_transcript = format_transcript_for_output(transcript, meeting_info)
        
        # Gerar relatório de ROI
        print(f"🔄 Gerando relatório de ROI...")
        roi_report = generate_roi_report(transcript, analysis)
        print(f"✅ Relatório de ROI gerado: {len(roi_report)} caracteres")
        
        # Definir nomes de arquivos mais estruturados
        transcript_filename = f"Transcrição - {date_formatted}.md"
        analysis_filename = f"Análise - {date_formatted}.md"
        roi_filename = f"Relatório ROI - {date_formatted}.md"
        
        # Salvar primeiro localmente para depois fazer upload
        safe_filename = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in Path(original_filename).stem)
        local_transcript_path = OUTPUT_DIR / f"{safe_filename}_transcript_{timestamp}.md"
        local_analysis_path = OUTPUT_DIR / f"{safe_filename}_analysis_{timestamp}.md"
        local_roi_path = OUTPUT_DIR / f"{safe_filename}_roi_{timestamp}.md"
        
        print(f"💾 Salvando arquivos localmente:")
        print(f"💾 - Transcrição: {local_transcript_path}")
        print(f"💾 - Análise: {local_analysis_path}")
        print(f"💾 - Relatório ROI: {local_roi_path}")
        
        # Salvar localmente
        with open(local_transcript_path, 'w', encoding='utf-8') as f:
            f.write(formatted_transcript)
        
        with open(local_analysis_path, 'w', encoding='utf-8') as f:
            f.write(analysis)
            
        with open(local_roi_path, 'w', encoding='utf-8') as f:
            f.write(roi_report)
        
        # Verificar se os arquivos foram salvos corretamente
        if os.path.exists(local_transcript_path) and os.path.getsize(local_transcript_path) > 0:
            print(f"✅ Arquivo de transcrição salvo localmente: {os.path.getsize(local_transcript_path) / 1024:.2f} KB")
        else:
            print(f"❌ Erro ao salvar arquivo de transcrição local")
            
        if os.path.exists(local_analysis_path) and os.path.getsize(local_analysis_path) > 0:
            print(f"✅ Arquivo de análise salvo localmente: {os.path.getsize(local_analysis_path) / 1024:.2f} KB")
        else:
            print(f"❌ Erro ao salvar arquivo de análise local")
            
        if os.path.exists(local_roi_path) and os.path.getsize(local_roi_path) > 0:
            print(f"✅ Arquivo de relatório ROI salvo localmente: {os.path.getsize(local_roi_path) / 1024:.2f} KB")
        else:
            print(f"❌ Erro ao salvar arquivo de relatório ROI local")
        
        # Upload para a pasta específica do vídeo
        print(f"💾 Fazendo upload para a pasta: {video_folder_name}")
        
        try:
            transcript_id = upload_file_to_drive(
                service, 
                str(local_transcript_path), 
                transcript_filename, 
                video_folder_id  # Usar a pasta específica do vídeo
            )
            
            print(f"✅ Transcrição enviada para o Drive (ID: {transcript_id})")
            
            analysis_id = upload_file_to_drive(
                service, 
                str(local_analysis_path), 
                analysis_filename, 
                video_folder_id  # Usar a pasta específica do vídeo
            )
            
            print(f"✅ Análise enviada para o Drive (ID: {analysis_id})")
            
            roi_id = upload_file_to_drive(
                service, 
                str(local_roi_path), 
                roi_filename, 
                video_folder_id  # Usar a pasta específica do vídeo
            )
            
            print(f"✅ Relatório ROI enviado para o Drive (ID: {roi_id})")
            
            print(f"✓ Documentos enviados para a pasta: {video_folder_name}")
            print(f"  - Transcrição: {transcript_filename} (ID: {transcript_id})")
            print(f"  - Análise: {analysis_filename} (ID: {analysis_id})")
            print(f"  - Relatório ROI: {roi_filename} (ID: {roi_id})")
            
            # Verificar os arquivos após upload
            print("🔍 Verificando arquivos criados...")
            for file_id in [transcript_id, analysis_id, roi_id]:
                try:
                    file_info = service.files().get(fileId=file_id, fields="name,parents").execute()
                    parents = file_info.get("parents", [])
                    if video_folder_id in parents:
                        print(f"✅ Arquivo {file_info.get('name')} está na pasta correta: {video_folder_name}")
                    else:
                        print(f"⚠️ Arquivo {file_info.get('name')} pode não estar na pasta correta!")
                        print(f"⚠️ Pasta esperada: {video_folder_id}")
                        print(f"⚠️ Pastas do arquivo: {parents}")
                except Exception as e:
                    print(f"⚠️ Erro ao verificar arquivo {file_id}: {str(e)}")
            
            return transcript_id, analysis_id, roi_id
        except Exception as e:
            print(f"❌ Erro ao fazer upload para o Google Drive: {str(e)}")
            print(f"💡 Os arquivos ainda estão disponíveis localmente em {OUTPUT_DIR}")
            raise
            
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {str(e)}")
        # Tentar salvar localmente como fallback
        try:
            fallback_transcript = OUTPUT_DIR / f"fallback_transcript_{timestamp}.md"
            fallback_analysis = OUTPUT_DIR / f"fallback_analysis_{timestamp}.md"
            fallback_roi = OUTPUT_DIR / f"fallback_roi_{timestamp}.md"
            
            with open(fallback_transcript, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            with open(fallback_analysis, 'w', encoding='utf-8') as f:
                f.write(analysis)
                
            with open(fallback_roi, 'w', encoding='utf-8') as f:
                f.write(roi_report if 'roi_report' in locals() else "Erro ao gerar relatório ROI")
                
            print(f"⚠️ Fallback: Arquivos salvos como {fallback_transcript}, {fallback_analysis} e {fallback_roi}")
        except Exception as e2:
            print(f"❌ Erro ao salvar fallback: {str(e2)}")
        
        raise

def process_video(service: build, video_path: str, seller_name: str, original_filename: str, original_folder_id: str):
    """Processa um vídeo, transcreve e gera análise de vendas."""
    print(f"\n🔄 Iniciando processamento do vídeo: {original_filename}")
    
    # Reiniciar contadores para este vídeo
    token_counter.reset()
    transcription_counter.reset()
    
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"O sistema não pode encontrar o arquivo especificado: {video_path}")
        
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(video_path)
        print(f"📊 Tamanho do arquivo: {file_size / (1024*1024):.2f}MB")
        
        # Tentar extrair nome do cliente do título do arquivo primeiro
        client_name = extract_client_name_from_filename(original_filename)
        print(f"🔍 Nome do cliente extraído do título: {client_name}")
        
        # Lista para armazenar todas as transcrições
        all_transcripts = []
    
        if file_size <= MAX_FILE_SIZE:
            # Se o arquivo for menor que 25MB, transcreve diretamente
            print("📝 Arquivo menor que 25MB, transcrevendo diretamente...")
            transcript = transcribe_audio(video_path)
            all_transcripts.append(transcript)
        else:
            # Se o arquivo for maior que 25MB, divide em partes
            print(f"📝 Arquivo maior que 25MB, dividindo em partes...")
            parts = split_video_by_size(video_path)
            
            if not parts:
                raise ValueError("Falha ao dividir o vídeo em partes menores")
            
            print(f"✅ Vídeo dividido em {len(parts)} partes")
            
            # Transcrever cada parte
            for i, part in enumerate(parts, 1):
                print(f"\n🔄 Transcrevendo parte {i} de {len(parts)}...")
                
                if not os.path.exists(part):
                    print(f"⚠️ Parte {i} não encontrada: {part}")
                    continue
                
                part_size = os.path.getsize(part)
                print(f"📊 Tamanho da parte {i}: {part_size / (1024*1024):.2f}MB")
            
                try:
                    transcript = transcribe_audio(part)
                    all_transcripts.append(transcript)
                    print(f"✅ Parte {i} transcrita com sucesso!")
                except Exception as e:
                    print(f"❌ Erro ao transcrever parte {i}: {str(e)}")
                    # Usar marcador de erro menos intrusivo
                    all_transcripts.append(f" [Falha na transcrição. Continua na próxima parte.] ")
                
                # Remover parte temporária
                if part != video_path and os.path.exists(part):
                    os.remove(part)
                    print(f"🧹 Parte {i} removida")
            
            # Verificar se pelo menos uma parte foi transcrita
            if not any(not t.startswith(" [Falha") for t in all_transcripts):
                raise ValueError("Todas as partes da transcrição falharam. Não foi possível transcrever o vídeo.")
        
        # Juntar todas as transcrições
        print("\n🔄 Juntando todas as transcrições...")
        full_transcript = " ".join(all_transcripts)
        
        # Verificar se o resultado final é razoável
        if len(full_transcript) < 100:
            print("⚠️ A transcrição resultante é muito curta, pode ter ocorrido um erro.")
        
        print(f"✅ Transcrições juntadas ({len(full_transcript)} caracteres)")
    
        # Gerar análise de vendas
        print("\n🔄 Gerando análise da reunião...")
        analysis = generate_sales_analysis(full_transcript)
        print("✅ Análise gerada com sucesso!")
        
        # Se não conseguimos extrair o nome do cliente do título, tentar extrair da análise
        if client_name == "Cliente Não Identificado":
            print("\n🔄 Tentando extrair nome do cliente da análise...")
            client_name = extract_client_name_from_analysis(analysis)
            print(f"✅ Nome do cliente extraído da análise: {client_name}")
    
        # Informações da reunião
        meeting_info = {
            'seller_name': seller_name,
            'client_name': client_name,
            'original_filename': original_filename
        }
    
        # Salvar resultados na pasta original do vídeo
        print("\n🔄 Salvando resultados na pasta original do vídeo...")
        save_results(service, full_transcript, analysis, meeting_info, original_folder_id)
        print("✅ Resultados salvos com sucesso!")
        
        # No final do processamento, exibir o resumo de tokens e transcrições
        token_summary = token_counter.get_summary()
        transcription_summary = transcription_counter.get_summary()
        
        print("\n📊 Resumo de Uso da API OpenAI:")
        print(f"Total de Tokens GPT-4.1: {token_summary['total_tokens']:,}")
        print(f"Tokens de Prompt: {token_summary['prompt_tokens']:,}")
        print(f"Tokens de Completão: {token_summary['completion_tokens']:,}")
        print(f"Custo Estimado GPT-4.1: ${token_summary['estimated_cost']:.4f}")
        
        print("\n📊 Resumo de Transcrições:")
        print(f"Total de Minutos: {transcription_summary['total_minutes']:.2f}")
        print(f"Total de Arquivos: {transcription_summary['total_files']}")
        print(f"Custo Estimado Transcrições: ${transcription_summary['estimated_cost']:.4f}")
        
        custo_atual = token_summary['estimated_cost'] + transcription_summary['estimated_cost']
        print(f"\n💰 Custo Total Estimado: ${custo_atual:.4f}")
        
        # Adicionar ao custo total acumulado
        total_cost_tracker.add_cost(custo_atual)
        total_summary = total_cost_tracker.get_summary()
        print(f"\n💸 Custo Total Acumulado: ${total_summary['total_cost']:.4f}")
        print(f"📅 Última atualização: {total_summary['last_updated']}")
        
    except Exception as e:
        print(f"\n❌ Erro ao processar vídeo: {str(e)}")
        raise
        
    finally:
        # Limpar arquivo temporário
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"🧹 Arquivo temporário removido: {video_path}")
            except Exception as e:
                print(f"⚠️ Erro ao remover arquivo temporário: {str(e)}")
        
        print("\n✅ Processamento finalizado!")

def extract_owner_name(file_name: str) -> str:
    """Extrai o nome da pessoa do nome do arquivo."""
    # Tenta extrair no formato "Nome da Pessoa - Descrição.mp4"
    if " - " in file_name:
        return file_name.split(" - ")[0].strip()
    
    # Tenta extrair no formato "Nome_Sobrenome_Data.mp4" ou "Nome_Sobrenome.mp4"
    name_match = re.match(r'^([A-Za-zÀ-ÿ]+[_\s][A-Za-zÀ-ÿ]+)', file_name)
    if name_match:
        return name_match.group(1).replace('_', ' ')
    
    # Se não conseguir extrair, usa o nome do arquivo sem extensão
    return Path(file_name).stem

def main():
    # Criar diretórios necessários
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Verificar configurações
    if not FOLDERS_TO_MONITOR or FOLDERS_TO_MONITOR[0] == '':
        print("ERRO: Nenhuma pasta para monitorar configurada no arquivo .env")
        print("Configure a variável GOOGLE_DRIVE_FOLDERS_TO_MONITOR com IDs separados por vírgula")
        return
    
    # Carregar IDs de vídeos já processados
    processed_ids = load_processed_ids()
    
    # Autenticar com Google Drive usando Service Account
    service = authenticate_google_drive()
    
    # Verificar e mostrar informação sobre as pastas
    print(f"Pastas monitoradas: {len(FOLDERS_TO_MONITOR)} pastas")
    for i, folder_id in enumerate(FOLDERS_TO_MONITOR):
        print(f"  {i+1}. Pasta ID: {folder_id}")
    
    print(f"Monitorando pastas do Google Drive...")
    print(f"Esperando por novos vídeos... ({len(processed_ids)} vídeos já processados)")
    
    while True:
        try:
            all_new_videos = []
            
            # Verificar cada pasta monitorada
            for folder_id in FOLDERS_TO_MONITOR:
                # Obter lista de vídeos na pasta
                video_files = get_video_files(service, folder_id)
                
                # Filtrar apenas vídeos novos
                new_videos = [v for v in video_files if v['id'] not in processed_ids]
                
                # Adicionar informação da pasta fonte
                for v in new_videos:
                    v['source_folder_id'] = folder_id
                
                all_new_videos.extend(new_videos)
            
            if all_new_videos:
                print(f"\nEncontrados {len(all_new_videos)} novos vídeos!")
                
                for video in all_new_videos:
                    print(f"\nProcessando: {video['name']} (ID: {video['id']})")
                    
                    # Extrair nome do vendedor do nome do arquivo
                    seller_name = extract_owner_name(video['name'])
                    print(f"Nome do vendedor (do arquivo): {seller_name}")
                    
                    try:
                        # Baixar vídeo
                        video_path = download_file(service, video['id'], video['name'])
                        
                        # Verificar se o download foi bem sucedido
                        if not os.path.exists(video_path):
                            raise FileNotFoundError(f"Falha ao baixar o vídeo: {video_path}")
                        
                        # Processar vídeo
                        process_video(service, video_path, seller_name, video['name'], video['source_folder_id'])
                        
                        # Marcar como processado
                        processed_ids.add(video['id'])
                        save_processed_ids(processed_ids)
                        
                        print(f"✓ Vídeo {video['name']} processado com sucesso!")
                        
                        # Adicionar tempo de espera de 2 minutos entre os vídeos
                        print("\n⏳ Aguardando 2 minutos antes de processar o próximo vídeo...")
                        time.sleep(120)  # 120 segundos = 2 minutos
                        
                    except Exception as e:
                        print(f"✗ Erro ao processar vídeo {video['name']}: {str(e)}")
                    finally:
                        # Limpar arquivo temporário
                        if 'video_path' in locals() and os.path.exists(video_path):
                            try:
                                os.remove(video_path)
                                print(f"🧹 Arquivo temporário removido: {video_path}")
                            except Exception as e:
                                print(f"⚠️ Não foi possível remover arquivo temporário: {str(e)}")
            else:
                print(".", end="", flush=True)  # Indicador de que o script está rodando
                
            time.sleep(60)  # Verificar a cada minuto
            
        except Exception as e:
            print(f"\nErro: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()
    main()