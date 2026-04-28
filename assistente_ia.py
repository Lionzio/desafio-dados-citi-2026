import os
import pandas as pd
import logging
from dotenv import load_dotenv
from pathlib import Path
import warnings

# Ocultar avisos do Pandas
warnings.filterwarnings('ignore')

# Telemetria Avançada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Tentativas de Importação Seguras
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    pass

# ==========================================
# 1. ENGENHARIA DO CONTEXTO COM PANDAS
# ==========================================
def gerar_resumo_contexto(file_name: str) -> str:
    """Extrai os KPIs da base de dados."""
    base_dir = Path(__file__).parent.resolve()
    file_path = base_dir / file_name
    
    if not file_path.exists():
        raise FileNotFoundError("Base tratada não encontrada! Rode o script de tratamento primeiro.")

    df = pd.read_csv(file_path)
    df['Status_Transacao'] = df['Status_Transacao'].replace('Aprov.', 'Aprovada')

    total_transacoes = len(df)
    valor_total = df['Valor_Final'].sum()
    status_counts = df['Status_Transacao'].value_counts()
    status_pct = df['Status_Transacao'].value_counts(normalize=True) * 100
    status_str = ", ".join([f"{k} {status_counts[k]} ({status_pct[k]:.1f}%)" for k in status_counts.index])
    tipo_str = ", ".join([f"{k} {v}" for k, v in df['Tipo_Transacao'].value_counts().items()])
    
    # Recusas por Banco
    df_recusadas = df[df['Status_Transacao'] == 'Recusada']
    recusas_banco = df.groupby('Banco')['Status_Transacao'].apply(lambda x: (x == 'Recusada').sum())
    total_banco = df['Banco'].value_counts()
    taxa_recusa_banco = (recusas_banco / total_banco * 100).fillna(0).sort_values(ascending=False)
    bancos_str = ", ".join([f"{b} ({taxa_recusa_banco[b]:.1f}% recusa)" for b in taxa_recusa_banco.index[:5]])
    
    # Clientes com mais problemas
    clientes_recusa = df_recusadas['Nome_Cliente'].value_counts().head(5)
    clientes_str = ", ".join([f"{k} ({v})" for k, v in clientes_recusa.items()])
    
    # Ticket Médio
    ticket_medio = df.groupby('Tipo_Transacao')['Valor_Final'].mean().round(2)
    ticket_str = ", ".join([f"{k} R${v}" for k, v in ticket_medio.items()])
    
    return f"""
    Resumo Analítico da Base de Transações (Jun/2022 a Nov/2024):
    - Total de transações válidas: {total_transacoes}
    - Valor total movimentado: R$ {valor_total:,.2f}
    - Status: {status_str}
    - Tipos: {tipo_str}
    - Top 5 Bancos com mais recusas (%): {bancos_str}
    - Top 5 Clientes com transações RECUSADAS: {clientes_str}
    - Ticket médio: {ticket_str}
    """

# ==========================================
# 2. MOTOR DE IA RESILIENTE
# ==========================================
def iniciar_assistente():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not gemini_key and not groq_key:
        logger.critical("Sem chaves de API. O assistente não pode arrancar.")
        return

    contexto_dados = gerar_resumo_contexto('Base_Tratada_PTC_26.csv')
    
    prompt_sistema = f"""
    Você é um Consultor Financeiro e de Dados Sénior do CITi.
    Use estritamente estes dados para responder de forma analítica e não invente números.
    DADOS DO BANCO: {contexto_dados}
    """

    # Inicialização das IAs
    ia_google = genai.Client(api_key=gemini_key) if gemini_key else None
    ia_groq = Groq(api_key=groq_key) if groq_key else None
    
    historico = [{"role": "system", "content": prompt_sistema}]

    print("\n" + "="*65)
    print("🤖 Assistente Financeiro CITi IA (Modo Blindado) - ONLINE")
    print("="*65)

    while True:
        pergunta = input("\n👤 Pergunta: ")
        if pergunta.strip().lower() == 'sair': break
        if not pergunta.strip(): continue
        
        logger.info("A pensar...")
        sucesso = False

        # TENTATIVA 1: GEMINI
        if ia_google:
            try:
                chat = ia_google.chats.create(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(system_instruction=prompt_sistema, temperature=0.2)
                )
                resposta = chat.send_message(pergunta)
                print(f"\n✅ RESPOSTA (Google Gemini):\n{resposta.text}\n")
                sucesso = True
            except Exception as e:
                logger.warning(f"Gemini indisponível: {e}. A tentar Groq...")

        # TENTATIVA 2: GROQ
        if not sucesso and ia_groq:
            try:
                historico.append({"role": "user", "content": pergunta})
                resposta = ia_groq.chat.completions.create(
                    model="llama3-70b-8192", 
                    messages=historico, 
                    temperature=0.2
                )
                txt = resposta.choices[0].message.content
                historico.append({"role": "assistant", "content": txt})
                print(f"\n✅ RESPOSTA (Groq / Llama 3):\n{txt}\n")
                sucesso = True
            except Exception as e:
                logger.critical(f"Groq falhou (Verifique se a chave é válida!): {e}")

        if not sucesso:
            print("\n❌ Nenhuma API respondeu. Reveja as suas chaves no .env")

if __name__ == "__main__":
    iniciar_assistente()