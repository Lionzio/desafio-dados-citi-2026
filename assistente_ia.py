import os
import sys
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv

# Módulo de LLM do Google e Fallback (Groq)
from groq import Groq
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==========================================
# 0. CONFIGURAÇÕES GLOBAIS E TELEMETRIA
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. MÓDULO FAIL-FAST: SANITIZAÇÃO E VALIDAÇÃO 
# ==========================================
def validar_infraestrutura_chaves() -> dict:
    """Carrega as variáveis e faz early-return (Fail-Fast) em anomalias críticas de credenciais."""
    # O override garante a leitura forçada do .env atual
    load_dotenv(override=True)
    
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_key or not groq_key.startswith("gsk_"):
        logger.critical(f"GROQ_API_KEY inválida ou ausente. O formato exigido inicia com 'gsk_'. Valor lido: '{str(groq_key)[:4]}...'")
        sys.exit(1)
        
    if not gemini_key or not str(gemini_key).startswith("AIza"):
        logger.warning("GEMINI_API_KEY ausente ou inválida. O Fallback para o Google estará desativado.")
        gemini_key = None

    logger.info("Validação de chaves concluída. Groq (Primária) OK.")
    return {"groq": groq_key.strip(), "gemini": gemini_key.strip() if gemini_key else None}

# ==========================================
# 2. MÓDULO: GERAÇÃO DE CONTEXTO ESTATÍSTICO
# ==========================================
def gerar_sumario_estatistico(file_name: str = 'Base_Tratada_PTC_26.csv') -> str:
    """Extrai features macroeconômicas e anomalias via Pandas."""
    base_dir = Path(__file__).parent.resolve()
    file_path = base_dir / file_name

    if not file_path.exists():
        logger.error(f"Dataset '{file_name}' não encontrado. Execute o pipeline de tratamento primeiro.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.critical(f"Falha de I/O na leitura do dataframe: {e}")
        sys.exit(1)

    total_transacoes = len(df)
    valor_total = df['Valor_Final'].sum()

    status_counts = df['Status_Transacao'].value_counts()
    status_pct = df['Status_Transacao'].value_counts(normalize=True) * 100
    status_str = ", ".join([f"{k} {status_counts.get(k,0)} ({status_pct.get(k,0):.1f}%)" for k in status_counts.index])

    df_recusadas = df[df['Status_Transacao'] == 'Recusada']
    recusas_banco = df.groupby('Banco')['Status_Transacao'].apply(lambda x: (x == 'Recusada').sum())
    total_banco = df['Banco'].value_counts()
    taxa_recusa_banco = (recusas_banco / total_banco * 100).fillna(0).sort_values(ascending=False)
    bancos_str = ", ".join([f"{b} ({taxa_recusa_banco[b]:.1f}% recusa)" for b in taxa_recusa_banco.index[:5]])

    clientes_recusa = df_recusadas['Nome_Cliente'].value_counts().head(5)
    clientes_str = ", ".join([f"{k} ({v} recusas)" for k, v in clientes_recusa.items()])

    ticket_medio = df.groupby('Tipo_Transacao')['Valor_Final'].mean().round(2)
    ticket_str = ", ".join([f"{k} R${v}" for k, v in ticket_medio.items()])

    return f"""
    --- SUMÁRIO DE TELEMETRIA FINANCEIRA (CITi) ---
    - Total Operações: {total_transacoes}
    - Volume Financeiro Agregado: R$ {valor_total:,.2f}
    - KPI Status Geral: {status_str}
    - TOP 5 Gateways Bancários com maior taxa de falha/recusa: {bancos_str}
    - TOP 5 Perfis suspeitos (Clientes com altíssimas recusas): {clientes_str}
    - Ticket Médio por modalidade: {ticket_str}
    """

# ==========================================
# 3. MÓDULO ROTEAMENTO: INVERTIDO (GROQ PRIMARY)
# ==========================================
class AgentRouter:
    """Roteador para abstrair a chamada, priorizando Groq e caindo para Gemini em caso de falha."""
    def __init__(self, keys: dict, system_prompt: str):
        self.system_prompt = system_prompt
        self.history_groq = [{"role": "system", "content": system_prompt}]
        
        self.client_groq = Groq(api_key=keys["groq"])
        self.client_gemini = genai.Client(api_key=keys["gemini"]) if keys["gemini"] and GEMINI_AVAILABLE else None

    def query(self, user_prompt: str) -> str:
        """Tenta invocar a Groq (Llama 3). Em caso de erro, realiza Fallback para o Gemini."""
        try:
            # 1. TENTATIVA PRINCIPAL (Groq / Llama 3)
            logger.info("Acionando Endpoint da Groq (Llama-3.3-70b)...")
            
            self.history_groq.append({"role": "user", "content": user_prompt})
            response_groq = self.client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.history_groq,
                temperature=0.2
            )
            answer = response_groq.choices[0].message.content
            self.history_groq.append({"role": "assistant", "content": answer})
            
            return f"🟠 [Provider: Groq / Llama 3] ->\n{answer}"

        except Exception as e_groq:
            logger.warning(f"Groq API indisponível ou falhou: {e_groq}. Acionando Fallback para Gemini.")
            # Remove a pergunta do histórico da Groq para não corromper o array caso voltemos a ela
            self.history_groq.pop() 
            
            # 2. MECANISMO DE FALLBACK (Google Gemini)
            return self._fallback_gemini(user_prompt)

    def _fallback_gemini(self, user_prompt: str) -> str:
        if not self.client_gemini:
            return "🔴 Erro Crítico: A API Primária (Groq) falhou e o Fallback (Gemini) não está configurado."
            
        logger.info("Executando Fallback: Roteando para a API do Google (Gemini)...")
        try:
            # Forçando a versão atualizada da API para evitar o erro 404 de v1beta
            chat_session = self.client_gemini.chats.create(
                model="gemini-1.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.2
                )
            )
            response = chat_session.send_message(user_prompt)
            return f"🟢 [Provider: Google Gemini] ->\n{response.text}"

        except Exception as e_gemini:
            logger.critical(f"Falha em Cascade: Ambos os provedores falharam. Erro Gemini: {e_gemini}")
            return "🔴 Erro: O sistema não conseguiu processar sua requisição através das múltiplas redes redundantes de IA."

# ==========================================
# 4. INTERFACE E ENTRYPOINT
# ==========================================
def main():
    logger.info("Inicializando Arquitetura Multi-Modelo...")
    
    keys = validar_infraestrutura_chaves()
    context_string = gerar_sumario_estatistico()
    
    system_prompt = (
        "Atue como um Consultor Financeiro e Cientista de Dados Sênior em uma reunião com diretoria executiva do CITi. "
        "Você analisa bases de dados transacionais com precisão clínica. Utilize as métricas exatas abaixo para "
        "basear todo o seu raciocínio lógico e suas conclusões. Responda diretamente e seja estratégico.\n"
        f"MÉTRICAS EXATAS DISPONÍVEIS:\n{context_string}"
    )

    router = AgentRouter(keys=keys, system_prompt=system_prompt)

    print("\n" + "="*80)
    print("🤖 TERMINAL CLI: ASSISTENTE DE IA ESTATÍSTICO (GROQ PRIMARY)")
    print("="*80)
    print("O sistema extraiu as anomalias da Etapa 1 e está pronto para a sabatina.")
    print("Digite 'sair', 'exit' ou 'quit' para encerrar.")

    while True:
        try:
            user_input = input("\n>>> (CITi User): ").strip()
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessão. Excelente trabalho!")
                break
                
            if not user_input:
                continue

            resposta = router.query(user_input)
            
            print("\n" + "-"*80)
            print(resposta)
            print("-"*80)

        except KeyboardInterrupt:
            print("\nSessão interrompida pelo usuário.")
            sys.exit(0)

if __name__ == "__main__":
    main()