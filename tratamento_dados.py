import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
import warnings

# Ocultar o aviso de data do Pandas (UserWarning) para manter o terminal limpo
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# ==========================================
# CONFIGURAÇÃO DE LOGS PROFISSIONAIS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, file_name: str):
        """Inicializa a pipeline e valida a existência do ficheiro dinamicamente."""
        self.base_dir = Path(__file__).parent.resolve()
        self.file_path = self.base_dir / file_name
        
        logger.info(f"A procurar a base de dados em: {self.file_path}")
        
        if not self.file_path.exists():
            logger.error(f"Ficheiro não encontrado! Verifique se '{file_name}' está na pasta.")
            raise FileNotFoundError(f"Ficheiro ausente: {self.file_path}")
            
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8')
            logger.info(f"Sucesso! Base carregada com {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.")
        except Exception as e:
            logger.critical(f"Falha ao ler o CSV. Erro: {e}")
            raise

    def sprint_1_diagnostico(self):
        """Exibe os problemas estruturais iniciais (Valores Nulos e Duplicatas)."""
        logger.info("--- INÍCIO DO SPRINT 1: DIAGNÓSTICO ---")
        nulos = self.df.isnull().sum().sum()
        duplicadas = self.df.duplicated().sum()
        logger.info(f"Total de valores nulos encontrados: {nulos}")
        logger.info(f"Total de linhas 100% duplicadas: {duplicadas}")

    def sprint_2_limpeza_estrutural(self):
        """Padronização de colunas, remoção de lixo, CPFs, Nomes, Datas e Parcelas."""
        logger.info("--- INÍCIO DO SPRINT 2: LIMPEZA ESTRUTURAL ---")
        
        # 1. Limpeza de colunas
        self.df.columns = self.df.columns.str.strip()
        
        # 2 e 3. Remoção de linhas vazias e duplicatas
        linhas_antes = len(self.df)
        self.df.dropna(how='all', inplace=True)
        self.df.drop_duplicates(inplace=True)
        linhas_depois = len(self.df)
        logger.info(f"Removidas {linhas_antes - linhas_depois} linhas (vazias ou duplicadas).")

        # 4. CPF
        self.df['CPF_Cliente'] = self.df['CPF_Cliente'].apply(self._formatar_cpf)
        
        # 5. Datas (format='mixed' trata a ambiguidade)
        self.df['Data_Transacao'] = pd.to_datetime(
            self.df['Data_Transacao'], 
            dayfirst=True,
            format='mixed',
            errors='coerce'
        ).dt.strftime('%Y-%m-%d')
        
        # 9. Nomes
        self.df['Nome_Cliente'] = self.df['Nome_Cliente'].apply(self._formatar_nome)
        
        # 10. Parcelas
        self.df['Num_Parcelas'] = self.df['Num_Parcelas'].apply(self._formatar_parcelas)
        
        logger.info("Sprint 2 Finalizado: CPFs, Datas, Nomes e Parcelas padronizados.")

    def sprint_3_regras_negocio(self):
        """Conversão cambial, remoção de outliers, mapeamento de status e recálculo financeiro."""
        logger.info("--- INÍCIO DO SPRINT 3: REGRAS DE NEGÓCIO ---")
        
        # 6. Valores e Moedas (Aplicação da nova função de conversão robusta)
        self.df['Valor_Transacao'] = self.df['Valor_Transacao'].apply(self._formatar_valor)
        
        # Tratamento de outliers: Valores <= 0 ou absurdamente altos viram nulos
        self.df['Valor_Transacao'] = self.df['Valor_Transacao'].apply(lambda x: np.nan if (x <= 0 or x >= 9999999) else x)
        
        taxas_cambio = {'USD': 5.00, 'EUR': 5.50, 'GBP': 6.30, 'BRL': 1.00}
        self.df['Moeda'] = self.df['Moeda'].fillna('BRL').str.strip().str.upper()
        
        # Conversão iterativa baseada na máscara
        for moeda, taxa in taxas_cambio.items():
            mask = self.df['Moeda'] == moeda
            self.df.loc[mask, 'Valor_Transacao'] *= taxa
            
        self.df['Moeda'] = 'BRL'
        mediana_valor = self.df['Valor_Transacao'].median()
        self.df['Valor_Transacao'] = self.df['Valor_Transacao'].fillna(mediana_valor)
        
        # 7. Tipos de Transação
        map_tipo = {
            r'(?i).*ted.*|.*doc.*|.*transf.*': 'Transferência',
            r'(?i).*boleto.*|.*pgto.*': 'Pagamento',
            r'(?i).*resgate.*|.*retirada.*|.*saque.*': 'Saque',
            r'(?i).*p-i-x.*|.*pix.*': 'PIX',
            r'(?i).*dep.*': 'Depósito'
        }
        self.df['Tipo_Transacao'] = self.df['Tipo_Transacao'].replace(map_tipo, regex=True)
        moda_tipo = self.df['Tipo_Transacao'].mode()[0]
        self.df['Tipo_Transacao'] = self.df['Tipo_Transacao'].fillna(moda_tipo).str.title()
        
        # 8. Status da Transação
        map_status = {
            r'(?i).*aprovad[ao].*|.*autorizada.*': 'Aprovada',
            r'(?i).*negada.*|.*bloqueada.*|.*recus.*': 'Recusada',
            r'(?i).*processamento.*|.*aguardando.*|.*pend.*': 'Pendente'
        }
        self.df['Status_Transacao'] = self.df['Status_Transacao'].replace(map_status, regex=True).str.title()
        
        # 11 e 12. Taxas Absolutas e Valor Final
        self.df['Taxa_Servico'] = self.df['Taxa_Servico'].abs()
        media_taxa = self.df['Taxa_Servico'].mean()
        self.df['Taxa_Servico'] = self.df['Taxa_Servico'].fillna(media_taxa)
        
        # Recálculo exigido no desafio
        self.df['Valor_Final'] = self.df['Valor_Transacao'] + self.df['Taxa_Servico']
        
        logger.info("Sprint 3 Finalizado: Conversão cambial efetuada e valores recalculados.")

    def sprint_4_exportacao(self, output_name: str = 'Base_Tratada_PTC_26.csv'):
        """Exporta os dados com tratamento de erro."""
        logger.info("--- INÍCIO DO SPRINT 4: EXPORTAÇÃO ---")
        output_path = self.base_dir / output_name
        try:
            self.df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"✅ EXCELENTE! Ficheiro exportado com sucesso para: {output_path}")
            logger.info(f"A base final possui {self.df.shape[0]} linhas perfeitamente tratadas.")
        except Exception as e:
            logger.critical(f"Erro ao salvar o ficheiro: {e}")

    # ==========================================
    # MÉTODOS DE FORMATAÇÃO PROTEGIDOS
    # ==========================================
    def _formatar_valor(self, val) -> float:
        """Lida inteligentemente com padrões monetários (BR vs US)"""
        if pd.isna(val) or str(val).strip() == '':
            return np.nan
            
        val_str = str(val).upper()
        # Remove letras (moedas) e símbolos
        val_str = re.sub(r'[A-Z\$]', '', val_str).strip()
        
        # Se contiver ponto E vírgula (ex: 5.903,58)
        if '.' in val_str and ',' in val_str:
            if val_str.rfind(',') > val_str.rfind('.'): # Padrão BR
                val_str = val_str.replace('.', '').replace(',', '.')
            else: # Padrão US (ex: 5,903.58)
                val_str = val_str.replace(',', '')
        elif ',' in val_str:
            # Apenas vírgula (ex: 1500,00)
            val_str = val_str.replace(',', '.')
            
        # Deixa apenas números, ponto decimal e sinal negativo
        val_str = re.sub(r'[^\d\.-]', '', val_str)
        
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    def _formatar_cpf(self, cpf) -> str:
        if pd.isna(cpf):
            return "Não informado"
        cpf_num = re.sub(r'\D', '', str(cpf))
        if len(cpf_num) == 11:
            return f"{cpf_num[:3]}.{cpf_num[3:6]}.{cpf_num[6:9]}-{cpf_num[9:]}"
        return "Não informado"

    def _formatar_nome(self, nome) -> str:
        if pd.isna(nome):
            return "Não informado"
        nome_limpo = str(nome).replace('_', ' ')
        return ' '.join(nome_limpo.split()).title()

    def _formatar_parcelas(self, val) -> int:
        if pd.isna(val):
            return 1
        val_str = str(val).lower()
        if any(word in val_str for word in ['única', 'unica', 'vista', '1x']):
            return 1
        nums = re.findall(r'\d+', val_str)
        return int(nums[0]) if nums else 1

# ==========================================
# GATILHO DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    try:
        nome_do_ficheiro = 'Base de Dados PTC 26.1 - Base_Financeira_PTC_26.csv'
        
        logger.info("A Iniciar a Pipeline de Dados do CITi...")
        pipeline = DataCleaner(nome_do_ficheiro)
        
        pipeline.sprint_1_diagnostico()
        pipeline.sprint_2_limpeza_estrutural()
        pipeline.sprint_3_regras_negocio()
        pipeline.sprint_4_exportacao()
        
        logger.info("Pipeline concluída com sucesso! Verifique a sua pasta.")
        
    except Exception as erro_fatal:
        logger.critical(f"A execução foi interrompida devido a um erro fatal: {erro_fatal}")