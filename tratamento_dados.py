import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Union
import warnings

# Supressão de warnings específicos do Pandas para manter stdout limpo
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# Configuração de Telemetria (Logging) centralizada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Módulo de engenharia de dados responsável pelo ETL local: extração, 
    higienização estrutural e aplicação de regras de negócio em dados financeiros.
    """
    def __init__(self, file_name: str) -> None:
        self.base_dir: Path = Path(__file__).parent.resolve()
        self.file_path: Path = self.base_dir / file_name
        
        logger.info(f"Localizando dataset alvo: {self.file_path}")
        
        if not self.file_path.exists():
            logger.error(f"Dataset não encontrado: '{file_name}'. Abortando pipeline.")
            raise FileNotFoundError(f"Arquivo ausente: {self.file_path}")
            
        try:
            self.df: pd.DataFrame = pd.read_csv(self.file_path, encoding='utf-8')
            logger.info(f"Dataset carregado com sucesso. Dimensões: {self.df.shape}")
        except pd.errors.EmptyDataError:
            logger.critical("O arquivo CSV está vazio.")
            raise
        except Exception as e:
            logger.critical(f"Falha inesperada ao ler o CSV: {e}")
            raise

    def sprint_1_diagnostico(self) -> None:
        """Diagnóstico rápido sobre integridade primária da estrutura."""
        logger.info("--- INÍCIO DO SPRINT 1: DIAGNÓSTICO ---")
        nulos = self.df.isnull().sum().sum()
        duplicadas = self.df.duplicated().sum()
        logger.info(f"Total de valores nulos detectados: {nulos}")
        logger.info(f"Total de linhas estritamente duplicadas: {duplicadas}")

    def sprint_2_limpeza_estrutural(self) -> None:
        """Sanitização das colunas, remoção de redundâncias e parser de tipos base."""
        logger.info("--- INÍCIO DO SPRINT 2: LIMPEZA ESTRUTURAL ---")
        
        self.df.columns = self.df.columns.str.strip()
        linhas_antes = len(self.df)
        
        self.df.dropna(how='all', inplace=True)
        self.df.drop_duplicates(inplace=True)
        
        linhas_depois = len(self.df)
        logger.info(f"Removidas {linhas_antes - linhas_depois} linhas anômalas (vazias/duplicadas).")
        
        self.df['CPF_Cliente'] = self.df['CPF_Cliente'].apply(self._formatar_cpf)
        
        # Casting de datas com coerção de erros (Nat) e cast robusto
        self.df['Data_Transacao'] = pd.to_datetime(
            self.df['Data_Transacao'], 
            dayfirst=True, 
            format='mixed', 
            errors='coerce'
        ).dt.strftime('%Y-%m-%d')
        
        self.df['Nome_Cliente'] = self.df['Nome_Cliente'].apply(self._formatar_nome)
        self.df['Num_Parcelas'] = self.df['Num_Parcelas'].apply(self._formatar_parcelas)
        
        logger.info("Sprint 2 Finalizado: Strings e Datas padronizados com sucesso.")

    def sprint_3_regras_negocio(self) -> None:
        """Implementação de cálculos financeiros, preenchimento de missing values e mapeamento de categorias."""
        logger.info("--- INÍCIO DO SPRINT 3: REGRAS DE NEGÓCIO ---")
        
        # Parser financeiro robusto
        self.df['Valor_Transacao'] = self.df['Valor_Transacao'].apply(self._formatar_valor)
        
        # Outlier Detection (Limites rígidos definidos pela área de negócios)
        self.df['Valor_Transacao'] = self.df['Valor_Transacao'].apply(
            lambda x: np.nan if (x <= 0 or x >= 9999999) else x
        )
        
        taxas_cambio = {'USD': 5.00, 'EUR': 5.50, 'GBP': 6.30, 'BRL': 1.00}
        self.df['Moeda'] = self.df['Moeda'].fillna('BRL').str.strip().str.upper()
        
        # Operação vetorizada para conversão cambial
        for moeda, taxa in taxas_cambio.items():
            mask = self.df['Moeda'] == moeda
            self.df.loc[mask, 'Valor_Transacao'] *= taxa
            
        self.df['Moeda'] = 'BRL'
        
        # Imputação de nulos baseada na mediana global da distribuição monetária
        mediana_valor = self.df['Valor_Transacao'].median()
        self.df['Valor_Transacao'] = self.df['Valor_Transacao'].fillna(mediana_valor)
        
        # Dicionários de mapeamento via Regex
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
        
        map_status = {
            r'(?i).*aprovad[ao].*|.*autorizada.*': 'Aprovada', 
            r'(?i).*negada.*|.*bloqueada.*|.*recus.*': 'Recusada', 
            r'(?i).*processamento.*|.*aguardando.*|.*pend.*': 'Pendente'
        }
        self.df['Status_Transacao'] = self.df['Status_Transacao'].replace(map_status, regex=True).str.title()
        
        # Sanitização de taxas lógicas (garante valores absolutos reais)
        self.df['Taxa_Servico'] = self.df['Taxa_Servico'].abs()
        media_taxa = self.df['Taxa_Servico'].mean()
        self.df['Taxa_Servico'] = self.df['Taxa_Servico'].fillna(media_taxa)
        
        # Geração de feature final
        self.df['Valor_Final'] = self.df['Valor_Transacao'] + self.df['Taxa_Servico']
        
        logger.info("Sprint 3 Finalizado: ETL de negócios concluído e métricas consolidadas.")

    def sprint_4_exportacao(self, output_name: str = 'Base_Tratada_PTC_26.csv') -> None:
        """Serializa o DataFrame em um arquivo persistente na base atual."""
        logger.info("--- INÍCIO DO SPRINT 4: EXPORTAÇÃO ---")
        output_path = self.base_dir / output_name
        try:
            self.df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"✅ SUCESSO: Dataset persistido em: {output_path}")
            logger.info(f"Entregável da Etapa 1 finalizado: {self.df.shape[0]} instâncias consolidadas.")
        except IOError as e:
            logger.critical(f"Erro de I/O ao gravar dataset final: {e}")

    # ==========================================
    # UTILITÁRIOS (Helpers Protegidos)
    # ==========================================
    def _formatar_valor(self, val: Union[str, float, int]) -> float:
        if pd.isna(val) or str(val).strip() == '': 
            return np.nan
        val_str = str(val).upper()
        val_str = re.sub(r'[A-Z\$]', '', val_str).strip()
        
        if '.' in val_str and ',' in val_str:
            if val_str.rfind(',') > val_str.rfind('.'): 
                val_str = val_str.replace('.', '').replace(',', '.')
            else: 
                val_str = val_str.replace(',', '')
        elif ',' in val_str: 
            val_str = val_str.replace(',', '.')
            
        val_str = re.sub(r'[^\d\.-]', '', val_str)
        try: 
            return float(val_str)
        except ValueError: 
            return np.nan

    def _formatar_cpf(self, cpf: Union[str, int]) -> str:
        if pd.isna(cpf): 
            return "Não informado"
        cpf_num = re.sub(r'\D', '', str(cpf))
        if len(cpf_num) == 11: 
            return f"{cpf_num[:3]}.{cpf_num[3:6]}.{cpf_num[6:9]}-{cpf_num[9:]}"
        return "Não informado"

    def _formatar_nome(self, nome: str) -> str:
        if pd.isna(nome): 
            return "Não informado"
        nome_limpo = str(nome).replace('_', ' ')
        return ' '.join(nome_limpo.split()).title()

    def _formatar_parcelas(self, val: Union[str, int]) -> int:
        if pd.isna(val): 
            return 1
        val_str = str(val).lower()
        if any(word in val_str for word in ['única', 'unica', 'vista', '1x']): 
            return 1
        nums = re.findall(r'\d+', val_str)
        return int(nums[0]) if nums else 1

if __name__ == "__main__":
    try:
        FILE_TARGET = 'Base de Dados PTC 26.1 - Base_Financeira_PTC_26.csv'
        logger.info("Iniciando a Pipeline de Engenharia de Dados...")
        
        pipeline = DataCleaner(FILE_TARGET)
        pipeline.sprint_1_diagnostico()
        pipeline.sprint_2_limpeza_estrutural()
        pipeline.sprint_3_regras_negocio()
        pipeline.sprint_4_exportacao()
        
        logger.info("Pipeline operou e encerrou sem erros críticos.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Execução interrompida prematuramente devido a: {e}")