import os
import random
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Service Desk AI | Suporte de TI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TEMA DARK MODERNO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #081028;
        --bg-card: #0B1739;
        --bg-card-hover: #0F1F4C;
        --border-color: #1B254B;
        --text-primary: #FFFFFF;
        --text-secondary: #AEB9E1;
        --accent-purple: #CB3CFF;
        --accent-indigo: #7F25FB;
        --accent-cyan: #00C2FF;
        --badge-green-bg: rgba(5, 193, 104, 0.18);
        --badge-green-text: #14CA74;
        --badge-red-bg: rgba(255, 90, 101, 0.18);
        --badge-red-text: #FF5A65;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-primary);
    }

    /* Fundo Principal e Barra Lateral */
    .stApp {
        background-color: var(--bg-main) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: var(--bg-main) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Limpeza de Topo e Rodapé Nativo */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1340px;
    }

    /* Identidade Visual na Sidebar */
    .app-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 1.2rem;
        padding-left: 0.2rem;
    }
    .app-logo-icon {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: linear-gradient(135deg, #00C2FF 0%, #CB3CFF 100%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(203, 60, 255, 0.4);
    }

    /* Indicador de Status na Barra Lateral */
    .side-nav-active {
        background: linear-gradient(90deg, rgba(203, 60, 255, 0.16) 0%, rgba(11, 23, 57, 0.0) 100%);
        border-left: 3px solid #CB3CFF;
        padding: 0.65rem 0.9rem;
        border-radius: 0 8px 8px 0;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }

    /* Cabeçalho Superior */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .header-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.02em;
    }
    .header-sub {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0;
    }
    .header-actions {
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    .btn-dark-ghost {
        background: #0B1739;
        border: 1px solid #1B254B;
        color: #AEB9E1;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .btn-neon-purple {
        background: linear-gradient(90deg, #CB3CFF 0%, #7F25FB 100%);
        color: #FFFFFF;
        padding: 0.5rem 1.1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(203, 60, 255, 0.35);
    }

    /* 4 Cards Superiores de Métricas */
    .kpi-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.22);
    }
    .kpi-card:hover {
        border-color: #CB3CFF;
        transform: translateY(-2px);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.65rem;
    }
    .kpi-body {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .kpi-number {
        font-size: 1.55rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .badge-up {
        background: var(--badge-green-bg);
        color: var(--badge-green-text);
        border: 1px solid rgba(20, 202, 116, 0.3);
        padding: 0.15rem 0.45rem;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-down {
        background: var(--badge-red-bg);
        color: var(--badge-red-text);
        border: 1px solid rgba(255, 90, 101, 0.3);
        padding: 0.15rem 0.45rem;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* Painéis Internos */
    .content-panel {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Balões de Mensagem do Chat */
    div[data-testid="stChatMessage"] {
        background-color: #0B1739 !important;
        border: 1px solid #1B254B !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    /* Inputs e Seletores */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #0B1739 !important;
        border-color: #1B254B !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    /* Botões Primários e Secundários */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #CB3CFF 0%, #7F25FB 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(203, 60, 255, 0.35) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #0B1739 !important;
        border: 1px solid #1B254B !important;
        color: #AEB9E1 !important;
        border-radius: 8px !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #CB3CFF !important;
        color: #FFFFFF !important;
    }

    /* Barras de Relevância RAG */
    .rag-box-dark {
        background: #081028;
        border: 1px solid #1B254B;
        border-left: 3px solid #CB3CFF;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        font-size: 0.85rem;
        color: #AEB9E1;
    }
    .rag-progress-bg {
        background: #1B254B;
        height: 6px;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.4rem 0 0.65rem 0;
    }
    .rag-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00C2FF 0%, #CB3CFF 100%);
    }

    /* Abas (st.tabs) */
    button[data-baseweb="tab"] {
        color: #AEB9E1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADOS DA SESSÃO ---
if "chamados_abertos" not in st.session_state:
    st.session_state.chamados_abertos = []

if "resolvidos_n1" not in st.session_state:
    st.session_state.resolvidos_n1 = 0

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": (
                "Olá! Bem-vindo à **Central de Suporte de TI**. "
                "Selecione um dos atalhos de incidente acima ou descreva sua dúvida técnica abaixo."
            )
        }
    ]

# --- PERFIS DE PERSONA PRÉ-DEFINIDOS ---
PERSONAS_PREDEFINIDAS = {
    "🛡️ Analista N1 Padrão": (
        "Você é um Analista de Suporte de TI Nível 1 prestativo, claro e direto ao ponto. "
        "Forneça instruções passo a passo numeradas e destaque comandos ou botões em negrito."
    ),
    "🤝 Instrutor Didático (Leigos)": (
        "Você é um especialista de TI extremamente paciente e acolhedor que atende usuários sem "
        "conhecimento técnico. Evite jargões complexos, explique onde clicar de forma simples e "
        "tranquilize o colaborador durante todo o passo a passo."
    ),
    "💻 Engenheiro N2 / SysAdmin": (
        "Você é um Engenheiro de Infraestrutura e Redes Sênior (Nível 2/3). Responda com foco técnico "
        "aprofundado, detalhando comandos de terminal (PowerShell, CMD, Linux), portas de rede, "
        "logs de diagnóstico e causas-raiz do incidente."
    ),
    "⚡ Triagem Rápida": (
        "Você é um bot de triagem expressa de Service Desk. Responda em no máximo 4 tópicos curtos "
        "(bullet points) contendo apenas a ação imediata necessária, sem introduções longas."
    ),
    "✍️ Personalizado": ""
}

# --- BASE DE CONHECIMENTO PADRÃO ---
DOCUMENTOS_PADRAO = [
    {
        "titulo": "Política de Reset de Senha e Active Directory",
        "conteudo": (
            "Para redefinir a senha do Windows ou e-mail corporativo, acesse o portal https://id.empresa.com.br "
            "e clique em 'Esqueci minha senha'. É necessário validar o token via aplicativo Microsoft Authenticator. "
            "Caso a conta esteja bloqueada após 5 tentativas incorretas, o desbloqueio automático ocorre em 30 minutos "
            "ou pode ser solicitado imediatamente no ramal 4040 informando a matrícula."
        )
    },
    {
        "titulo": "Configuração e Erros de VPN (FortiClient)",
        "conteudo": (
            "Para conectar na VPN corporativa, abra o FortiClient, selecione o perfil 'Acesso_Remoto_Matriz' "
            "e insira seu usuário de rede e senha + código MFA. Se aparecer o erro 'Erro 45% - Falha de Certificado' "
            "ou 'Credenciais Inválidas', limpe o cache de rede executando 'ipconfig /flushdns' no prompt de comando e "
            "reinicie o serviço do FortiClient. O servidor de gateway padrão é vpn.empresa.com.br (Porta 10443)."
        )
    },
    {
        "titulo": "Rede Wi-Fi Corporativa e Visitantes",
        "conteudo": (
            "A rede Wi-Fi 'Corp-Segura' é exclusiva para notebooks homologados da empresa e autentica automaticamente "
            "via certificado digital. Celulares pessoais e visitantes devem se conectar na rede 'Corp-Guest'. "
            "A senha semanal da rede 'Corp-Guest' fica disponível na intranet (Menu TI - Wi-Fi) e todo acesso "
            "expira a cada 24 horas."
        )
    },
    {
        "titulo": "Mapeamento de Impressoras e Chamados de Hardware",
        "conteudo": (
            "Para instalar a impressora do seu andar, pressione Windows + R, digite \\\\servidor-imp01 e dê dois cliques "
            "no modelo correspondente ao seu setor (Ex: IMP_FINANCEIRO_2ANDAR). Em caso de atolamento de papel ou "
            "falta de toner, abra um chamado no sistema GLPI na categoria 'Hardware - Impressoras' informando o número "
            "de patrimônio colado na frente do equipamento."
        )
    },
    {
        "titulo": "Solicitação de Softwares e Licenças",
        "conteudo": (
            "Usuários não possuem permissão de administrador local para instalar programas. Softwares homologados "
            "(como VS Code, 7-Zip, Chrome e Power BI) podem ser instalados diretamente pelo aplicativo 'Portal da Empresa' "
            "sem precisar de senha de admin. Licenças pagas (Adobe, AutoCAD, Office Project) exigem aprovação do gestor "
            "via chamado no GLPI."
        )
    }
]

# --- DETECÇÃO AUTOMÁTICA DE PROVEDOR E LISTAGEM DE MODELOS ---
@st.cache_data(show_spinner=False)
def detectar_e_listar_modelos(chave: str):
    chave = chave.strip()
    if not chave:
        return None, [], None

    if chave.startswith("AIza"):
        try:
            from google import genai
            client = genai.Client(api_key=chave)
            modelos = [
                m.name.replace("models/", "") for m in client.models.list()
                if "gemini" in m.name.lower() and "embedding" not in m.name.lower()
            ]
            modelos.sort(reverse=True)
            return "Google Gemini", modelos, None
        except Exception as e:
            return "Google Gemini", [], f"Chave Gemini inválida: {e}"

    if chave.startswith("gsk_"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=chave, base_url="https://api.groq.com/openai/v1")
            lista = client.models.list()
            modelos = [
                m.id for m in lista.data
                if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "embed"])
            ]
            modelos.sort()
            return "Groq", modelos, None
        except Exception as e:
            return "Groq", [], f"Chave Groq inválida: {e}"

    if chave.startswith("sk-ant-"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=chave)
            lista = client.models.list()
            return "Anthropic", [m.id for m in lista.data], None
        except Exception as e:
            return "Anthropic", [], f"Chave Anthropic inválida: {e}"

    if chave.startswith("sk-or-"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=chave, base_url="https://openrouter.ai/api/v1")
            lista = client.models.list()
            return "OpenRouter", [m.id for m in lista.data], None
        except Exception as e:
            return "OpenRouter", [], f"Chave OpenRouter inválida: {e}"

    from openai import OpenAI
    try:
        client = OpenAI(api_key=chave)
        lista = client.models.list()
        prefixos_chat = ("gpt-", "o1", "o3", "o4", "chatgpt-")
        excluidos = ("instruct", "realtime", "audio", "tts", "whisper", "embedding", "moderation", "dall-e", "transcribe", "search")
        modelos = [
            m.id for m in lista.data
            if m.id.startswith(prefixos_chat) and not any(x in m.id for x in excluidos)
        ]
        modelos.sort(reverse=True)
        return "OpenAI", modelos, None
    except Exception:
        pass

    try:
        client = OpenAI(api_key=chave, base_url="https://api.deepseek.com/v1")
        lista = client.models.list()
        return "DeepSeek", [m.id for m in lista.data], None
    except Exception:
        pass

    return None, [], "Não foi possível autenticar a chave em nenhum provedor suportado."

# --- CHAMADA UNIVERSAL DE IA ---
def gerar_resposta_ia(provedor, chave, modelo_escolhido, prompt_sistema, pergunta, temperatura=0.2):
    chave = chave.strip()
    if provedor == "Google Gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=chave)
        resp = client.models.generate_content(
            model=modelo_escolhido,
            contents=pergunta,
            config=types.GenerateContentConfig(
                system_instruction=prompt_sistema,
                temperature=temperatura
            )
        )
        return resp.text

    elif provedor == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=chave)
        resp = client.messages.create(
            model=modelo_escolhido,
            max_tokens=1024,
            system=prompt_sistema,
            temperature=temperatura,
            messages=[{"role": "user", "content": pergunta}]
        )
        return resp.content[0].text

    else:
        from openai import OpenAI
        urls = {
            "Groq": "https://api.groq.com/openai/v1",
            "OpenRouter": "https://openrouter.ai/api/v1",
            "DeepSeek": "https://api.deepseek.com/v1",
            "OpenAI": None
        }
        base_url = urls.get(provedor)
        client = OpenAI(api_key=chave, base_url=base_url) if base_url else OpenAI(api_key=chave)
        resp = client.chat.completions.create(
            model=modelo_escolhido,
            temperature=temperatura,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pergunta}
            ]
        )
        return resp.choices[0].message.content

# --- MOTOR RAG LOCAL ---
def preparar_documentos(docs_padrao, uploads):
    chunks = []
    if os.path.exists("base_conhecimento_ti.txt"):
        with open("base_conhecimento_ti.txt", "r", encoding="utf-8", errors="ignore") as f:
            partes = [p.strip() for p in f.read().split("\n\n") if len(p.strip()) > 30]
            for p in partes:
                chunks.append(p)
    else:
        chunks = [f"[{d['titulo']}]\n{d['conteudo']}" for d in docs_padrao]

    if uploads:
        for arq in uploads:
            texto = arq.read().decode("utf-8", errors="ignore")
            partes = [p.strip() for p in texto.split("\n\n") if p.strip()]
            for i, parte in enumerate(partes):
                chunks.append(f"[Arquivo: {arq.name} - Seção {i+1}]\n{parte}")
    return chunks

def buscar_contexto_local(pergunta, lista_textos, top_k=3):
    vetorizador = TfidfVectorizer(ngram_range=(1, 2))
    matriz_docs = vetorizador.fit_transform(lista_textos)
    vetor_pergunta = vetorizador.transform([pergunta])

    similaridades = cosine_similarity(vetor_pergunta, matriz_docs).flatten()
    melhores_indices = np.argsort(similaridades)[::-1][:top_k]

    return [
        {"trecho": lista_textos[idx], "score": float(similaridades[idx])}
        for idx in melhores_indices
    ]

# --- RENDERIZAR FONTES RAG ---
def renderizar_fontes_rag(fontes):
    for f in fontes:
        pct = min(max(int(f["score"] * 100), 6), 100)
        st.markdown(
            f'<div class="rag-box-dark">'
            f'<div style="display:flex; justify-content:space-between; font-weight:700; font-size:0.76rem; color:#FFFFFF;">'
            f'<span>⚡ DOCUMENTO RECUPERADO</span><span style="color:#00C2FF;">{pct}% Relevância</span>'
            f'</div>'
            f'<div class="rag-progress-bg"><div class="rag-progress-fill" style="width:{pct}%;"></div></div>'
            f'<div style="white-space:pre-wrap;">{f["trecho"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# --- MODAL DE ABERTURA DE CHAMADO (GLPI / NÍVEL 2) ---
@st.dialog("🎫 Abrir Chamado • Suporte Nível 2")
def modal_abrir_chamado(indice_msg, problema_relatado, orientacao_ia):
    st.caption("Escalone este incidente para a fila Nível 2 do GLPI com o histórico anexado.")

    with st.form(f"form_chamado_{indice_msg}"):
        col_a, col_b = st.columns(2)
        with col_a:
            nome_solicitante = st.text_input("Solicitante / Matrícula *", placeholder="Ex: João Silva (Mat. 4520)")
            setor = st.selectbox("Setor *", ["TI / Desenvolvimento", "Financeiro", "RH", "Comercial", "Operações", "Outro"])
        with col_b:
            patrimonio = st.text_input("Patrimônio / Hostname", placeholder="Ex: NB-SOR-0142")
            prioridade = st.selectbox("Prioridade *", ["Média - Impacto parcial", "Alta - Trabalho parado", "Baixa - Solicitação"])

        categoria = st.selectbox(
            "Categoria do Incidente",
            ["Rede / VPN / Wi-Fi", "Acesso / Senha / MFA", "Hardware / Impressoras", "Software / Instalação", "Sistemas Internos / Banco de Dados"]
        )

        descricao_adicional = st.text_area(
            "Logs e Diagnóstico Prévio:",
            value=(
                f"Incidente relatado:\n{problema_relatado}\n\n"
                f"Triagem realizada pelo Chatbot N1:\n{orientacao_ia[:350]}..."
            ),
            height=135
        )

        confirmar = st.form_submit_button("🚀 Registrar Chamado Nível 2", use_container_width=True, type="primary")

        if confirmar:
            if not nome_solicitante.strip():
                st.error("Informe o nome ou matrícula do solicitante.")
            else:
                protocolo = f"INC-{datetime.now().strftime('%Y%m')}-{random.randint(1000, 9999)}"
                horario = datetime.now().strftime("%d/%m/%Y %H:%M")

                novo_chamado = {
                    "Protocolo": protocolo,
                    "Data/Hora": horario,
                    "Solicitante": nome_solicitante,
                    "Setor": setor,
                    "Patrimônio": patrimonio or "Não informado",
                    "Prioridade": prioridade,
                    "Categoria": categoria,
                    "Problema Original": problema_relatado,
                    "Descrição Completa": descricao_adicional,
                    "Status": "Aberto • Fila N2"
                }

                st.session_state.chamados_abertos.append(novo_chamado)
                st.session_state.mensagens[indice_msg]["status_resolucao"] = "chamado_aberto"
                st.session_state.mensagens[indice_msg]["protocolo_gerado"] = protocolo

                texto_confirmacao = (
                    f"### 🎫 Chamado `{protocolo}` Registrado!\n"
                    f"Seu atendimento foi encaminhado para a fila **Suporte Nível 2**.\n\n"
                    f"- **Solicitante:** {nome_solicitante} ({setor})\n"
                    f"- **Categoria:** {categoria} | **Prioridade:** {prioridade}\n"
                    f"- **Horário:** {horario}\n\n"
                    f"*Consulte a aba **📊 Relatórios & Chamados** para exportar a planilha completa.*"
                )
                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": texto_confirmacao,
                    "is_ticket": True,
                    "dados_chamado": novo_chamado
                })
                st.toast(f"Chamado {protocolo} registrado!", icon="🎫")
                st.rerun()

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("""
    <div class="app-logo">
        <div class="app-logo-icon">🛡️</div>
        <span>Service Desk AI</span>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "🔑 Chave de API Universal",
        type="password",
        placeholder="Cole sua chave (AIza, sk-, gsk_...)"
    )

    provedor_detectado = None
    modelo_selecionado = None

    if api_key:
        with st.spinner("Autenticando chave..."):
            provedor_detectado, modelos_disponiveis, erro = detectar_e_listar_modelos(api_key)

        if erro:
            st.error(erro)
        elif provedor_detectado and modelos_disponiveis:
            st.markdown(f'<div class="side-nav-active">🟣 {provedor_detectado} Conectado</div>', unsafe_allow_html=True)
            modelo_selecionado = st.selectbox("🧠 Modelo Selecionado", options=modelos_disponiveis)
    else:
        st.markdown('<div class="side-nav-active">🏠 Painel • Suporte de TI</div>', unsafe_allow_html=True)
        st.caption("Insira qualquer chave (Gemini, OpenAI, Groq, Claude) para listar as IAs.")

    st.divider()
    st.markdown("##### 🎭 Persona & Comportamento")

    perfil_escolhido = st.selectbox(
        "Perfil do Agente",
        options=list(PERSONAS_PREDEFINIDAS.keys()),
        label_visibility="collapsed"
    )

    texto_padrao_persona = (
        PERSONAS_PREDEFINIDAS[perfil_escolhido]
        if PERSONAS_PREDEFINIDAS[perfil_escolhido]
        else "Você é um especialista de Suporte Técnico. Responda com clareza, empatia e foco na resolução rápida."
    )

    with st.popover("⚙️ Configurar Instruções da IA", use_container_width=True):
        instrucao_persona = st.text_area(
            "Instrução de Comportamento (Persona):",
            value=texto_padrao_persona,
            height=130
        )
        temperatura_ia = st.slider("Temperatura (Criatividade)", 0.0, 1.0, 0.2, 0.1)
        top_k_slider = st.slider("Documentos RAG (Top-K)", 1, 5, 2)
        restringir_ao_rag = st.checkbox("Restringir estritamente ao RAG", value=True)

    st.divider()
    st.markdown("##### 📂 Upload de Manuais (RAG)")
    arquivos_enviados = st.file_uploader(
        "Anexar manuais (.txt, .md)",
        type=["txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    todos_os_chunks = preparar_documentos(DOCUMENTOS_PADRAO, arquivos_enviados)

    st.divider()
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.toast("Histórico de conversa reiniciado!", icon="🧹")
        st.rerun()

# --- CABEÇALHO SUPERIOR ---
st.markdown("""
<div class="top-header">
    <div>
        <div class="header-title">Central Inteligente de Suporte de TI</div>
        <div class="header-sub">Monitore métricas de suporte N1, consulte a base RAG e gerencie chamados GLPI.</div>
    </div>
    <div class="header-actions">
        <span class="btn-dark-ghost">SLA Nível 1 • Ramal 4040</span>
        <span class="btn-neon-purple">✦ Assistente IA Ativo</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4 CARDS DE MÉTRICAS SUPERIORES ---
k1, k2, k3, k4 = st.columns(4)

with k1:
    prov_nome = provedor_detectado if provedor_detectado else "Desconectado"
    badge_html = '<span class="badge-up">ONLINE ↗</span>' if provedor_detectado else '<span class="badge-down">SEM CHAVE ↘</span>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>👁️ Provedor de IA</span><span>•••</span></div>
        <div class="kpi-body">
            <div class="kpi-number">{prov_nome}</div>
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>📚 Base de Conhecimento</span><span>•••</span></div>
        <div class="kpi-body">
            <div class="kpi-number">{len(todos_os_chunks)}</div>
            <span class="badge-up">RAG ATIVO ↗</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>✅ Resolvidos no N1</span><span>•••</span></div>
        <div class="kpi-body">
            <div class="kpi-number">{st.session_state.resolvidos_n1}</div>
            <span class="badge-up">AUTO ↗</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    qtd_t = len(st.session_state.chamados_abertos)
    badge_t = f'<span class="badge-down">{qtd_t} ABERTO(S) ↘</span>' if qtd_t > 0 else '<span class="badge-up">0 FILA ↗</span>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>★ Chamados N2 (GLPI)</span><span>•••</span></div>
        <div class="kpi-body">
            <div class="kpi-number">{qtd_t}</div>
            {badge_t}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- NAVEGAÇÃO EM ABAS ---
aba_chat, aba_chamados, aba_rag = st.tabs([
    "💬 Atendimento IA",
    f"📊 Relatórios & Chamados ({len(st.session_state.chamados_abertos)})",
    f"🗂️ Explorador da Base RAG ({len(todos_os_chunks)})"
])

# ==========================================================
# ABA 1: CHAT DE SUPORTE E DIAGNÓSTICO
# ==========================================================
with aba_chat:
    opcoes_pilulas = [
        "🔑 Esqueci minha senha do Windows",
        "🌐 Erro 45% ao conectar no FortiClient VPN",
        "🖨️ Como mapear a impressora do meu setor?",
        "📶 Qual a senha do Wi-Fi para visitantes?",
        "💻 Como instalar o VS Code ou Power BI?"
    ]

    atalho_selecionado = st.pills(
        "⚡ Incidentes Frequentes:",
        options=opcoes_pilulas,
        selection_mode="single",
        key="pills_suporte"
    )

    st.write("")

    for idx, msg in enumerate(st.session_state.mensagens):
        avatar = "🧑‍💻" if msg["role"] == "user" else "🛡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg.get("is_ticket") and "dados_chamado" in msg:
                ch = msg["dados_chamado"]
                conteudo_txt = (
                    f"=== FICHA DE CHAMADO TÉCNICO (GLPI) ===\n"
                    f"Protocolo: {ch['Protocolo']}\n"
                    f"Data/Hora: {ch['Data/Hora']}\n"
                    f"Solicitante: {ch['Solicitante']} | Setor: {ch['Setor']}\n"
                    f"Patrimônio: {ch['Patrimônio']}\n"
                    f"Categoria: {ch['Categoria']} | Prioridade: {ch['Prioridade']}\n"
                    f"Status: {ch['Status']}\n"
                    f"----------------------------------------\n"
                    f"{ch['Descrição Completa']}\n"
                )
                st.download_button(
                    label=f"📥 Baixar Ficha do Chamado ({ch['Protocolo']}.txt)",
                    data=conteudo_txt,
                    file_name=f"{ch['Protocolo']}.txt",
                    mime="text/plain",
                    key=f"dl_chat_{ch['Protocolo']}_{idx}"
                )

            if "fontes" in msg:
                with st.expander("🔍 Ver Fontes da Base de Conhecimento Consultadas (RAG)"):
                    renderizar_fontes_rag(msg["fontes"])

            if msg.get("avaliavel"):
                status_res = msg.get("status_resolucao")
                if status_res == "resolvido":
                    st.success("✅ Incidente marcado como resolvido no Nível 1!")
                elif status_res == "chamado_aberto":
                    st.info(f"🎫 Escalonado para fila N2 • Protocolo **{msg.get('protocolo_gerado')}**.")
                else:
                    st.caption("**O diagnóstico acima solucionou o incidente?**")
                    col_ok, col_ticket, _ = st.columns([1.3, 1.9, 3])
                    if col_ok.button("✅ Sim, resolvido", key=f"btn_ok_{idx}", use_container_width=True):
                        st.session_state.mensagens[idx]["status_resolucao"] = "resolvido"
                        st.session_state.resolvidos_n1 += 1
                        st.toast("Métrica de resolução N1 atualizada!", icon="✅")
                        st.rerun()
                    if col_ticket.button("🎫 Abrir Chamado (N2)", key=f"btn_ticket_{idx}", type="primary", use_container_width=True):
                        modal_abrir_chamado(
                            indice_msg=idx,
                            problema_relatado=msg.get("pergunta_origem", "Incidente relatado no chat"),
                            orientacao_ia=msg["content"]
                        )

    pergunta_input = st.chat_input("Descreva seu problema técnico ou cole a mensagem de erro aqui...")

    if "ultimo_atalho_usado" not in st.session_state:
        st.session_state.ultimo_atalho_usado = None

    pergunta_ativa = None
    if pergunta_input:
        pergunta_ativa = pergunta_input
    elif atalho_selecionado and atalho_selecionado != st.session_state.ultimo_atalho_usado:
        st.session_state.ultimo_atalho_usado = atalho_selecionado
        pergunta_ativa = atalho_selecionado

    if pergunta_ativa:
        st.session_state.mensagens.append({"role": "user", "content": pergunta_ativa})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(pergunta_ativa)

        if not api_key or not modelo_selecionado:
            with st.chat_message("assistant", avatar="🛡️"):
                st.warning("⚠️ Insira sua **Chave de API** na barra lateral esquerda para habilitar os modelos.")
            st.stop()

        with st.chat_message("assistant", avatar="🛡️"):
            with st.status("Consultando Base RAG e Gerando Diagnóstico...", expanded=True) as status_box:
                try:
                    st.write("🔎 Calculando similaridade de cosseno nos manuais de TI...")
                    fontes_recuperadas = buscar_contexto_local(
                        pergunta_ativa, todos_os_chunks, top_k=top_k_slider
                    )
                    contexto_formatado = "\n\n---\n\n".join([f["trecho"] for f in fontes_recuperadas])

                    st.write(f"🎭 Carregando persona **{perfil_escolhido}**...")
                    if restringir_ao_rag:
                        regra_rag = (
                            "Responda à dúvida do colaborador baseando-se EXCLUSIVAMENTE nos manuais técnicos "
                            "fornecidos no CONTEXTO abaixo. Se a resposta não estiver no contexto, informe "
                            "educadamente que o procedimento não consta na base interna e oriente o usuário a clicar "
                            "no botão 'Abrir Chamado (N2)' logo abaixo da mensagem."
                        )
                    else:
                        regra_rag = (
                            "Priorize as informações fornecidas no CONTEXTO abaixo. Caso o procedimento exato não "
                            "esteja nos manuais internos, utilize seu conhecimento técnico de TI para sugerir um "
                            "diagnóstico seguro e lembre o usuário de que ele pode clicar em 'Abrir Chamado (N2)' se precisar."
                        )

                    prompt_sistema = (
                        f"### SUA PERSONA E INSTRUÇÕES DE COMPORTAMENTO:\n{instrucao_persona}\n\n"
                        f"### REGRAS DE CONSULTA À BASE:\n{regra_rag}\n\n"
                        f"### CONTEXTO DOS MANUAIS DE TI:\n{contexto_formatado}"
                    )

                    st.write(f"⚡ Executando modelo **{modelo_selecionado}** ({provedor_detectado})...")
                    texto_resposta = gerar_resposta_ia(
                        provedor=provedor_detectado,
                        chave=api_key,
                        modelo_escolhido=modelo_selecionado,
                        prompt_sistema=prompt_sistema,
                        pergunta=pergunta_ativa,
                        temperatura=temperatura_ia
                    )

                    status_box.update(label="✅ Diagnóstico concluído!", state="complete", expanded=False)

                    st.session_state.mensagens.append({
                        "role": "assistant",
                        "content": texto_resposta,
                        "fontes": fontes_recuperadas,
                        "avaliavel": True,
                        "pergunta_origem": pergunta_ativa,
                        "status_resolucao": None
                    })
                    st.rerun()

                except Exception as e:
                    status_box.update(label="❌ Erro na chamada da API", state="error", expanded=True)
                    st.error(f"Falha ao consultar `{modelo_selecionado}`: {e}")

# ==========================================================
# ABA 2: RELATÓRIOS E TABELA DE CHAMADOS
# ==========================================================
with aba_chamados:
    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    st.subheader("📊 Relatórios & Chamados Escalonados (Nível 2)")
    st.caption("Histórico de chamados gerados durante a sessão com exportação direta para planilha CSV.")

    if not st.session_state.chamados_abertos:
        st.info("Nenhum chamado N2 aberto até o momento. Use o botão **'🎫 Abrir Chamado (N2)'** abaixo de qualquer resposta no chat.")
    else:
        df_chamados = pd.DataFrame(st.session_state.chamados_abertos)
        st.dataframe(
            df_chamados[["Protocolo", "Data/Hora", "Solicitante", "Setor", "Categoria", "Prioridade", "Patrimônio", "Status"]],
            use_container_width=True,
            hide_index=True
        )

        csv_bytes = df_chamados.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Exportar Planilha de Chamados (CSV)",
            data=csv_bytes,
            file_name=f"chamados_suporte_ti_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# ABA 3: EXPLORADOR DA BASE RAG
# ==========================================================
with aba_rag:
    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    st.subheader("🗂️ Explorador da Base de Conhecimento (RAG)")
    st.caption("Inspecione os blocos de documentação indexados pelo motor vetorial TF-IDF.")

    termo_filtro = st.text_input("🔍 Pesquisar nos manuais (ex: VPN, senha, VLAN, SQL, impressora):", "")

    chunks_filtrados = [
        c for c in todos_os_chunks
        if termo_filtro.lower() in c.lower()
    ]

    st.write(f"Exibindo **{len(chunks_filtrados)}** de **{len(todos_os_chunks)}** seções:")

    for i, texto_bloco in enumerate(chunks_filtrados, 1):
        primeira_linha = texto_bloco.split("\n")[0][:95]
        with st.expander(f"📄 Seção #{i} — {primeira_linha}"):
            st.code(texto_bloco, language="markdown")
    st.markdown('</div>', unsafe_allow_html=True)
