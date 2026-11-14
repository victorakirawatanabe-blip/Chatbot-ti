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
    page_title="Service Desk AI | Enterprise Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS MODERNO (FONTE INTER, GLASSMORPHISM E LIMPEZA DE UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Limpeza de elementos padrão do Streamlit */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Hero Banner Compacto SaaS */
    .saas-header {
        background: linear-gradient(135deg, #090d16 0%, #172554 55%, #1d4ed8 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 30px -8px rgba(15, 23, 42, 0.45);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .saas-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.025em;
    }
    .saas-subtitle {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin: 0;
        font-weight: 400;
    }
    .saas-badges {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    .badge-pill {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Cards de KPI Superiores */
    .kpi-card {
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.8rem 1.1rem;
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-size: 0.96rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Balões de Chat Estilizados */
    div[data-testid="stChatMessage"] {
        background-color: rgba(148, 163, 184, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.8rem;
    }

    /* Cards de Fontes RAG com Barra Visual */
    .rag-source-box {
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        font-size: 0.86rem;
    }
    .rag-bar-bg {
        background: rgba(148, 163, 184, 0.25);
        height: 6px;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.35rem 0 0.6rem 0;
    }
    .rag-bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    /* Estilização das Abas (st.tabs) */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADOS DA SESSÃO ---
if "chamados_abertos" not in st.session_state:
    st.session_state.chamados_abertos = []

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": (
                "Olá! Bem-vindo à **Central Inteligente de Service Desk**. "
                "Clique em uma das pílulas de atalho acima ou descreva seu incidente técnico abaixo."
            )
        }
    ]

# --- PERFIS DE PERSONA PRÉ-DEFINIDOS ---
PERSONAS_PREDEFINIDAS = {
    "🛡️ Analista N1 Padrão (Objetivo)": (
        "Você é um Analista de Suporte de TI Nível 1 prestativo, claro e direto ao ponto. "
        "Forneça instruções passo a passo numeradas e destaque comandos ou botões em negrito."
    ),
    "🤝 Instrutor Didático (Para Leigos)": (
        "Você é um especialista de TI extremamente paciente e acolhedor que atende usuários sem "
        "conhecimento técnico. Evite jargões complexos, explique onde clicar de forma simples e "
        "tranquilize o colaborador durante todo o passo a passo."
    ),
    "💻 Engenheiro N2 / SysAdmin (Técnico)": (
        "Você é um Engenheiro de Infraestrutura e Redes Sênior (Nível 2/3). Responda com foco técnico "
        "aprofundado, detalhando comandos de terminal (PowerShell, CMD, Linux), portas de rede, "
        "logs de diagnóstico e causas-raiz do incidente."
    ),
    "⚡ Triagem Rápida (Ultra-Conciso)": (
        "Você é um bot de triagem expressa de Service Desk. Responda em no máximo 4 tópicos curtos "
        "(bullet points) contendo apenas a ação imediata necessária, sem introduções longas."
    ),
    "✍️ Personalizado (Escrever manualmente)": ""
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

# --- FUNÇÃO AUXILIAR PARA RENDERIZAR FONTES RAG COM BARRA DE PROGRESSO ---
def renderizar_fontes_rag(fontes):
    for f in fontes:
        pct = min(max(int(f["score"] * 100), 5), 100)
        cor_barra = "#10b981" if pct >= 35 else ("#f59e0b" if pct >= 18 else "#64748b")
        st.markdown(
            f'<div class="rag-source-box">'
            f'<div style="display:flex; justify-content:space-between; font-weight:700; font-size:0.78rem; color:#475569;">'
            f'<span>TRECHO RECUPERADO</span><span>Relevância: {pct}%</span>'
            f'</div>'
            f'<div class="rag-bar-bg"><div class="rag-bar-fill" style="width:{pct}%; background:{cor_barra};"></div></div>'
            f'<div style="white-space:pre-wrap;">{f["trecho"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# --- MODAL DE ABERTURA DE CHAMADO (GLPI / NÍVEL 2) ---
@st.dialog("🎫 Abrir Chamado para Suporte Nível 2")
def modal_abrir_chamado(indice_msg, problema_relatado, orientacao_ia):
    st.caption("Preencha os campos abaixo para escalonar o incidente diretamente para a fila Nível 2 do GLPI.")

    with st.form(f"form_chamado_{indice_msg}"):
        col_a, col_b = st.columns(2)
        with col_a:
            nome_solicitante = st.text_input("Nome / Matrícula *", placeholder="Ex: Victor Akira (Mat. 4520)")
            setor = st.selectbox("Setor *", ["TI / Desenvolvimento", "Financeiro", "RH", "Comercial", "Operações", "Outro"])
        with col_b:
            patrimonio = st.text_input("Patrimônio / Hostname", placeholder="Ex: NB-SOR-0142 ou N/A")
            prioridade = st.selectbox("Urgência *", ["Média - Impacto parcial", "Alta - Trabalho parado", "Baixa - Dúvida / Solicitação"])

        categoria = st.selectbox(
            "Categoria do Incidente",
            ["Rede / VPN / Wi-Fi", "Acesso / Senha / MFA", "Hardware / Impressoras", "Software / Instalação", "Sistemas Internos / Banco de Dados"]
        )

        descricao_adicional = st.text_area(
            "Histórico Técnico Anexado:",
            value=(
                f"Dúvida inicial do usuário:\n{problema_relatado}\n\n"
                f"Procedimento já orientado pelo Chatbot N1:\n{orientacao_ia[:350]}..."
            ),
            height=140
        )

        confirmar = st.form_submit_button("🚀 Registrar Chamado no GLPI", use_container_width=True, type="primary")

        if confirmar:
            if not nome_solicitante.strip():
                st.error("Por favor, informe seu Nome ou Matrícula.")
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
                    "Status": "Aberto (Fila N2)"
                }

                st.session_state.chamados_abertos.append(novo_chamado)
                st.session_state.mensagens[indice_msg]["status_resolucao"] = "chamado_aberto"
                st.session_state.mensagens[indice_msg]["protocolo_gerado"] = protocolo

                texto_confirmacao = (
                    f"### 🎫 Chamado `{protocolo}` Registrado!\n"
                    f"Seu atendimento foi escalonado para a equipe de **Suporte Nível 2**.\n\n"
                    f"- **Solicitante:** {nome_solicitante} ({setor})\n"
                    f"- **Categoria:** {categoria} | **Prioridade:** {prioridade}\n"
                    f"- **Data/Hora:** {horario}\n\n"
                    f"*Você também pode acompanhar ou exportar seus tickets na aba **🎫 Fila de Chamados (GLPI)** no topo da página.*"
                )
                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": texto_confirmacao,
                    "is_ticket": True,
                    "dados_chamado": novo_chamado
                })
                st.toast(f"Chamado {protocolo} criado na fila N2!", icon="🎫")
                st.rerun()

# --- BARRA LATERAL ENXUTA E MODERNA ---
with st.sidebar:
    st.markdown("### 🛡️ Painel de Controle")
    api_key = st.text_input(
        "🔑 Chave de API Universal",
        type="password",
        placeholder="Cole sua key (AIza, sk-, gsk_...)"
    )

    provedor_detectado = None
    modelo_selecionado = None

    if api_key:
        with st.spinner("Validando credencial..."):
            provedor_detectado, modelos_disponiveis, erro = detectar_e_listar_modelos(api_key)

        if erro:
            st.error(erro)
        elif provedor_detectado and modelos_disponiveis:
            st.success(f"Provedor: **{provedor_detectado}**", icon="✅")
            modelo_selecionado = st.selectbox("🧠 Modelo Ativo:", options=modelos_disponiveis)
    else:
        st.info("Cole qualquer API Key acima para liberar os modelos automaticamente.")

    st.divider()
    st.markdown("### 🎭 Persona do Assistente")

    perfil_escolhido = st.selectbox(
        "Perfil de atendimento:",
        options=list(PERSONAS_PREDEFINIDAS.keys())
    )

    texto_padrao_persona = (
        PERSONAS_PREDEFINIDAS[perfil_escolhido]
        if PERSONAS_PREDEFINIDAS[perfil_escolhido]
        else "Você é um especialista de Suporte Técnico. Responda com clareza, empatia e foco na resolução rápida."
    )

    # Popover flutuante para manter a sidebar limpa
    with st.popover("⚙️ Personalizar Instruções & RAG", use_container_width=True):
        instrucao_persona = st.text_area(
            "System Prompt (Instrução da Persona):",
            value=texto_padrao_persona,
            height=130
        )
        temperatura_ia = st.slider("Criatividade (Temperatura)", 0.0, 1.0, 0.2, 0.1)
        top_k_slider = st.slider("Trechos consultados no RAG (Top-K)", 1, 5, 2)
        restringir_ao_rag = st.checkbox("Restringir 100% aos manuais (RAG)", value=True)

    st.divider()
    st.markdown("### 📤 Upload Rápido de Manuais")
    arquivos_enviados = st.file_uploader(
        "Anexar arquivos (.txt ou .md)",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    todos_os_chunks = preparar_documentos(DOCUMENTOS_PADRAO, arquivos_enviados)

    st.divider()
    if st.button("🗑️ Limpar Sessão de Chat", use_container_width=True):
        st.session_state.mensagens = []
        st.toast("Histórico de conversa reiniciado!", icon="🧹")
        st.rerun()

# --- HERO BANNER E KPI CARDS ---
st.markdown("""
<div class="saas-header">
    <div>
        <div class="saas-title">🛡️ Service Desk AI • Central de Suporte Técnico</div>
        <div class="saas-subtitle">Diagnóstico inteligente com RAG local, persona dinâmica e integração Nível 2 (GLPI)</div>
    </div>
    <div class="saas-badges">
        <span class="badge-pill">⚡ SLA Nível 1</span>
        <span class="badge-pill">📞 Ramal 4040</span>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    status_prov = f"🟢 {provedor_detectado}" if provedor_detectado else "🟡 Sem Chave"
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Provedor</div><div class="kpi-value">{status_prov}</div></div>', unsafe_allow_html=True)
with c2:
    status_mod = modelo_selecionado if modelo_selecionado else "Aguardando..."
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Modelo Ativo</div><div class="kpi-value">🤖 {status_mod}</div></div>', unsafe_allow_html=True)
with c3:
    nome_curto_persona = perfil_escolhido.split("(")[0].strip()
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Persona</div><div class="kpi-value">{nome_curto_persona}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Base RAG</div><div class="kpi-value">📚 {len(todos_os_chunks)} blocos</div></div>', unsafe_allow_html=True)
with c5:
    qtd_tickets = len(st.session_state.chamados_abertos)
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Fila Nível 2</div><div class="kpi-value">🎫 {qtd_tickets} ticket(s)</div></div>', unsafe_allow_html=True)

st.write("")

# --- NAVEGAÇÃO PRINCIPAL EM ABAS (st.tabs) ---
aba_chat, aba_chamados, aba_rag = st.tabs([
    "💬 Atendimento Inteligente",
    f"🎫 Fila de Chamados GLPI ({len(st.session_state.chamados_abertos)})",
    f"📚 Explorador da Base RAG ({len(todos_os_chunks)})"
])

# ==========================================================
# ABA 1: CHAT DE ATENDIMENTO COM PÍLULAS E ST.STATUS
# ==========================================================
with aba_chat:
    opcoes_pilulas = [
        "🔑 Esqueci minha senha do Windows",
        "🌐 Erro 45% ao conectar no FortiClient VPN",
        "🖨️ Como mapear a impressora do meu setor?",
        "📶 Qual a senha do Wi-Fi para visitantes?",
        "💻 Como instalar o VS Code ou Power BI?"
    ]

    # Componente st.pills para atalhos rápidos modernos
    atalho_selecionado = st.pills(
        "⚡ Atalhos de Incidentes Frequentes:",
        options=opcoes_pilulas,
        selection_mode="single",
        key="pills_atalho"
    )

    st.divider()

    # Renderização das mensagens do chat
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
                    label=f"📥 Baixar Ficha Técnica ({ch['Protocolo']}.txt)",
                    data=conteudo_txt,
                    file_name=f"{ch['Protocolo']}.txt",
                    mime="text/plain",
                    key=f"dl_chat_{ch['Protocolo']}_{idx}"
                )

            if "fontes" in msg:
                with st.expander("🔍 Inspecionar trechos consultados na Base de Conhecimento (RAG)"):
                    renderizar_fontes_rag(msg["fontes"])

            if msg.get("avaliavel"):
                status_res = msg.get("status_resolucao")
                if status_res == "resolvido":
                    st.success("✅ Incidente resolvido no Primeiro Nível (N1)!")
                elif status_res == "chamado_aberto":
                    st.info(f"🎫 Escalonado para o Suporte Nível 2 sob o protocolo **{msg.get('protocolo_gerado')}**.")
                else:
                    st.caption("**O procedimento acima resolveu o seu problema?**")
                    col_ok, col_ticket, _ = st.columns([1.3, 1.9, 3])
                    if col_ok.button("✅ Sim, resolvido!", key=f"btn_ok_{idx}", use_container_width=True):
                        st.session_state.mensagens[idx]["status_resolucao"] = "resolvido"
                        st.toast("Ótimo! Atendimento concluído com sucesso.", icon="🎉")
                        st.rerun()
                    if col_ticket.button("🎫 Não, abrir chamado N2", key=f"btn_ticket_{idx}", type="primary", use_container_width=True):
                        modal_abrir_chamado(
                            indice_msg=idx,
                            problema_relatado=msg.get("pergunta_origem", "Problema relatado no chat"),
                            orientacao_ia=msg["content"]
                        )

    # Entrada de texto ou disparo por pílula
    pergunta_input = st.chat_input("Descreva seu problema técnico ou cole o código de erro aqui...")

    # Evita loop infinito ao clicar na pílula
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
                st.warning("⚠️ Insira uma **Chave de API válida** no menu lateral esquerdo para liberar o assistente.")
            st.stop()

        with st.chat_message("assistant", avatar="🛡️"):
            # Componente moderno st.status mostrando as etapas em tempo real
            with st.status("Processando diagnóstico técnico...", expanded=True) as status_box:
                try:
                    st.write("🔎 Vetorizando pergunta e buscando nos manuais de TI (RAG)...")
                    fontes_recuperadas = buscar_contexto_local(
                        pergunta_ativa, todos_os_chunks, top_k=top_k_slider
                    )
                    contexto_formatado = "\n\n---\n\n".join([f["trecho"] for f in fontes_recuperadas])

                    st.write(f"🎭 Aplicando persona **{nome_curto_persona}**...")
                    if restringir_ao_rag:
                        regra_rag = (
                            "Responda à dúvida do colaborador baseando-se EXCLUSIVAMENTE nos manuais técnicos "
                            "fornecidos no CONTEXTO abaixo. Se a resposta não estiver no contexto, informe "
                            "educadamente que o procedimento não consta na base interna e oriente o usuário a clicar "
                            "no botão 'Não, abrir chamado N2' logo abaixo da mensagem."
                        )
                    else:
                        regra_rag = (
                            "Priorize as informações fornecidas no CONTEXTO abaixo. Caso o procedimento exato não "
                            "esteja nos manuais internos, utilize seu conhecimento técnico de TI para sugerir um "
                            "diagnóstico seguro e lembre o usuário de que ele pode clicar em 'Não, abrir chamado N2' se precisar."
                        )

                    prompt_sistema = (
                        f"### SUA PERSONA E INSTRUÇÕES DE COMPORTAMENTO:\n{instrucao_persona}\n\n"
                        f"### REGRAS DE CONSULTA À BASE:\n{regra_rag}\n\n"
                        f"### CONTEXTO DOS MANUAIS DE TI:\n{contexto_formatado}"
                    )

                    st.write(f"🤖 Gerando solução com **{modelo_selecionado}** ({provedor_detectado})...")
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
                    status_box.update(label="❌ Falha na comunicação com a API", state="error", expanded=True)
                    st.error(f"Erro ao gerar resposta com `{modelo_selecionado}`: {e}")

# ==========================================================
# ABA 2: CENTRAL E TABELA DE CHAMADOS (GLPI)
# ==========================================================
with aba_chamados:
    st.subheader("🎫 Gestão de Chamados Escalonados (Nível 2)")
    st.caption("Visualize todos os tickets abertos durante a sessão e exporte a planilha para integração com o GLPI.")

    if not st.session_state.chamados_abertos:
        st.info("Nenhum chamado foi escalonado ainda. Caso o chatbot não resolva uma dúvida na aba de Atendimento, clique em **'🎫 Não, abrir chamado N2'**.")
    else:
        df_chamados = pd.DataFrame(st.session_state.chamados_abertos)
        st.dataframe(
            df_chamados[["Protocolo", "Data/Hora", "Solicitante", "Setor", "Categoria", "Prioridade", "Patrimônio", "Status"]],
            use_container_width=True,
            hide_index=True
        )

        csv_bytes = df_chamados.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📊 Exportar Fila Completa em CSV (Excel)",
            data=csv_bytes,
            file_name=f"chamados_glpi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )

# ==========================================================
# ABA 3: EXPLORADOR DA BASE DE CONHECIMENTO (RAG)
# ==========================================================
with aba_rag:
    st.subheader("📚 Explorador da Base de Conhecimento Indexada")
    st.caption("Pesquise e inspecione os blocos de documentação que o motor TF-IDF utiliza para fundamentar as respostas.")

    termo_filtro = st.text_input("🔍 Filtrar blocos por palavra-chave (ex: VPN, senha, VLAN, SQL, impressora):", "")

    chunks_filtrados = [
        c for c in todos_os_chunks
        if termo_filtro.lower() in c.lower()
    ]

    st.write(f"Exibindo **{len(chunks_filtrados)}** de **{len(todos_os_chunks)}** blocos indexados:")

    for i, texto_bloco in enumerate(chunks_filtrados, 1):
        primeira_linha = texto_bloco.split("\n")[0][:95]
        with st.expander(f"📄 Bloco #{i} — {primeira_linha}"):
            st.code(texto_bloco, language="markdown")
