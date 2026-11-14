import os
import random
from datetime import datetime
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Service Desk AI | Suporte & Chamados",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS CUSTOMIZADA ---
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin: 0;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.05rem;
        font-weight: 700;
    }
    .rag-card {
        background: rgba(30, 41, 59, 0.04);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    .rag-score {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
    }
    .ticket-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 5px solid #10b981;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADOS ---
if "chamados_abertos" not in st.session_state:
    st.session_state.chamados_abertos = []

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
def gerar_resposta_ia(provedor, chave, modelo_escolhido, prompt_sistema, pergunta):
    chave = chave.strip()
    if provedor == "Google Gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=chave)
        resp = client.models.generate_content(
            model=modelo_escolhido,
            contents=pergunta,
            config=types.GenerateContentConfig(system_instruction=prompt_sistema, temperature=0.2)
        )
        return resp.text

    elif provedor == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=chave)
        resp = client.messages.create(
            model=modelo_escolhido,
            max_tokens=1024,
            system=prompt_sistema,
            temperature=0.2,
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
            temperature=0.2,
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

# --- MODAL DE ABERTURA DE CHAMADO (GLPI / NÍVEL 2) ---
@st.dialog("🎫 Abrir Chamado para Suporte Nível 2")
def modal_abrir_chamado(indice_msg, problema_relatado, orientacao_ia):
    st.caption("Como o autoatendimento não resolveu o incidente, preencha os dados abaixo para escalonar ao time técnico.")

    with st.form(f"form_chamado_{indice_msg}"):
        col_a, col_b = st.columns(2)
        with col_a:
            nome_solicitante = st.text_input("Nome / Matrícula *", placeholder="Ex: João Silva (Mat. 4520)")
            setor = st.selectbox("Setor *", ["Financeiro", "TI / Desenvolvimento", "RH", "Comercial", "Operações", "Outro"])
        with col_b:
            patrimonio = st.text_input("Patrimônio / Hostname", placeholder="Ex: NB-SOR-0142 ou N/A")
            prioridade = st.selectbox("Urgência *", ["Média - Impacto parcial", "Alta - Trabalho parado", "Baixa - Dúvida / Solicitação"])

        categoria = st.selectbox(
            "Categoria do Incidente",
            ["Rede / VPN / Wi-Fi", "Acesso / Senha / MFA", "Hardware / Impressoras", "Software / Instalação", "Sistemas Internos / Banco de Dados"]
        )

        descricao_adicional = st.text_area(
            "Detalhes do problema (pré-preenchido pelo chat):",
            value=(
                f"Dúvida inicial do usuário:\n{problema_relatado}\n\n"
                f"Procedimento já tentado via Chatbot N1:\n{orientacao_ia[:350]}..."
            ),
            height=150
        )

        confirmar = st.form_submit_button("🚀 Registrar Chamado no Service Desk", use_container_width=True, type="primary")

        if confirmar:
            if not nome_solicitante.strip():
                st.error("Por favor, informe seu Nome ou Matrícula.")
            else:
                protocolo = f"INC-{datetime.now().strftime('%Y%m')}-{random.randint(1000, 9999)}"
                horario = datetime.now().strftime("%d/%m/%Y %H:%M")

                novo_chamado = {
                    "protocolo": protocolo,
                    "data_hora": horario,
                    "solicitante": nome_solicitante,
                    "setor": setor,
                    "patrimonio": patrimonio or "Não informado",
                    "prioridade": prioridade,
                    "categoria": categoria,
                    "problema_original": problema_relatado,
                    "descricao": descricao_adicional,
                    "status": "Aberto - Fila Nível 2"
                }

                st.session_state.chamados_abertos.append(novo_chamado)
                st.session_state.mensagens[indice_msg]["status_resolucao"] = "chamado_aberto"
                st.session_state.mensagens[indice_msg]["protocolo_gerado"] = protocolo

                # Adiciona mensagem de confirmação no próprio chat
                texto_confirmacao = (
                    f"### 🎫 Chamado Registrado com Sucesso!\n"
                    f"Seu atendimento foi escalonado para a equipe de **Suporte Nível 2**.\n\n"
                    f"- **Protocolo:** `{protocolo}`\n"
                    f"- **Solicitante:** {nome_solicitante} ({setor})\n"
                    f"- **Categoria:** {categoria} | **Prioridade:** {prioridade}\n"
                    f"- **Data/Hora:** {horario}\n\n"
                    f"*Um analista entrará em contato via Teams ou ramal interno em breve.*"
                )
                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": texto_confirmacao,
                    "is_ticket": True,
                    "dados_chamado": novo_chamado
                })
                st.rerun()

# --- BARRA LATERAL ---
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
            st.success(f"Conectado: **{provedor_detectado}**", icon="✅")
            modelo_selecionado = st.selectbox("🧠 Modelo Ativo:", options=modelos_disponiveis)
    else:
        st.info("Cole qualquer API Key acima para liberar os modelos automaticamente.")

    st.divider()
    st.markdown("### 🎫 Chamados Abertos")
    if not st.session_state.chamados_abertos:
        st.caption("Nenhum chamado aberto nesta sessão.")
    else:
        for ch in reversed(st.session_state.chamados_abertos):
            with st.expander(f"🟢 {ch['protocolo']} ({ch['setor']})"):
                st.write(f"**Solicitante:** {ch['solicitante']}")
                st.write(f"**Categoria:** {ch['categoria']}")
                st.write(f"**Urgência:** {ch['prioridade']}")
                st.caption(f"Criado em {ch['data_hora']}")

    st.divider()
    st.markdown("### 📚 Base de Conhecimento")
    top_k_slider = st.slider("Trechos consultados (Top-K)", 1, 5, 2)

    arquivos_enviados = st.file_uploader(
        "Anexar manuais (.txt ou .md)",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    todos_os_chunks = preparar_documentos(DOCUMENTOS_PADRAO, arquivos_enviados)

    if st.button("🗑️ Reiniciar Atendimento", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

# --- CABEÇALHO E CARDS DE STATUS ---
st.markdown("""
<div class="hero-banner">
    <div>
        <div class="hero-title">🛡️ Central Inteligente de Suporte de TI</div>
        <div class="hero-subtitle">Diagnóstico rápido Nível 1 com RAG e abertura integrada de chamados Nível 2</div>
    </div>
    <div class="hero-badge">SLA Nível 1 • Ramal 4040</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    status_prov = f"🟢 {provedor_detectado}" if provedor_detectado else "🟡 Aguardando Chave"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Provedor</div><div class="metric-value">{status_prov}</div></div>', unsafe_allow_html=True)
with col2:
    status_mod = modelo_selecionado if modelo_selecionado else "Nenhum selecionado"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Modelo IA</div><div class="metric-value">🤖 {status_mod}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Base RAG Indexada</div><div class="metric-value">📚 {len(todos_os_chunks)} seções</div></div>', unsafe_allow_html=True)
with col4:
    qtd_tickets = len(st.session_state.chamados_abertos)
    st.markdown(f'<div class="metric-card"><div class="metric-label">Chamados na Sessão</div><div class="metric-value">🎫 {qtd_tickets} aberto(s)</div></div>', unsafe_allow_html=True)

st.write("")

# --- ATALHOS DE CHAMADOS FREQUENTES ---
st.markdown("##### ⚡ Dúvidas Rápidas (Clique para testar):")
atalhos = [
    "🔑 Esqueci minha senha do Windows",
    "🌐 Erro 45% ao conectar no FortiClient VPN",
    "🖨️ Como mapear a impressora do meu setor?",
    "📶 Qual a senha do Wi-Fi para visitantes?"
]

cols_atalhos = st.columns(4)
pergunta_atalho = None
for i, texto_atalho in enumerate(atalhos):
    if cols_atalhos[i].button(texto_atalho, use_container_width=True):
        pergunta_atalho = texto_atalho

st.divider()

# --- HISTÓRICO E CHAT ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "Olá! Sou o assistente técnico do **Service Desk**. Escolha um dos atalhos acima ou descreva seu problema. Se minhas instruções não resolverem, abrirei um chamado para você na hora!"
        }
    ]

for idx, msg in enumerate(st.session_state.mensagens):
    avatar = "🧑‍💻" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        # Se for uma mensagem de confirmação de chamado, mostra botão para baixar o TXT do ticket
        if msg.get("is_ticket") and "dados_chamado" in msg:
            ch = msg["dados_chamado"]
            conteudo_txt = (
                f"=== FICHA DE CHAMADO TÉCNICO (GLPI) ===\n"
                f"Protocolo: {ch['protocolo']}\n"
                f"Data/Hora: {ch['data_hora']}\n"
                f"Solicitante: {ch['solicitante']} | Setor: {ch['setor']}\n"
                f"Patrimônio: {ch['patrimonio']}\n"
                f"Categoria: {ch['categoria']} | Prioridade: {ch['prioridade']}\n"
                f"Status: {ch['status']}\n"
                f"----------------------------------------\n"
                f"{ch['descricao']}\n"
            )
            st.download_button(
                label=f"📥 Baixar Ficha do Chamado ({ch['protocolo']}.txt)",
                data=conteudo_txt,
                file_name=f"{ch['protocolo']}.txt",
                mime="text/plain",
                key=f"dl_{ch['protocolo']}_{idx}"
            )

        # Exibe fontes consultadas pelo RAG
        if "fontes" in msg:
            with st.expander("🔍 Fontes da Base de Conhecimento utilizadas nesta resposta"):
                for f in msg["fontes"]:
                    pct = int(f["score"] * 100)
                    st.markdown(
                        f'<div class="rag-card">'
                        f'<span class="rag-score">Similaridade: {pct}%</span><br>'
                        f'{f["trecho"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Exibe botões de Resolução / Abertura de Chamado após respostas técnicas da IA
        if msg.get("avaliavel"):
            status_res = msg.get("status_resolucao")
            if status_res == "resolvido":
                st.success("✅ Problema marcado como resolvido pelo autoatendimento!")
            elif status_res == "chamado_aberto":
                st.info(f"🎫 Escalonado para o Suporte Nível 2 sob o protocolo **{msg.get('protocolo_gerado')}**.")
            else:
                st.caption("**Isso resolveu o seu problema?**")
                col_ok, col_ticket, _ = st.columns([1.2, 1.8, 3])
                if col_ok.button("✅ Sim, resolveu!", key=f"btn_ok_{idx}", use_container_width=True):
                    st.session_state.mensagens[idx]["status_resolucao"] = "resolvido"
                    st.rerun()
                if col_ticket.button("🎫 Não, abrir chamado de TI", key=f"btn_ticket_{idx}", type="primary", use_container_width=True):
                    modal_abrir_chamado(
                        indice_msg=idx,
                        problema_relatado=msg.get("pergunta_origem", "Problema relatado no chat"),
                        orientacao_ia=msg["content"]
                    )

# Captura pergunta pelo input do chat ou pelo botão de atalho
pergunta_chat = st.chat_input("Digite sua dúvida ou cole a mensagem de erro aqui...")
pergunta_usuario = pergunta_atalho or pergunta_chat

if pergunta_usuario:
    st.session_state.mensagens.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(pergunta_usuario)

    if not api_key or not modelo_selecionado:
        with st.chat_message("assistant", avatar="🛡️"):
            st.warning("⚠️ Insira uma **Chave de API válida** no menu lateral esquerdo para liberar o assistente.")
        st.stop()

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner(f"Consultando documentação técnica com {modelo_selecionado}..."):
            try:
                fontes_recuperadas = buscar_contexto_local(
                    pergunta_usuario, todos_os_chunks, top_k=top_k_slider
                )
                contexto_formatado = "\n\n---\n\n".join([f["trecho"] for f in fontes_recuperadas])

                prompt_sistema = (
                    "Você é um Analista de Suporte de TI Nível 1 prestativo, claro e objetivo. "
                    "Responda à dúvida do colaborador baseando-se EXCLUSIVAMENTE nos manuais técnicos "
                    "fornecidos no CONTEXTO abaixo. Use listas numeradas ou marcadores em negrito para "
                    "facilitar a leitura do passo a passo. Ao final, diga que, caso o procedimento não "
                    "resolva, ele pode clicar no botão 'Não, abrir chamado de TI' logo abaixo da mensagem.\n\n"
                    f"CONTEXTO DOS MANUAIS DE TI:\n{contexto_formatado}"
                )

                texto_resposta = gerar_resposta_ia(
                    provedor_detectado, api_key, modelo_selecionado, prompt_sistema, pergunta_usuario
                )

                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": texto_resposta,
                    "fontes": fontes_recuperadas,
                    "avaliavel": True,
                    "pergunta_origem": pergunta_usuario,
                    "status_resolucao": None
                })
                st.rerun()

            except Exception as e:
                st.error(f"Erro ao gerar resposta com `{modelo_selecionado}`: {e}")
