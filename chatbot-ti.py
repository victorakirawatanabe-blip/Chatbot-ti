import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="Helpdesk TI Auto-IA (RAG)", page_icon="⚡", layout="wide")

st.title("⚡ Chatbot de Suporte de TI com Detecção Automática de IA")
st.caption("Insira apenas sua chave de API: o sistema detecta o provedor e lista os modelos disponíveis automaticamente.")

# Base de conhecimento padrão de TI
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
    """
    Identifica o provedor a partir da API Key e consulta o endpoint oficial
    para retornar apenas os modelos de geração de texto disponíveis na conta.
    """
    chave = chave.strip()
    if not chave:
        return None, [], None

    # 1. Google Gemini (Chaves começam com AIza)
    if chave.startswith("AIza"):
        try:
            from google import genai
            client = genai.Client(api_key=chave)
            modelos = []
            for m in client.models.list():
                nome = m.name.replace("models/", "")
                # Filtra modelos de geração de texto (gemini) e ignora embeddings/áudio puro
                if "gemini" in nome.lower() and "embedding" not in nome.lower():
                    modelos.append(nome)
            modelos.sort(reverse=True)
            return "Google Gemini", modelos, None
        except Exception as e:
            return "Google Gemini", [], f"Chave Gemini inválida ou erro de conexão: {e}"

    # 2. Groq (Chaves começam com gsk_)
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

    # 3. Anthropic Claude (Chaves começam com sk-ant-)
    if chave.startswith("sk-ant-"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=chave)
            lista = client.models.list()
            modelos = [m.id for m in lista.data]
            return "Anthropic", modelos, None
        except Exception as e:
            return "Anthropic", [], f"Chave Anthropic inválida: {e}"

    # 4. OpenRouter (Chaves começam com sk-or-)
    if chave.startswith("sk-or-"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=chave, base_url="https://openrouter.ai/api/v1")
            lista = client.models.list()
            modelos = [m.id for m in lista.data]
            return "OpenRouter", modelos, None
        except Exception as e:
            return "OpenRouter", [], f"Chave OpenRouter inválida: {e}"

    # 5. Chaves genéricas "sk-..." (Testa OpenAI primeiro; se falhar, testa DeepSeek)
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
        modelos = [m.id for m in lista.data]
        return "DeepSeek", modelos, None
    except Exception:
        pass

    return None, [], "Não foi possível validar a chave em nenhum provedor (OpenAI, Gemini, Groq, Anthropic, OpenRouter ou DeepSeek)."


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
            config=types.GenerateContentConfig(
                system_instruction=prompt_sistema,
                temperature=0.2
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
    chunks = [f"[{d['titulo']}]\n{d['conteudo']}" for d in docs_padrao]
    if uploads:
        for arq in uploads:
            texto = arq.read().decode("utf-8", errors="ignore")
            partes = [p.strip() for p in texto.split("\n\n") if p.strip()]
            for i, parte in enumerate(partes):
                chunks.append(f"[Arquivo: {arq.name} - Parte {i+1}]\n{parte}")
    return chunks

def buscar_contexto_local(pergunta, lista_textos, top_k=2):
    vetorizador = TfidfVectorizer(ngram_range=(1, 2))
    matriz_docs = vetorizador.fit_transform(lista_textos)
    vetor_pergunta = vetorizador.transform([pergunta])

    similaridades = cosine_similarity(vetor_pergunta, matriz_docs).flatten()
    melhores_indices = np.argsort(similaridades)[::-1][:top_k]

    return [
        {"trecho": lista_textos[idx], "score": float(similaridades[idx])}
        for idx in melhores_indices
    ]


# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🔑 Chave de API")
    api_key = st.text_input(
        "Cole sua API Key de qualquer IA:",
        type="password",
        placeholder="AIza..., sk-..., gsk_..., sk-ant-..."
    )

    provedor_detectado = None
    modelo_selecionado = None

    if api_key:
        with st.spinner("Identificando provedor e buscando modelos..."):
            provedor_detectado, modelos_disponiveis, erro = detectar_e_listar_modelos(api_key)

        if erro:
            st.error(erro)
        elif provedor_detectado and modelos_disponiveis:
            st.success(f"Provedor detectado: **{provedor_detectado}**")
            modelo_selecionado = st.selectbox(
                "Selecione a IA disponível:",
                options=modelos_disponiveis
            )
        else:
            st.warning("Provedor reconhecido, mas nenhum modelo de chat foi retornado para esta chave.")
    else:
        st.info("Insira sua chave acima para carregar automaticamente a lista de IAs disponíveis.")

    st.divider()
    st.header("📚 Base de Conhecimento (RAG)")
    arquivos_enviados = st.file_uploader(
        "Adicionar manuais extras (.txt ou .md)",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    with st.expander("Ver manuais padrão integrados"):
        for doc in DOCUMENTOS_PADRAO:
            st.markdown(f"**• {doc['titulo']}**")
            st.caption(doc["conteudo"])

    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()


# --- INTERFACE DE CHAT ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "Olá! Cole sua chave de API na barra lateral para carregar os modelos disponíveis e tire suas dúvidas de Suporte de TI."
        }
    ]

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fontes" in msg:
            with st.expander("🔍 Ver documentos consultados pelo RAG"):
                for f in msg["fontes"]:
                    st.markdown(f"**Relevância: `{f['score']:.2f}`**\n```text\n{f['trecho']}\n```")

pergunta_usuario = st.chat_input("Descreva seu problema técnico (ex: erro na VPN, reset de senha, impressora)...")

if pergunta_usuario:
    st.session_state.mensagens.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    if not api_key or not modelo_selecionado:
        with st.chat_message("assistant"):
            st.warning("⚠️ Insira uma **Chave de API válida** na barra lateral para habilitar a lista de IAs antes de perguntar.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner(f"Consultando manuais e respondendo com {modelo_selecionado} ({provedor_detectado})..."):
            try:
                todos_os_chunks = preparar_documentos(DOCUMENTOS_PADRAO, arquivos_enviados)
                fontes_recuperadas = buscar_contexto_local(pergunta_usuario, todos_os_chunks, top_k=2)
                contexto_formatado = "\n\n---\n\n".join([f["trecho"] for f in fontes_recuperadas])

                prompt_sistema = (
                    "Você é um Analista de Suporte de TI Nível 1 prestativo, claro e objetivo. "
                    "Responda à dúvida do colaborador baseando-se EXCLUSIVAMENTE nos manuais técnicos "
                    "fornecidos no CONTEXTO abaixo. Se a resposta não estiver no contexto, informe educadamente "
                    "que o procedimento não consta na base atual e oriente o usuário a abrir um chamado no GLPI "
                    "ou ligar no ramal 4040.\n\n"
                    f"CONTEXTO DOS MANUAIS DE TI:\n{contexto_formatado}"
                )

                texto_resposta = gerar_resposta_ia(
                    provedor_detectado, api_key, modelo_selecionado, prompt_sistema, pergunta_usuario
                )

                st.markdown(texto_resposta)

                with st.expander("🔍 Ver documentos consultados pelo RAG"):
                    for f in fontes_recuperadas:
                        st.markdown(f"**Relevância: `{f['score']:.2f}`**\n```text\n{f['trecho']}\n```")

                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": texto_resposta,
                    "fontes": fontes_recuperadas
                })

            except Exception as e:
                st.error(f"Erro ao gerar resposta com o modelo `{modelo_selecionado}`: {e}")
