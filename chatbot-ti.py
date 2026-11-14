import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="Helpdesk TI Universal (RAG)", page_icon="🌐", layout="wide")

st.title("🌐 Chatbot de Suporte de TI Multi-IA (RAG)")
st.caption("Conecte sua chave da OpenAI, Gemini, Groq, Anthropic ou qualquer API compatível.")

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

# Configurações padrão de modelos por provedor
PROVEDORES = {
    "OpenAI (ChatGPT)": {"modelo_padrao": "gpt-4o-mini", "placeholder_key": "sk-..."},
    "Google Gemini": {"modelo_padrao": "gemini-2.5-flash", "placeholder_key": "AIzaSy..."},
    "Groq (Llama / Mixtral)": {"modelo_padrao": "llama-3.3-70b-versatile", "placeholder_key": "gsk_..."},
    "Anthropic (Claude)": {"modelo_padrao": "claude-3-5-sonnet-latest", "placeholder_key": "sk-ant-..."},
    "DeepSeek / OpenRouter / Custom": {"modelo_padrao": "deepseek-chat", "placeholder_key": "sk-..."}
}

# --- BARRA LATERAL: Seleção de Provedor, API Key e Upload ---
with st.sidebar:
    st.header("🤖 Provedor de IA")
    provedor = st.selectbox("Escolha a plataforma:", list(PROVEDORES.keys()))
    
    api_key = st.text_input(
        f"Chave de API ({provedor}):",
        type="password",
        placeholder=PROVEDORES[provedor]["placeholder_key"]
    )
    
    modelo = st.text_input(
        "Nome do Modelo:",
        value=PROVEDORES[provedor]["modelo_padrao"],
        help="Você pode digitar qualquer modelo válido do provedor selecionado."
    )
    
    base_url_custom = ""
    if provedor == "DeepSeek / OpenRouter / Custom":
        base_url_custom = st.text_input(
            "Base URL da API:",
            value="https://api.deepseek.com/v1",
            help="Ex: https://openrouter.ai/api/v1 ou https://api.deepseek.com/v1"
        )

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

# --- MOTOR RAG LOCAL (Independente de provedor de IA) ---
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
    """Realiza busca vetorial TF-IDF + Similaridade de Cosseno localmente."""
    vetorizador = TfidfVectorizer(ngram_range=(1, 2))
    matriz_docs = vetorizador.fit_transform(lista_textos)
    vetor_pergunta = vetorizador.transform([pergunta])
    
    similaridades = cosine_similarity(vetor_pergunta, matriz_docs).flatten()
    melhores_indices = np.argsort(similaridades)[::-1][:top_k]
    
    resultados = []
    for idx in melhores_indices:
        resultados.append({
            "trecho": lista_textos[idx],
            "score": float(similaridades[idx])
        })
    return resultados

# --- FUNÇÃO UNIVERSAL DE CHAMADA DE IA ---
def gerar_resposta_ia(provedor_escolhido, chave, nome_modelo, prompt_sistema, pergunta, url_custom=""):
    """Roteia a chamada para o SDK correto de acordo com o provedor escolhido."""
    
    if provedor_escolhido == "Google Gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=chave)
        resp = client.models.generate_content(
            model=nome_modelo,
            contents=pergunta,
            config=types.GenerateContentConfig(
                system_instruction=prompt_sistema,
                temperature=0.2
            )
        )
        return resp.text

    elif provedor_escolhido == "Anthropic (Claude)":
        import anthropic
        client = anthropic.Anthropic(api_key=chave)
        resp = client.messages.create(
            model=nome_modelo,
            max_tokens=1024,
            system=prompt_sistema,
            temperature=0.2,
            messages=[{"role": "user", "content": pergunta}]
        )
        return resp.content[0].text

    else:
        # OpenAI, Groq, DeepSeek, OpenRouter usam o padrão OpenAI SDK
        from openai import OpenAI
        
        if provedor_escolhido == "Groq (Llama / Mixtral)":
            client = OpenAI(api_key=chave, base_url="https://api.groq.com/openai/v1")
        elif provedor_escolhido == "DeepSeek / OpenRouter / Custom":
            client = OpenAI(api_key=chave, base_url=url_custom)
        else:
            client = OpenAI(api_key=chave)
            
        resp = client.chat.completions.create(
            model=nome_modelo,
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pergunta}
            ]
        )
        return resp.choices[0].message.content

# --- INTERFACE DE CHAT ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "Olá! Escolha seu provedor de IA na barra lateral, insira sua chave de API e tire suas dúvidas técnicas de TI."
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

    if not api_key:
        with st.chat_message("assistant"):
            st.warning(f"⚠️ Por favor, insira sua chave de API para **{provedor}** na barra lateral esquerda.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner(f"Consultando base de TI e gerando resposta via {provedor}..."):
            try:
                # 1. Busca RAG local independente de API
                todos_os_chunks = preparar_documentos(DOCUMENTOS_PADRAO, arquivos_enviados)
                fontes_recuperadas = buscar_contexto_local(pergunta_usuario, todos_os_chunks, top_k=2)
                contexto_formatado = "\n\n---\n\n".join([f["trecho"] for f in fontes_recuperadas])

                # 2. Monta o prompt de sistema
                prompt_sistema = (
                    "Você é um Analista de Suporte de TI Nível 1 prestativo, claro e objetivo. "
                    "Responda à dúvida do colaborador baseando-se EXCLUSIVAMENTE nos manuais técnicos "
                    "fornecidos no CONTEXTO abaixo. Se a resposta não estiver no contexto, informe educadamente "
                    "que o procedimento não consta na base atual e oriente o usuário a abrir um chamado no GLPI "
                    "ou ligar no ramal 4040.\n\n"
                    f"CONTEXTO DOS MANUAIS DE TI:\n{contexto_formatado}"
                )

                # 3. Gera a resposta no provedor escolhido
                texto_resposta = gerar_resposta_ia(
                    provedor, api_key, modelo, prompt_sistema, pergunta_usuario, base_url_custom
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
                st.error(f"Erro ao conectar na API ({provedor}). Verifique sua chave e o nome do modelo. Detalhes: {e}")
