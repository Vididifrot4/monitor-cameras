"""
Monitor das câmeras "Catarina Aeroporto Executivo" (Sudeste/Sudoeste) do Clima ao Vivo.

Fluxo por ciclo:
  1. Abre a página com Playwright (Chrome real, por causa dos codecs H.264).
  2. Para cada câmera, acha o player pelo título e tira 2 capturas separadas por FRAME_GAP s.
  3. Classifica localmente:
       OFFLINE   -> texto de erro no card, ou imagem preta/azul/uniforme, ou player não encontrado
       CONGELADA -> as duas capturas são praticamente idênticas (vídeo parado)
       ONLINE    -> caso contrário
  4. (Opcional) Se o resultado local não for ONLINE, pede confirmação a um modelo de visão.
  5. Envia alerta no Telegram só quando o status MUDA (inclui aviso de recuperação).

Uso:
  python monitor_cameras.py            # loop contínuo (intervalo = CHECK_INTERVAL)
  python monitor_cameras.py --once     # um ciclo só (para cron / Agendador de Tarefas)
  python monitor_cameras.py --once --debug   # salva capturas e mostra métricas
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ----------------------------------------------------------------- configuração
URL = os.getenv("CAV_URL", "https://www.climaaovivo.com.br/cidades/brasil/sp/sao-roque/")

# nome exibido -> palavras que precisam aparecer no título do card
CAMERAS = {
    "Catarina Aeroporto Executivo - Sudeste": ["Catarina", "Sudeste"],
    "Catarina Aeroporto Executivo - Sudoeste": ["Catarina", "Sudoeste"],
}

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))   # 5 min entre ciclos
FRAME_GAP = int(os.getenv("FRAME_GAP", "20"))              # s entre as 2 capturas
LOAD_WAIT = int(os.getenv("LOAD_WAIT", "15"))              # s para o player carregar
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "chrome")   # "chrome" ou "" (chromium)

# limiares (calibre com --debug)
DARK_MEAN = float(os.getenv("DARK_MEAN", "18"))            # brilho médio 0-255
UNIFORM_STD = float(os.getenv("UNIFORM_STD", "6"))         # desvio-padrão = tela lisa
FROZEN_DIFF = float(os.getenv("FROZEN_DIFF", "0.4"))       # diferença média absoluta
FROZEN_CONFIRM = int(os.getenv("FROZEN_CONFIRM", "2"))     # ciclos seguidos p/ alertar

ERROR_PATTERNS = re.compile(
    r"stream offline|offline|carregando|erro ao carregar|sem conex[aã]o|"
    r"indispon[ií]vel|video unavailable|this video is unavailable|"
    r"an error occurred|ocorreu um erro|tente novamente|try again",
    re.IGNORECASE,
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE", "")         # WhatsApp, ex: 5511999998888
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")                 # opcional (Discord etc.)

USE_LLM = os.getenv("USE_LLM", "0") == "1"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")

DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR", "8"))  # hora do resumo diário (-1 desliga)

STATE_FILE = Path(os.getenv("STATE_FILE", "estado_cameras.json"))
DEBUG_DIR = Path("capturas")

LLM_PROMPT = """Você é um sistema de monitoramento de câmeras de transmissão meteorológica ao vivo.
Recebeu DUAS capturas da câmera "{nome}" do site Clima ao Vivo, tiradas com {gap} segundos de intervalo.
Determine se a transmissão está ativa, travada ou fora do ar.

Critérios de falha:
1. TELA PRETA OU AZUL: imagem escura, sem sinal ou erro de player.
2. MENSAGEM DE ERRO: textos como "Stream offline", "Carregando", "Erro ao carregar vídeo", "Sem conexão", ou botão de play estático.
3. CONGELAMENTO: nada mudou entre as duas capturas (nuvens, pista, árvores, relógio na sobreposição).
4. IMAGEM CORROMPIDA: artefatos severos que impedem a visualização.
Observação: à noite a cena pode ser quase estática; considere o relógio da sobreposição e ruído do sensor.

Responda SOMENTE com JSON:
{{"status": "ONLINE" | "OFFLINE" | "CONGELADA", "confianca": "alta" | "media" | "baixa", "motivo": "texto curto ou 'OK'"}}"""


# ----------------------------------------------------------------- captura
FIND_PLAYER_JS = """
(words) => {
  const norm = s => (s || '').toLowerCase();
  const all = Array.from(document.querySelectorAll('body *'));
  // menor elemento cujo texto contém todas as palavras
  let best = null;
  for (const el of all) {
    const t = norm(el.innerText);
    if (!t || !words.every(w => t.includes(w.toLowerCase()))) continue;
    if (!best || t.length < norm(best.innerText).length) best = el;
  }
  if (!best) return null;
  // sobe até achar um ancestral que tenha um player
  let node = best;
  for (let i = 0; i < 12 && node; i++) {
    const media = node.querySelector('video, iframe, canvas, img[src*="snap"], img[src*="cam"]');
    if (media) {
      const id = 'cav-mon-' + Math.random().toString(36).slice(2);
      media.setAttribute('data-cav-mon', id);
      node.setAttribute('data-cav-card', id);
      return id;
    }
    node = node.parentElement;
  }
  return null;
}
"""


def try_start_player(page, media):
    """Clica no player caso precise de interação para iniciar (play overlay)."""
    try:
        media.scroll_into_view_if_needed(timeout=5000)
        tag = media.evaluate("e => e.tagName.toLowerCase()")
        if tag == "video":
            media.evaluate("v => { v.muted = true; v.play && v.play().catch(()=>{}); }")
        else:
            box = media.bounding_box()
            if box:
                page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    except Exception:
        pass


def card_text(page, card_sel, media):
    """Texto visível do card e, se for iframe, de dentro do iframe."""
    texts = []
    try:
        texts.append(page.locator(card_sel).inner_text(timeout=3000))
    except Exception:
        pass
    try:
        if media.evaluate("e => e.tagName.toLowerCase()") == "iframe":
            frame = media.content_frame()
            if frame:
                texts.append(frame.locator("body").inner_text(timeout=3000))
    except Exception:
        pass
    return "\n".join(texts)


def locate(page, words):
    """Acha (de novo) o player; o site pode trocar o elemento depois do play."""
    pid = page.evaluate(FIND_PLAYER_JS, words)
    if not pid:
        raise RuntimeError("player não encontrado na página (título mudou ou câmera removida)")
    return page.locator(f'[data-cav-mon="{pid}"]'), f'[data-cav-card="{pid}"]'


def shot(page, words):
    """Captura o player; se o elemento sumiu, procura de novo e tenta outra vez."""
    last = None
    for _ in range(3):
        media, card_sel = locate(page, words)
        try:
            media.scroll_into_view_if_needed(timeout=5000)
            return Image.open(io.BytesIO(media.screenshot(timeout=10000))), media, card_sel
        except Exception as e:
            last = e
            time.sleep(3)
    # último recurso: captura o card inteiro
    media, card_sel = locate(page, words)
    try:
        return Image.open(io.BytesIO(page.locator(card_sel).screenshot(timeout=10000))), media, card_sel
    except Exception:
        raise RuntimeError(f"não foi possível capturar o player: {last}")


def capture(page, words):
    """Retorna (frame1, frame2, texto) ou levanta RuntimeError."""
    media, _ = locate(page, words)
    try_start_player(page, media)
    time.sleep(LOAD_WAIT)

    f1, _, _ = shot(page, words)
    time.sleep(FRAME_GAP)
    f2, media, card_sel = shot(page, words)
    return f1, f2, card_text(page, card_sel, media)


# ----------------------------------------------------------------- análise local
def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def is_blue_screen(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
    return b > 120 and b > r * 1.8 and b > g * 1.5 and a.std() < 25


def analyse_local(f1, f2, text):
    g1, g2 = gray(f1), gray(f2)
    mean, std = float(g2.mean()), float(g2.std())
    diff = float(np.abs(g1 - g2).mean()) if g1.shape == g2.shape else 999.0
    m = {"brilho": round(mean, 1), "desvio": round(std, 1), "diferenca": round(diff, 3)}

    hit = ERROR_PATTERNS.search(text or "")
    if hit:
        return "OFFLINE", f'mensagem no player: "{hit.group(0)}"', m
    if mean < DARK_MEAN and std < UNIFORM_STD * 2:
        return "OFFLINE", "tela preta / sem sinal", m
    if is_blue_screen(f2):
        return "OFFLINE", "tela azul (sem sinal)", m
    if std < UNIFORM_STD:
        return "OFFLINE", "imagem uniforme (player vazio ou corrompido)", m
    if diff < FROZEN_DIFF:
        return "CONGELADA", f"sem mudança entre capturas com {FRAME_GAP}s de intervalo", m
    return "ONLINE", "OK", m


# ----------------------------------------------------------------- confirmação por IA
def to_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def analyse_llm(nome, f1, f2):
    if not (USE_LLM and ANTHROPIC_API_KEY):
        return None
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_b64(f1)}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_b64(f2)}},
        {"type": "text", "text": LLM_PROMPT.format(nome=nome, gap=FRAME_GAP)},
    ]
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": LLM_MODEL, "max_tokens": 300,
                  "messages": [{"role": "user", "content": content}]},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["content"][0]["text"]
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception as e:
        log(f"  [IA] falhou: {e}")
        return None


# ----------------------------------------------------------------- alertas e estado
def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def notify(text, image=None):
    """Envia o alerta. Retorna True se pelo menos um canal confirmou o envio."""
    ok = False
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        try:
            if image is not None:
                buf = io.BytesIO()
                image.convert("RGB").save(buf, format="JPEG", quality=85)
                r = requests.post(f"{base}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": text},
                                  files={"photo": ("cam.jpg", buf.getvalue())}, timeout=30)
            else:
                r = requests.post(f"{base}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                                  timeout=30)
            if r.ok:
                ok = True
                log("  [Telegram] alerta enviado")
            else:
                log(f"  [Telegram] erro {r.status_code}: {r.text[:300]}")
        except Exception as e:
            log(f"  [Telegram] falhou: {e}")
    if CALLMEBOT_PHONE and CALLMEBOT_APIKEY:
        try:
            # o CallMeBot só envia texto (sem foto)
            r = requests.get("https://api.callmebot.com/whatsapp.php",
                             params={"phone": CALLMEBOT_PHONE, "text": text, "apikey": CALLMEBOT_APIKEY},
                             timeout=60)
            body = r.text.lower()
            if r.ok and "error" not in body and "invalid" not in body:
                ok = True
                log("  [WhatsApp] alerta enviado")
            else:
                log(f"  [WhatsApp] erro {r.status_code}: {r.text[:300]}")
        except Exception as e:
            log(f"  [WhatsApp] falhou: {e}")
    if WEBHOOK_URL:
        try:
            ok = requests.post(WEBHOOK_URL, json={"content": text, "text": text}, timeout=30).ok or ok
        except Exception as e:
            log(f"  [Webhook] falhou: {e}")
    if not (TELEGRAM_TOKEN or CALLMEBOT_APIKEY or WEBHOOK_URL):
        log(f"  [ALERTA] {text}  (nenhum canal configurado)")
    return ok


def load_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


ICON = {"ONLINE": "✅", "OFFLINE": "🔴", "CONGELADA": "🧊"}


def update_and_alert(state, nome, result, frame):
    prev = state.get(nome, {})
    status = result["status"]

    # congelamento precisa se repetir FROZEN_CONFIRM ciclos antes de virar alerta
    streak = prev.get("streak_congelada", 0) + 1 if status == "CONGELADA" else 0
    effective = status
    if status == "CONGELADA" and streak < FROZEN_CONFIRM:
        effective = prev.get("status", "ONLINE")

    first_run = prev.get("status") is None
    changed = effective != prev.get("status")
    if (first_run and effective != "ONLINE") or (not first_run and changed):
        if effective == "ONLINE":
            notify(f"{ICON['ONLINE']} {nome} voltou ao ar.")
        else:
            notify(f"{ICON[effective]} {nome}: {effective}\n"
                   f"Motivo: {result['motivo']} (confiança {result['confianca']})\n{URL}", frame)

    state[nome] = {
        "status": effective, "motivo": result["motivo"], "streak_congelada": streak,
        "desde": prev.get("desde") if not changed else datetime.now().isoformat(timespec="seconds"),
        "ultima_verificacao": datetime.now().isoformat(timespec="seconds"),
    }


def daily_report(state):
    """Manda um resumo por dia, no primeiro ciclo a partir de DAILY_REPORT_HOUR."""
    if DAILY_REPORT_HOUR < 0:
        return
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    meta = state.setdefault("_monitor", {})
    if now.hour < DAILY_REPORT_HOUR:
        return
    if meta.get("ultimo_resumo") == today:
        log(f"Resumo diário: já enviado hoje.")
        return
    linhas = [f"📋 Resumo diário – {now:%d/%m %H:%M}"]
    for nome in CAMERAS:
        st = state.get(nome, {}).get("status", "SEM DADOS")
        linhas.append(f"{ICON.get(st, '❔')} {nome.split(' - ')[-1]}: {st}")
    linhas.append("Monitor rodando normalmente.")
    log("Resumo diário: enviando...")
    if notify("\n".join(linhas)):
        meta["ultimo_resumo"] = today   # só marca se chegou; senão tenta no próximo ciclo


# ----------------------------------------------------------------- ciclo
def run_cycle(debug=False):
    state = load_state()
    with sync_playwright() as p:
        kw = {"headless": True, "args": ["--autoplay-policy=no-user-gesture-required"]}
        try:
            browser = p.chromium.launch(channel=BROWSER_CHANNEL or None, **kw)
        except Exception:
            log("Chrome não encontrado; usando Chromium (vídeos H.264 podem não tocar).")
            browser = p.chromium.launch(**kw)
        ctx = browser.new_context(viewport={"width": 1366, "height": 900}, locale="pt-BR")
        page = ctx.new_page()

        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_load_state("networkidle", timeout=30000)
        except PWTimeout:
            pass  # players de vídeo mantêm a rede ocupada; segue mesmo assim

        for label in ("Aceitar", "Aceito", "Concordo", "OK", "Entendi"):
            try:
                page.get_by_role("button", name=label).first.click(timeout=1500)
                break
            except Exception:
                continue

        for nome, words in CAMERAS.items():
            frame = None
            try:
                f1, f2, text = capture(page, words)
                frame = f2
                status, motivo, m = analyse_local(f1, f2, text)
                result = {"status": status, "confianca": "media", "motivo": motivo}
                if status != "ONLINE":
                    llm = analyse_llm(nome, f1, f2)
                    if llm and llm.get("status") in ICON:
                        result = llm
                log(f"{nome}: {result['status']} | {result['motivo']} | {m}")
                if debug:
                    DEBUG_DIR.mkdir(exist_ok=True)
                    tag = f"{datetime.now():%Y%m%d_%H%M%S}_{words[-1].lower()}"
                    f1.save(DEBUG_DIR / f"{tag}_a.png")
                    f2.save(DEBUG_DIR / f"{tag}_b.png")
            except Exception as e:
                result = {"status": "OFFLINE", "confianca": "baixa", "motivo": str(e)[:200]}
                log(f"{nome}: OFFLINE | {e}")
                try:
                    frame = Image.open(io.BytesIO(page.screenshot()))
                except Exception:
                    pass
            update_and_alert(state, nome, result, frame)

        browser.close()
    daily_report(state)
    save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="executa um ciclo e sai")
    ap.add_argument("--debug", action="store_true", help="salva capturas em ./capturas")
    ap.add_argument("--teste", "--teste-telegram", dest="teste", action="store_true",
                    help="envia uma mensagem de teste para os canais configurados e sai")
    args = ap.parse_args()

    if args.teste:
        if not ((TELEGRAM_TOKEN and TELEGRAM_CHAT_ID) or (CALLMEBOT_PHONE and CALLMEBOT_APIKEY) or WEBHOOK_URL):
            log("Nenhum canal configurado (Telegram, WhatsApp ou webhook).")
            return
        log("Enviando mensagem de teste...")
        notify("✅ Teste do monitor de câmeras: os alertas estão funcionando.")
        return
    if args.once:
        run_cycle(args.debug)
        return
    log(f"Monitorando {len(CAMERAS)} câmeras a cada {CHECK_INTERVAL}s. Ctrl+C para parar.")
    while True:
        start = time.time()
        try:
            run_cycle(args.debug)
        except Exception as e:
            log(f"Erro no ciclo: {e}")
        time.sleep(max(10, CHECK_INTERVAL - (time.time() - start)))


if __name__ == "__main__":
    sys.exit(main())
