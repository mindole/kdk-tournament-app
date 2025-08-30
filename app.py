# app.py
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd
import random
import sqlite3, json, secrets, time
from pathlib import Path
from contextlib import contextmanager

# =========================
# App & Global State
# =========================
st.set_page_config(page_title="KDK 대진표", page_icon="🎾", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "setup"
if "tour" not in st.session_state:
    st.session_state.tour = None  # 초기 None → 아래에서 TourState 생성
if "current_tid" not in st.session_state:
    st.session_state.current_tid = None
if "view_only" not in st.session_state:
    st.session_state.view_only = False

# =========================
# Models
# =========================
@dataclass
class Match:
    team_a: Tuple[int, int]
    team_b: Tuple[int, int]
    score_a: Optional[int] = None
    score_b: Optional[int] = None

@dataclass
class Round:
    matches: List[Match] = field(default_factory=list)

@dataclass
class PlayerStat:
    name: str
    gp: int = 0
    w: int = 0
    l: int = 0
    gf: int = 0
    ga: int = 0
    @property
    def gd(self): return self.gf - self.ga
    @property
    def pts(self): return self.w

@dataclass
class TourState:
    title: str = ""
    num_players: int = 5
    players: List[str] = field(default_factory=list)
    rounds: List[Round] = field(default_factory=list)
    current_round: int = 0
    stats: Dict[int, PlayerStat] = field(default_factory=dict)
    tie_rule: str = "GD_FIRST"      # "GD_FIRST" or "GF_FIRST"
    assignment_mode: str = "fixed"  # "fixed" or "random"
    mapping: Dict[int, int] = field(default_factory=dict)  # number->player idx

if st.session_state.tour is None:
    st.session_state.tour = TourState()
tour: TourState = st.session_state.tour

# =========================
# Constants / DB
# =========================
DB_PATH = Path("kdk_share.db")
EDIT_SECRET = "SECRET"  # 요구사항: 고정

def ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tournaments (
      tid TEXT PRIMARY KEY,
      edit_key TEXT NOT NULL,
      title TEXT,
      num_players INTEGER,
      players TEXT,
      rounds  TEXT,
      stats   TEXT,
      current_round INTEGER,
      tie_rule TEXT,
      mapping TEXT,
      updated_at REAL
    )
    """)
    con.commit()
    return con

def _state_to_payload(t: TourState) -> dict:
    rounds = []
    for rd in t.rounds:
        matches = []
        for m in rd.matches:
            matches.append({
                "team_a": list(m.team_a),
                "team_b": list(m.team_b),
                "score_a": m.score_a,
                "score_b": m.score_b,
            })
        rounds.append({"matches": matches})
    stats = {i: {"name": s.name, "gp": s.gp, "w": s.w, "l": s.l, "gf": s.gf, "ga": s.ga}
             for i, s in t.stats.items()}
    return {
        "title": t.title,
        "num_players": t.num_players,
        "players": t.players,
        "rounds": rounds,
        "stats": stats,
        "current_round": t.current_round,
        "tie_rule": t.tie_rule,
        "mapping": t.mapping,
    }

def _apply_payload_to_state(payload: dict, t: TourState, recompute_cb=None):
    t.title = payload.get("title", "")
    t.num_players = int(payload.get("num_players", len(payload.get("players", [])) or 5))
    t.players = list(payload.get("players", []))
    t.tie_rule = payload.get("tie_rule", "GD_FIRST")
    t.current_round = int(payload.get("current_round", 0))
    t.mapping = {int(k): v for k, v in payload.get("mapping", {}).items()}
    # rounds
    t.rounds.clear()
    for rd in payload.get("rounds", []):
        r = Round()
        for m in rd.get("matches", []):
            r.matches.append(
                Match(tuple(m["team_a"]), tuple(m["team_b"]), m.get("score_a"), m.get("score_b"))
            )
        t.rounds.append(r)
    # stats
    if payload.get("stats"):
        t.stats = {int(i): PlayerStat(**s) for i, s in payload["stats"].items()}
    if recompute_cb:
        recompute_cb(t)

def _tid_exists(tid: str) -> bool:
    con = ensure_db()
    row = con.execute("SELECT 1 FROM tournaments WHERE tid=?", (tid,)).fetchone()
    con.close()
    return bool(row)

def _gen_tid4() -> str:
    for _ in range(50):
        t = f"{secrets.randbelow(10000):04d}"
        if not _tid_exists(t):
            return t
    return f"{int(time.time())%10000:04d}"

def save_tournament(t: TourState, tid=None):
    con = ensure_db()
    cur = con.cursor()
    if not tid:
        tid = _gen_tid4()
    payload = _state_to_payload(t)
    cur.execute("""
    INSERT INTO tournaments(tid, edit_key, title, num_players, players, rounds, stats,
                            current_round, tie_rule, mapping, updated_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(tid) DO UPDATE SET
      title=excluded.title,
      num_players=excluded.num_players,
      players=excluded.players,
      rounds=excluded.rounds,
      stats=excluded.stats,
      current_round=excluded.current_round,
      tie_rule=excluded.tie_rule,
      mapping=excluded.mapping,
      updated_at=excluded.updated_at
    """, (
        tid, EDIT_SECRET, payload["title"], payload["num_players"],
        json.dumps(payload["players"], ensure_ascii=False),
        json.dumps(payload["rounds"],  ensure_ascii=False),
        json.dumps(payload["stats"],   ensure_ascii=False),
        payload["current_round"], payload["tie_rule"],
        json.dumps(payload["mapping"], ensure_ascii=False),
        time.time()
    ))
    con.commit()
    con.close()
    return tid, EDIT_SECRET

def load_tournament(tid: str):
    con = ensure_db()
    row = con.execute(
        "SELECT tid, edit_key, title, num_players, players, rounds, stats, current_round, tie_rule, mapping FROM tournaments WHERE tid=?",
        (tid,)
    ).fetchone()
    con.close()
    if not row:
        return None
    _, edit_key, title, num_players, players, rounds, stats, current_round, tie_rule, mapping = row
    return {
        "tid": tid,
        "edit_key": edit_key,
        "title": title,
        "num_players": num_players,
        "players": json.loads(players or "[]"),
        "rounds": json.loads(rounds or "[]"),
        "stats": json.loads(stats or "{}"),
        "current_round": current_round,
        "tie_rule": tie_rule,
        "mapping": json.loads(mapping or "{}"),
    }

# =========================
# Scoring / Stats
# =========================
def init_stats(names: List[str]) -> Dict[int, PlayerStat]:
    return {i: PlayerStat(name=names[i]) for i in range(len(names))}

def apply_round_results(state: TourState, rnd_idx: int):
    r = state.rounds[rnd_idx]
    for m in r.matches:
        if m.score_a is None or m.score_b is None:
            continue
        a1, a2 = m.team_a; b1, b2 = m.team_b; sa, sb = m.score_a, m.score_b
        for p in (a1, a2):
            stp = state.stats[p]; stp.gp += 1; stp.gf += sa; stp.ga += sb
        for p in (b1, b2):
            stp = state.stats[p]; stp.gp += 1; stp.gf += sb; stp.ga += sa
        if sa > sb:
            state.stats[a1].w += 1; state.stats[a2].w += 1
            state.stats[b1].l += 1; state.stats[b2].l += 1
        elif sb > sa:
            state.stats[b1].w += 1; state.stats[b2].w += 1
            state.stats[a1].l += 1; state.stats[a2].l += 1

def recompute_all_stats(state: TourState):
    state.stats = init_stats(state.players)
    for ri in range(len(state.rounds)):
        apply_round_results(state, ri)

def sort_key(state: TourState, s: PlayerStat):
    return (s.pts, s.gf, s.gd) if state.tie_rule == "GF_FIRST" else (s.pts, s.gd, s.gf)

# =========================
# KDK 4-Game Templates (5~10명, N게임, 개인당 4게임)
# =========================
KDK_V2010_4G_TEMPLATES: Dict[int, List[str]] = {
    5:  ["12:34", "13:25", "14:35", "15:24", "23:45"],
    6:  ["12:34", "15:46", "23:56", "14:25", "24:36", "16:35"],
    7:  ["12:34", "56:17", "35:24", "14:36", "23:57", "16:25", "46:37"],
    8:  ["12:34", "56:78", "13:57", "24:68", "37:48", "15:26", "16:38", "25:47"],
    9:  ["12:34", "56:78", "19:57", "23:68", "49:38", "15:26", "17:89", "36:45", "24:79"],
    10: ["12:34", "56:78", "23:6A", "19:58", "3A:45", "27:89", "4A:68", "13:79", "46:59", "17:2A"],
}

def _parse_pair(token: str, mapping: Dict[int, int]) -> Tuple[int, int]:
    def _to_num(ch: str) -> int:
        return 10 if ch.upper() == "A" else int(ch)
    p1 = mapping[_to_num(token[0])]
    p2 = mapping[_to_num(token[1])]
    return (p1, p2)

def build_schedule_4g(num_players: int, mapping: Dict[int, int]) -> List[Round]:
    if num_players not in KDK_V2010_4G_TEMPLATES:
        raise ValueError("KDK 4게임 템플릿은 5~10명만 지원합니다.")
    rounds: List[Round] = []
    for ent in KDK_V2010_4G_TEMPLATES[num_players]:
        left, right = ent.split(":")
        a1, a2 = _parse_pair(left, mapping)
        b1, b2 = _parse_pair(right, mapping)
        rounds.append(Round(matches=[Match((a1, a2), (b1, b2))]))
    return rounds

# =========================
# UI Helpers
# =========================
def _qp_get(name: str):
    v = st.query_params.get(name)
    return v[0] if isinstance(v, list) and v else v

@contextmanager
def section(title: str, subtitle: Optional[str] = None):
    st.markdown(
        f"<div class='section'><h3>{title}</h3>"
        + (f"<div class='mode-strip'>{subtitle}</div>" if subtitle else ""),
        unsafe_allow_html=True
    )
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

def styled_table_html(df: pd.DataFrame, font_px: int = 15) -> str:
    styler = df.style
    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass
    styler = (
        styler.set_table_styles([
            {"selector": "table",
             "props": [("width","100%"), ("table-layout","fixed"),
                       ("border-collapse","separate"), ("border-spacing","0"),
                       ("background","#ffffff"), ("border","1px solid #e2e8f0"),
                       ("border-radius","12px"), ("overflow","hidden")]},
            {"selector": "thead th",
             "props": [("background","#dcfce7"), ("color","#065f46"),
                       ("font-weight","700"), ("text-align","center"),
                       ("padding","10px 12px"), ("border-bottom","1px solid #22c55e"),
                       ("font-size", f"{font_px}px")]},
            {"selector": "tbody td",
             "props": [("padding","10px 12px"), ("font-size", f"{font_px}px"),
                       ("color","#0f172a"), ("border-bottom","1px solid #eef2f7"),
                       ("text-align","center")]},
            {"selector": "tbody tr:nth-child(even)",
             "props": [("background-color","#f8fafc")]},
            {"selector": "tbody tr:hover",
             "props": [("background-color","#f0fdf4")]},
        ])
    )
    return styler.to_html()

# =========================
# CSS (요약본)
# =========================
CUSTOM_CSS = """
<style>
:root{
  --ink:#0f172a; --line:#e2e8f0; --accent:#22c55e;
}
html, body, .block-container{
  background: radial-gradient(1200px 800px at 15% 0%, #f0fff4, #fafffb 65%);
}
html, body, [data-testid="stAppViewContainer"] *{ color:var(--ink) !important; }

/* Expander/Header */
.stExpander, .st-expander, .st-expanderHeader, div[role="button"][aria-expanded]{
  background:#ffffff !important; color:#0f172a !important;
  border:2px solid var(--line) !important; border-radius:12px !important; font-weight:700;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"], div[role="combobox"]{
  background:#ffffff !important; color:#0f172a !important;
  border:2px solid var(--line) !important; border-radius:10px !important; font-weight:600;
}

/* Buttons */
.stButton>button{
  background:#2ea043 !important; color:#fff !important;
  border:2px solid #1c7c36 !important; border-radius:14px !important; font-weight:700;
}

/* Match card */
.match-card{
  border: 3px solid var(--accent); border-radius: 16px;
  background: linear-gradient(180deg,#f8fff9,#f3fff6);
  box-shadow: 0 0 0 4px rgba(34,197,94,.10), 0 10px 20px rgba(0,0,0,.06);
  padding: 18px 20px; margin: 16px 0;
}
.match-header{ text-align:center; font-weight:900; color:#022c22; font-size: clamp(18px,2.2vw,26px); margin-bottom: 12px; }
.score-row{ display:grid; grid-template-columns: 4fr 2fr 80px 2fr 4fr; align-items:center; gap:12px; }
.team-label{ text-align:center; font-weight:900; font-size: clamp(16px,1.8vw,22px); }
.vs-chip{ height:58px; width:58px; border-radius:999px; background:#dcfce7; border:2px solid #86efac;
          display:flex; align-items:center; justify-content:center; font-weight:900; font-size: clamp(16px,1.8vw,22px); color:#047857; margin:0 auto; }
@media (max-width: 880px){ .score-row{ grid-template-columns:1fr; gap:8px; } .vs-chip{ margin:6px auto; } }

/* TOP Cards */
.topcards{ display:grid; gap:14px; grid-template-columns: repeat(3, minmax(220px, 1fr)); margin:10px 0 14px 0; }
.topcard{ background:#fff; border:2px solid #e5e7eb; border-radius:18px; box-shadow:0 6px 18px rgba(0,0,0,.06); padding:18px 16px; }
.topcard h4{ margin:0 0 8px 0; font-size: clamp(16px,1.2vw,18px); color:#64748b; font-weight:800; }
.topcard .name{ margin:2px 0 6px 0; font-size: clamp(22px,2.6vw,34px); font-weight:900; line-height:1.15; }
.topcard .sub{ display:inline-flex; gap:8px; align-items:center; background:#f1f5f9; color:#0f172a; border-radius:999px; padding:6px 10px; font-weight:700; font-size: clamp(12px,.95vw,14px); }
.topcard.top1{ position:relative; border-color:#22c55e; background: radial-gradient(120% 120% at 20% -10%, #ecffe9, #fff);
               box-shadow: 0 0 0 4px rgba(34,197,94,.12), 0 12px 26px rgba(0,0,0,.10); }
.topcard.top1::after{ content:"🎉"; position:absolute; right:14px; top:10px; font-size:24px; }
.topcard.top1 .name{ color:#065f46; }
.topcard.top1 .sub{ background:#dcfce7; color:#065f46; border:1px solid #86efac; }
.topcard.top2{ border-color:#93c5fd; background:linear-gradient(180deg,#f5fbff,#fff); }
.topcard.top3{ border-color:#fca5a5; background:linear-gradient(180deg,#fff5f5,#fff); }

/* Table card + Styler HTML 래퍼 */
.table-card{ background:#fff; border:2px solid var(--accent); border-radius:14px; box-shadow:0 6px 18px rgba(0,0,0,.06); padding:14px 16px; margin:16px 0; }
.table-title{ font-weight:900; font-size: clamp(16px, 1.1vw, 18px); color:#065f46; margin-bottom:10px; display:flex; gap:8px; align-items:center; }

.table-wrap { width:100%; }
.table-wrap table { width:100% !important; table-layout: fixed; border-collapse: separate; border-spacing: 0; background:#fff;
                    border:1px solid #e2e8f0; border-radius:12px; overflow:hidden; }
.table-wrap thead th{ background:#dcfce7 !important; color:#065f46 !important; font-weight:800 !important; text-align:center !important; padding:10px 12px !important; border-bottom:1px solid #22c55e !important; font-size:15px; }
.table-wrap tbody td{ padding:10px 12px !important; text-align:center !important; color:#0f172a !important; font-weight:600 !important; font-size:15px; border-bottom:1px solid #eef2f7 !important; }
.table-wrap tbody tr:nth-child(even) td{ background:#f8fafc !important; }
.table-wrap tbody tr:hover td{ background:#f0fdf4 !important; }
/* 인덱스 강제 숨김 */
.table-wrap th.row_heading, .table-wrap th.blank, .table-wrap tbody th[scope="row"]{ display:none !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Router helpers
# =========================
def ensure_loaded_from_query(auto_switch: bool = True):
    """?tid & ?key로 접근 시 로드/모드 설정/페이지 전환."""
    tid = _qp_get("tid")
    key = _qp_get("key")
    if not tid:
        return
    # 이미 로드된 동일 tid + 라운드 존재 시 모드만 갱신
    if st.session_state.current_tid == tid and len(tour.rounds) > 0:
        st.session_state.view_only = not (key and key == EDIT_SECRET)
        return
    rec = load_tournament(tid)
    if not rec:
        st.warning("해당 참여코드의 대진을 찾을 수 없어요.")
        return
    _apply_payload_to_state(rec, tour, recompute_cb=recompute_all_stats)
    st.session_state.current_tid = tid
    st.session_state.view_only = not (key and key == EDIT_SECRET)
    if auto_switch:
        st.session_state.page = "tournament"
        st.rerun()

# 쿼리 먼저 반영
ensure_loaded_from_query(auto_switch=True)

# =========================
# Pages
# =========================
def render_setup():
    with st.sidebar:
        st.markdown("### 🙌 토너먼트 참여하기")
        join_code = st.text_input("참여 코드 (4자리 숫자)", max_chars=4, placeholder="예) 0427")
        join_secret = st.text_input("비밀키 (선택)", type="password", placeholder="")
        if st.button("참여"):
            code = (join_code or "").strip()
            if len(code) == 4 and code.isdigit():
                qp = {"tid": code}
                if (join_secret or "").strip() == EDIT_SECRET:
                    qp["key"] = EDIT_SECRET
                st.query_params.update(qp)
                st.rerun()
            else:
                st.warning("참여 코드는 4자리 숫자여야 합니다.")

    st.title("🎾 KDK 대진표 어플리케이션 🎾")

    if st.session_state.view_only:
        total = len(tour.rounds)
        if "view_only_round" not in st.session_state:
            st.session_state.view_only_round = int(max(0, min(tour.current_round, max(total - 1, 0))))
        tour.current_round = int(max(0, min(st.session_state.view_only_round, max(total - 1, 0))))

    with st.expander("1) 기본 정보 & 설정", expanded=True if not tour.players else False):
        tour.title = st.text_input("대진 이름", value=tour.title or "어프로치 정규 월례대회 A조")
        tour.num_players = st.selectbox("인원수", options=[5, 6, 7, 8, 9, 10], index=0)
        tie_sel = st.selectbox("순위 동률 규정", options=["승점>득실차>득게임", "승점>득게임>득실차"], index=0)
        tour.tie_rule = "GF_FIRST" if "득게임" in tie_sel else "GD_FIRST"

        assignment_mode = st.radio("배치 방식", options=["고정(입력순)", "랜덤(셔플)"], horizontal=True)
        tour.assignment_mode = "random" if "랜덤" in assignment_mode else "fixed"
        seed_val = st.number_input("Seed (같은 Seed=같은 배치)", min_value=0, value=0, step=1) if tour.assignment_mode == "random" else None

        # 선수 입력
        default_names = tour.players or [f"Player {i+1}" for i in range(tour.num_players)]
        cols = st.columns(4); entered = []
        for i in range(tour.num_players):
            with cols[i % 4]:
                entered.append(st.text_input(f"선수 {i+1}", value=default_names[i] if i < len(default_names) else f"Player {i+1}"))

        if st.button("✅ 대진 생성", type="primary", disabled=st.session_state.view_only):
            tour.players = [n.strip() or f"Player {i+1}" for i, n in enumerate(entered)]
            # 번호→인덱스
            idxs = list(range(len(tour.players)))
            if tour.assignment_mode == "random":
                if seed_val is not None:
                    random.seed(int(seed_val))
                random.shuffle(idxs)
            tour.mapping = {i + 1: idxs[i] for i in range(len(idxs))}
            # 스케줄 생성
            tour.rounds = build_schedule_4g(tour.num_players, tour.mapping)
            tour.current_round = 0
            tour.stats = init_stats(tour.players)

            tid, edit_key = save_tournament(tour)
            st.session_state.current_tid = tid
            st.session_state.view_only = False
            st.query_params.update({"tid": tid, "key": edit_key})
            st.session_state.page = "tournament"
            st.rerun()

def render_tournament():
    # 배너
    tid_show = st.session_state.current_tid
    mode_label = "👀 보기 모드" if st.session_state.view_only else "✍️ 수정 모드"
    mode_color = "#2563eb" if st.session_state.view_only else "#16a34a"
    if tid_show:
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0 6px 0;">
          <span style="background:{mode_color};color:white;padding:4px 12px;border-radius:999px;font-weight:700;">
            {mode_label}
          </span>
          <span style="color:#9ca3af;">
            참여코드: <b>{tid_show}</b>{"" if st.session_state.view_only else f" · 비밀키: <b>{EDIT_SECRET}</b>"}
          </span>
        </div>
        """, unsafe_allow_html=True)

    # 번호 매칭표
    with section("코드 공유시 동일한 번호 체계를 사용합니다."):
        mapping_df = pd.DataFrame({
            "KDK 번호": sorted(tour.mapping.keys()),
            "선수 이름": [tour.players[tour.mapping[i]] for i in sorted(tour.mapping.keys())]
        })
        st.markdown("<div class='table-card'><div class='table-title'>🧾 선수 번호 매칭표</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='table-wrap'>{styled_table_html(mapping_df, font_px=15)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 라운드 진행
    with section(f"🎾 라운드 진행 – Round {tour.current_round+1}/{len(tour.rounds)}"):
        total = len(tour.rounds)
        if total == 0:
            st.info("아직 생성된 대진이 없습니다. 상단에서 선수 입력 후 **대진 생성**을 눌러주세요.")
        else:
            def clamp_idx(idx: int, total: int) -> int:
                return int(max(0, min(idx, total - 1)))

            if st.session_state.view_only:
                st.session_state.view_only_round = clamp_idx(
                    st.session_state.get("view_only_round", tour.current_round), total
                )
                tour.current_round = st.session_state.view_only_round
            else:
                tour.current_round = clamp_idx(tour.current_round, total)

            finished = sum(
                1 for r_ in tour.rounds
                if all(m.score_a is not None and m.score_b is not None for m in r_.matches)
            )
            st.progress(finished / total if total else 0.0,
                        text=f"진행률: {finished}/{total} 라운드 완료")

            def _persist_round_if_editable():
                if (not st.session_state.view_only) and st.session_state.current_tid:
                    save_tournament(tour, tid=st.session_state.current_tid)

            def go_prev():
                if st.session_state.view_only:
                    cur = st.session_state.get("view_only_round", tour.current_round)
                    st.session_state.view_only_round = clamp_idx(cur - 1, total)
                else:
                    tour.current_round = clamp_idx(tour.current_round - 1, total)
                    _persist_round_if_editable()

            def go_next():
                if st.session_state.view_only:
                    cur = st.session_state.get("view_only_round", tour.current_round)
                    st.session_state.view_only_round = clamp_idx(cur + 1, total)
                else:
                    tour.current_round = clamp_idx(tour.current_round + 1, total)
                    _persist_round_if_editable()

            can_prev = tour.current_round > 0
            can_next = tour.current_round < total - 1

            c1, c2, c3 = st.columns([1, 2, 1])
            with c1: st.button("◀ 이전", disabled=not can_prev, use_container_width=True, on_click=go_prev)
            with c3: st.button("다음 ▶", disabled=not can_next, use_container_width=True, on_click=go_next)
            with c2: st.markdown(f"**현재 라운드: {tour.current_round + 1} / {total}**")

            # 안전 보정
            if st.session_state.view_only:
                tour.current_round = clamp_idx(st.session_state.get("view_only_round", tour.current_round), total)
            else:
                tour.current_round = clamp_idx(tour.current_round, total)

            r = tour.rounds[tour.current_round]
            for mi, m in enumerate(r.matches):
                a1, a2 = tour.players[m.team_a[0]], tour.players[m.team_a[1]]
                b1, b2 = tour.players[m.team_b[0]], tour.players[m.team_b[1]]
                sa = m.score_a or 0
                sb = m.score_b or 0
                global_no = tour.current_round * len(r.matches) + (mi + 1)

                st.markdown(
                    f"<div class='match-card'><div class='match-header'>{global_no} 라운드 : {a1}&{a2} vs {b1}&{b2}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("<div class='score-row'>", unsafe_allow_html=True)
                st.markdown(f"<div class='team-label'>{a1}&{a2}</div>", unsafe_allow_html=True)

                # 점수 A
                m.score_a = st.selectbox(
                    label="", options=list(range(7)), index=sa,
                    key=f"sa_{tour.current_round}_{mi}",
                    label_visibility="collapsed", disabled=st.session_state.view_only
                )
                st.markdown("<div class='vs-chip'>VS</div>", unsafe_allow_html=True)

                # 점수 B
                m.score_b = st.selectbox(
                    label="", options=list(range(7)), index=sb,
                    key=f"sb_{tour.current_round}_{mi}",
                    label_visibility="collapsed", disabled=st.session_state.view_only
                )
                st.markdown(f"<div class='team-label'>{b1}&{b2}</div>", unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            if st.button("💾 이 라운드 저장/갱신", type="primary", disabled=st.session_state.view_only, key=f"save_round_{tour.current_round}"):
                recompute_all_stats(tour)
                if tour.current_round < max(total - 1, 0):
                    tour.current_round += 1
                if st.session_state.current_tid:
                    save_tournament(tour, tid=st.session_state.current_tid)
                st.success("라운드 결과가 반영되었습니다. 다음 라운드로 이동합니다.")
                st.rerun()

    # 순위/통계
    with section("🏆 순위 / 통계"):
        all_stats = sorted(tour.stats.values(), key=lambda s: sort_key(tour, s), reverse=True)
        top = all_stats[:3] + [None] * max(0, 3 - len(all_stats))

        cards = []
        for rank, s in enumerate(top, start=1):
            crown = " 👑" if rank == 1 else ""
            if s is None:
                cards.append(
                    f'<div class="topcard top{rank}"><h4>TOP{rank}{crown}</h4><div class="name">-</div><div class="sub">승점 0 · 득실차 0</div></div>'
                )
            else:
                cards.append(
                    f'<div class="topcard top{rank}"><h4>TOP{rank}{crown}</h4><div class="name">{s.name}</div><div class="sub">승점 {s.pts} · 득실차 {s.gd}</div></div>'
                )
        st.markdown('<div class="topcards">' + "".join(cards) + "</div>", unsafe_allow_html=True)

        df = pd.DataFrame([
            {"선수": s.name, "경기수": s.gp, "승": s.w, "패": s.l, "득점": s.gf, "실점": s.ga, "득실차": s.gd, "승점": s.pts}
            for s in all_stats
        ])
        st.markdown("<div class='table-card'><div class='table-title'>🏆 전체 성적표</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='table-wrap'>{styled_table_html(df, font_px=15)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # CSV 다운로드 (UTF-8 권장 / Windows용 CP949)
        csv_utf8 = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 성적표 CSV (UTF-8 권장)", data=csv_utf8, file_name="standings_utf8.csv", mime="text/csv")
        csv_cp949 = df.to_csv(index=False).encode("cp949", errors="replace")
        st.download_button("⬇️ 성적표 CSV (Windows 엑셀용)", data=csv_cp949, file_name="standings_cp949.csv", mime="text/csv")

# =========================
# Router
# =========================
if st.session_state.page == "setup":
    render_setup()
else:
    # 안전 보정
    total = len(tour.rounds)
    if total == 0:
        ensure_loaded_from_query(auto_switch=False)
        total = len(tour.rounds)
    if total == 0:
        st.session_state.page = "setup"
        st.info("아직 생성된 대진이 없습니다. 먼저 대진을 생성해주세요.")
        st.stop()
    tour.current_round = max(0, min(tour.current_round, total - 1))
    render_tournament()
