import re
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from html import escape
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from streamlit_calendar import calendar

from core import generate_schedule, is_mock_mode


load_dotenv()

INK = "#0D1821"
YALE = "#344966"
POWDER = "#B4CDED"
PORCELAIN = "#F0F4EF"
SAGE = "#BFCC94"

st.set_page_config(page_title="AI 목표 플래너", page_icon="✦", layout="wide")
st.markdown(
    f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/sunn-us/SUIT/fonts/variable/woff2/SUIT-Variable.css');

    :root {{
        --ink: {INK};
        --yale: {YALE};
        --powder: {POWDER};
        --porcelain: {PORCELAIN};
        --sage: {SAGE};
        --surface: #ffffff;
        --line: rgba(52, 73, 102, 0.14);
        --muted: #647080;
    }}

    html, body, [class*="css"] {{
        font-family: "SUIT Variable", SUIT, -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{ background: var(--porcelain); color: var(--ink); }}
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}

    .block-container {{ max-width: 1540px; padding: 1.55rem 2rem 2.5rem; }}

    .top-brand {{ display: flex; align-items: center; gap: .75rem; min-height: 2.8rem; }}
    .brand-mark {{
        display: inline-grid; width: 2.15rem; height: 2.15rem; place-items: center;
        border-radius: .65rem; background: var(--yale); color: white; font-size: .95rem;
    }}
    .brand-copy {{ min-width: 0; }}
    .brand-title {{ margin: 0; color: var(--ink); font-size: 1.35rem; font-weight: 750; letter-spacing: -.035em; }}
    .brand-meta {{ margin: .12rem 0 0; color: var(--muted); font-size: .78rem; font-weight: 550; }}
    .demo-badge {{
        display: inline-flex; align-items: center; margin-left: .45rem; padding: .13rem .42rem;
        border: 1px solid rgba(52, 73, 102, .18); border-radius: 999px;
        background: rgba(180, 205, 237, .45); color: var(--yale);
        font-size: .58rem; font-weight: 750; letter-spacing: .07em; vertical-align: middle;
    }}

    .summary-row {{ display: flex; flex-wrap: wrap; gap: .42rem; margin: .75rem 0 1rem; }}
    .goal-chip {{
        display: inline-flex; align-items: center; gap: .38rem; max-width: 100%;
        padding: .34rem .58rem; border: 1px solid var(--line); border-radius: 999px;
        background: rgba(255, 255, 255, .76); color: var(--ink); font-size: .72rem;
    }}
    .goal-chip-name {{ max-width: 14rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; }}
    .goal-chip-hours {{ color: var(--yale); font-weight: 700; white-space: nowrap; }}
    .goal-chip-shortage {{ color: #8a4b45; font-weight: 700; white-space: nowrap; }}

    .stButton > button {{
        min-height: 2.55rem; border: 1px solid var(--yale); border-radius: .7rem;
        background: var(--yale); color: white; font-weight: 700; box-shadow: none;
    }}
    .stButton > button:hover, .stButton > button:focus {{
        border-color: var(--ink); background: var(--ink); color: white; box-shadow: none;
    }}
    [data-testid="stDownloadButton"] > button {{
        min-height: 2.55rem; width: 100%; border: 1px solid rgba(52, 73, 102, .28);
        border-radius: .7rem; background: white; color: var(--yale); font-weight: 700; box-shadow: none;
    }}
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:focus {{
        border-color: var(--yale); background: rgba(180, 205, 237, .24); color: var(--ink); box-shadow: none;
    }}

    [data-testid="stDialog"] [data-testid="stDialogContainer"] > div {{ border-radius: 1rem; }}
    [data-testid="stTextArea"] textarea {{
        min-height: 270px; padding: 1rem; border: 1px solid var(--line); border-radius: .75rem;
        background: #fbfcfa; color: var(--ink); font-size: .92rem; line-height: 1.62; box-shadow: none;
    }}
    [data-testid="stTextArea"] textarea:focus {{
        border-color: var(--yale); box-shadow: 0 0 0 3px rgba(52, 73, 102, .1);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        overflow: hidden; border: 1px solid var(--line) !important; border-radius: 1rem !important;
        background: var(--surface); box-shadow: 0 7px 22px rgba(13, 24, 33, .04);
    }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 1rem 1.05rem 1.05rem; }}

    .calendar-heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-bottom: .65rem; }}
    .calendar-title {{ margin: 0; color: var(--ink); font-size: 1rem; font-weight: 750; }}
    .calendar-help {{ margin: 0; color: var(--muted); font-size: .7rem; }}
    .hover-detail {{
        margin: 0 0 .6rem; padding: .5rem .68rem; border-radius: .55rem;
        background: rgba(180, 205, 237, .25); color: var(--yale); font-size: .72rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .empty-state {{
        display: grid; min-height: 62vh; place-items: center; padding: 2rem;
        border: 1px dashed rgba(52, 73, 102, .25); border-radius: 1rem;
        background: rgba(255, 255, 255, .52); text-align: center;
    }}
    .empty-icon {{ margin-bottom: .65rem; color: var(--yale); font-size: 1.5rem; }}
    .empty-title {{ margin: 0; color: var(--ink); font-size: 1.02rem; font-weight: 750; }}
    .empty-copy {{ max-width: 28rem; margin: .4rem auto 0; color: var(--muted); font-size: .78rem; line-height: 1.55; }}

    [data-testid="stExpander"] {{ border-color: var(--line); border-radius: .75rem; background: rgba(255, 255, 255, .58); }}
    [data-testid="stAlert"] {{ border-radius: .65rem; font-size: .78rem; }}
    .strategy-list, .detail-list {{ margin: .35rem 0 .75rem; padding-left: 1.15rem; }}
    .strategy-list li, .detail-list li {{
        margin: .28rem 0; color: var(--ink); font-size: .82rem; line-height: 1.48;
        word-break: keep-all; overflow-wrap: anywhere;
    }}
    .legend {{ display: flex; flex-wrap: wrap; gap: .8rem; margin: .7rem 0 .1rem; color: var(--muted); font-size: .69rem; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: .33rem; }}
    .legend-dot {{ width: .52rem; height: .52rem; border-radius: 50%; }}

    @media (max-width: 760px) {{
        .block-container {{ padding: 1rem .65rem 1.5rem; }}
        [data-testid="stHorizontalBlock"] {{ gap: .55rem; align-items: center; }}
        .brand-title {{ font-size: 1.13rem; }}
        .brand-meta {{ font-size: .68rem; }}
        .goal-chip-name {{ max-width: 9rem; }}
        .calendar-heading {{ align-items: flex-start; flex-direction: column; gap: .2rem; }}
        [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: .7rem .55rem .8rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


EXAMPLE_INPUT = """10월 초 한국사 시험이 있고 SKCT도 준비해야 해.
평일 9시부터 18시까지는 부트캠프야.
영단어는 매일 30분씩 하고 싶고, 평일 저녁에는 하루 2시간 정도만 쓸 수 있어.
주말은 비교적 여유로워."""


def _ics_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\r\n", "\\n").replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,")


def _fold_ics_line(line: str) -> str:
    parts: list[str] = []
    current = ""
    for character in line:
        if current and len((current + character).encode("utf-8")) > 75:
            parts.append(current)
            current = " " + character
        else:
            current += character
    parts.append(current)
    return "\r\n".join(parts)


def _schedule_to_ics(result) -> bytes:
    seoul = ZoneInfo("Asia/Seoul")
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AI Goal Planner//Schedule//KO",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:AI 목표 플래너",
    ]
    for event in result.events:
        start = event.start.replace(tzinfo=seoul) if event.start.tzinfo is None else event.start
        end = event.end.replace(tzinfo=seoul) if event.end.tzinfo is None else event.end
        uid_source = f"{event.title}|{start.isoformat()}|{end.isoformat()}|{event.kind}"
        description = [f"유형: {'고정 일정' if event.kind == 'fixed' else 'AI 생성 일정'}"]
        if event.goal:
            description.append(f"목표: {event.goal}")
        if event.reason:
            description.append(f"배치 이유: {event.reason}")
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{sha1(uid_source.encode('utf-8')).hexdigest()}@ai-goal-planner",
            f"DTSTAMP:{created_at}",
            f"DTSTART:{start.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}",
            f"DTEND:{end.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}",
            f"SUMMARY:{_ics_escape(event.title)}",
            f"DESCRIPTION:{_ics_escape(chr(10).join(description))}",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    return ("\r\n".join(_fold_ics_line(line) for line in lines) + "\r\n").encode("utf-8")


def _strategy_items(summary: str | None) -> list[str]:
    if not summary:
        return []
    chunks = re.split(r"\n+|(?<=[.!?])\s+", summary.strip())
    items = [re.sub(r"^(?:[-•]|\d+[.)])\s*", "", chunk).strip() for chunk in chunks]
    return [item for item in items if item][:4]


def _calendar_events(result) -> list[dict]:
    events = []
    for event in result.events:
        duration = int((event.end - event.start).total_seconds() // 60)
        if event.kind == "fixed":
            color, text_color, class_name = YALE, "#FFFFFF", "event-fixed event-long"
        elif duration <= 30:
            color, text_color, class_name = SAGE, INK, "event-habit event-short"
        else:
            color, text_color, class_name = POWDER, INK, "event-focus event-long"

        time_label = f"{event.start:%H:%M}–{event.end:%H:%M}"
        events.append({
            "title": event.title,
            "start": event.start.isoformat(),
            "end": event.end.isoformat(),
            "backgroundColor": color,
            "borderColor": color,
            "textColor": text_color,
            "classNames": class_name.split(),
            "extendedProps": {
                "fullTitle": event.title,
                "timeLabel": time_label,
                "kind": event.kind,
                "durationMinutes": duration,
                "goal": event.goal or "",
            },
        })
    return events


@st.dialog("새 계획 만들기", width="large")
def _plan_dialog() -> None:
    st.caption("목표, 마감, 고정 일정과 가능한 시간을 자연어로 입력해 주세요.")
    user_input = st.text_area(
        "목표와 일정",
        value=st.session_state.get("planner_input", ""),
        placeholder=EXAMPLE_INPUT,
        height=285,
        label_visibility="collapsed",
    )
    if st.button("계획 생성", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("일정을 입력해 주세요.")
            return
        try:
            with st.spinner("일정을 해석하고 검증하는 중입니다..."):
                st.session_state.schedule = generate_schedule(user_input)
                st.session_state.planner_input = user_input
            st.rerun()
        except Exception as error:
            st.error(f"일정을 만들지 못했습니다: {error}")


def _render_goal_chips(result) -> None:
    chips = []
    for goal in result.goals:
        required = f"/{goal.required_hours:g}h" if goal.required_hours is not None else ""
        shortage = (
            f'<span class="goal-chip-shortage">-{goal.shortage_hours:g}h</span>'
            if goal.shortage_hours > 0
            else ""
        )
        chips.append(
            f'<span class="goal-chip">'
            f'<span class="goal-chip-name">{escape(goal.title)}</span>'
            f'<span class="goal-chip-hours">{goal.allocated_hours:g}h{required}</span>'
            f'{shortage}</span>'
        )
    st.html(f'<div class="summary-row">{"".join(chips)}</div>')


def _render_plan_details(result) -> None:
    recurrence_labels = {
        "daily": "매일", "weekdays": "평일", "weekends": "주말",
        "flexible": "유연 배치", "once": "1회",
    }
    priority_labels = {"high": "높음", "medium": "보통", "low": "낮음"}

    with st.expander("AI가 이해한 계획"):
        if result.strategy_summary:
            st.caption("계획 전략")
            strategy_html = "".join(f"<li>{escape(item)}</li>" for item in _strategy_items(result.strategy_summary))
            st.html(f'<ul class="strategy-list">{strategy_html}</ul>')

        st.caption("고정 일정")
        if result.plan_spec.fixed_schedules:
            st.dataframe(
                [{
                    "일정": item.title,
                    "반복": recurrence_labels[item.recurrence],
                    "시간": f"{item.start_time}~{item.end_time}",
                    "날짜": item.event_date.isoformat() if item.event_date else "-",
                } for item in result.plan_spec.fixed_schedules],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("해석된 고정 일정이 없습니다.")

        st.caption("목표와 우선순위")
        st.dataframe(
            [{
                "목표": goal.title,
                "우선순위": priority_labels[goal.priority],
                "반복": recurrence_labels[goal.recurrence],
                "회당": f"{goal.session_minutes}분" if goal.session_minutes else "-",
                "주당 목표": f"{goal.target_sessions_per_week}회" if goal.target_sessions_per_week else "-",
                "필요": f"{goal.required_minutes}분" if goal.required_minutes else "-",
                "마감": goal.deadline.isoformat() if goal.deadline else "-",
            } for goal in result.plan_spec.goals],
            hide_index=True,
            use_container_width=True,
        )

        if result.plan_spec.availability_rules:
            st.caption("가능 시간과 선호")
            st.dataframe(
                [{
                    "적용": recurrence_labels[rule.recurrence],
                    "시작": rule.start_time or "-",
                    "종료": rule.end_time or "-",
                    "하루 최대": f"{rule.max_minutes_per_day}분" if rule.max_minutes_per_day else "-",
                    "선호": rule.preference,
                } for rule in result.plan_spec.availability_rules],
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("가정 및 주의사항"):
        if result.warning:
            st.warning(result.warning)
        elif not result.plan_spec.warnings and not result.plan_spec.constraints:
            st.caption("추가 가정이나 주의사항이 없습니다.")

        if result.plan_spec.warnings:
            st.caption("해석 과정의 가정")
            assumptions = "".join(f"<li>{escape(note)}</li>" for note in result.plan_spec.warnings)
            st.html(f'<ul class="detail-list">{assumptions}</ul>')
        if result.plan_spec.constraints:
            st.caption("기타 사용자 조건")
            constraints = "".join(f"<li>{escape(note)}</li>" for note in result.plan_spec.constraints)
            st.html(f'<ul class="detail-list">{constraints}</ul>')


def render_app() -> None:
    result = st.session_state.get("schedule")
    demo_badge = '<span class="demo-badge">DEMO</span>' if is_mock_mode() else ""

    header_columns = [4.8, 1.1, 1.1] if result else [5.2, 1]
    columns = st.columns(header_columns, vertical_alignment="center")
    header_main, header_action = columns[:2]
    with header_main:
        period = (
            f"{result.plan_start:%Y.%m.%d} – {result.plan_end:%Y.%m.%d}"
            if result
            else "목표와 제약을 입력해 실행 가능한 계획을 만드세요"
        )
        st.html(
            f'<div class="top-brand"><span class="brand-mark">✦</span>'
            f'<div class="brand-copy"><h1 class="brand-title">AI 목표 플래너 {demo_badge}</h1>'
            f'<p class="brand-meta">{escape(period)}</p></div></div>'
        )
    with header_action:
        button_label = "새 계획 만들기 / 수정" if result else "계획 만들기"
        if st.button(button_label, type="primary", use_container_width=True):
            _plan_dialog()
    if result:
        with columns[2]:
            st.download_button(
                "캘린더 다운로드",
                data=_schedule_to_ics(result),
                file_name=f"ai_plan_{result.plan_start:%Y%m%d}_{result.plan_end:%Y%m%d}.ics",
                mime="text/calendar; charset=utf-8",
                use_container_width=True,
            )

    if not result:
        st.html(
            '<div class="empty-state"><div><div class="empty-icon">✦</div>'
            '<h2 class="empty-title">아직 생성된 계획이 없습니다</h2>'
            '<p class="empty-copy">상단의 계획 만들기 버튼을 눌러 목표, 마감, 고정 일정과 가능한 시간을 입력해 주세요.</p>'
            '</div></div>'
        )
        return

    _render_goal_chips(result)

    with st.container(border=True):
        st.html(
            '<div class="calendar-heading"><h2 class="calendar-title">일정 캘린더</h2>'
            '<p class="calendar-help">주·일·목록 보기를 전환할 수 있습니다. 일정에 마우스를 올리면 전체 정보를 확인할 수 있습니다.</p></div>'
        )
        hover_slot = st.empty()
        hover_slot.html('<div class="hover-detail">일정에 마우스를 올리면 시간과 전체 제목이 표시됩니다.</div>')

        calendar_options = {
            "initialView": "timeGridWeek",
            "firstDay": 1,
            "locale": "ko",
            "height": "auto",
            "allDaySlot": False,
            "slotMinTime": "08:00:00",
            "slotMaxTime": "24:00:00",
            "scrollTime": "08:00:00",
            "slotDuration": "00:30:00",
            "expandRows": True,
            "nowIndicator": True,
            "editable": False,
            "selectable": False,
            "eventDisplay": "block",
            "eventMinHeight": 24,
            "eventShortHeight": 22,
            "slotEventOverlap": False,
            "handleWindowResize": True,
            "stickyHeaderDates": True,
            "dayHeaderFormat": {"weekday": "short", "month": "numeric", "day": "numeric", "omitCommas": True},
            "slotLabelFormat": {"hour": "2-digit", "minute": "2-digit", "hour12": False},
            "eventTimeFormat": {"hour": "2-digit", "minute": "2-digit", "hour12": False},
            "buttonText": {"today": "오늘", "week": "주", "day": "일", "list": "목록"},
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": "timeGridWeek,timeGridDay,listWeek",
            },
            "initialDate": result.plan_start.isoformat(),
            "validRange": {
                "start": result.plan_start.isoformat(),
                "end": (result.plan_end + timedelta(days=1)).isoformat(),
            },
        }

        calendar_state = calendar(
            events=_calendar_events(result),
            options=calendar_options,
            callbacks=["eventMouseEnter", "eventClick"],
            custom_css=f"""
                .fc {{ color: {INK}; font-family: "SUIT Variable", SUIT, sans-serif; font-size: 13px; }}
                .fc .fc-toolbar.fc-header-toolbar {{ flex-wrap: wrap; gap: 8px; margin: 0 0 15px; }}
                .fc .fc-toolbar-chunk {{ display: flex; align-items: center; gap: 4px; }}
                .fc .fc-toolbar-title {{ color: {INK}; font-size: 1rem; font-weight: 750; letter-spacing: -.025em; }}
                .fc .fc-button-primary {{
                    padding: .37rem .55rem; border: 1px solid rgba(52,73,102,.2); border-radius: .48rem;
                    background: white; color: {YALE}; box-shadow: none; font-size: .7rem; font-weight: 650;
                }}
                .fc .fc-button-primary:hover, .fc .fc-button-primary:focus,
                .fc .fc-button-primary:not(:disabled).fc-button-active,
                .fc .fc-button-primary:not(:disabled):active {{
                    border-color: {YALE}; background: {YALE}; color: white; box-shadow: none;
                }}
                .fc .fc-scrollgrid, .fc-theme-standard .fc-scrollgrid {{ border: 0 !important; }}
                .fc-theme-standard td, .fc-theme-standard th {{ border-color: rgba(52,73,102,.10); }}
                .fc .fc-col-header-cell {{ border-top: 0; background: white; }}
                .fc .fc-col-header-cell-cushion {{
                    padding: 8px 3px 10px; color: #687482; font-size: .72rem; font-weight: 650; text-decoration: none;
                }}
                .fc .fc-day-today, .fc .fc-timegrid-col.fc-day-today {{ background: rgba(180,205,237,.18) !important; }}
                .fc .fc-day-today .fc-col-header-cell-cushion {{ color: {YALE}; }}
                .fc .fc-timegrid-axis, .fc .fc-timegrid-slot-label {{ color: #8a929c; font-size: .64rem; }}
                .fc .fc-timegrid-slot {{ height: 1.95rem; }}
                .fc .fc-timegrid-now-indicator-line {{ border-color: {YALE}; border-width: 1px; }}
                .fc .fc-timegrid-now-indicator-arrow {{ border-color: {YALE}; border-bottom-color: transparent; border-top-color: transparent; }}
                .fc .fc-timegrid-event-harness {{ margin: 2px 3px; }}
                .fc .fc-event {{
                    padding: 4px 6px; border: 0 !important; border-radius: 6px !important;
                    box-shadow: none; overflow: hidden; cursor: default;
                }}
                .fc .fc-event-main, .fc .fc-event-main-frame,
                .fc .fc-event-title-container {{ min-width: 0; overflow: hidden; }}
                .fc .fc-event-main-frame {{ display: flex; flex-direction: column; gap: 1px; }}
                .fc .fc-event-time {{ flex: 0 0 auto; overflow: hidden; font-size: .61rem; font-weight: 550; line-height: 1.15; text-overflow: ellipsis; white-space: nowrap; }}
                .fc .fc-event-title {{
                    display: block; min-width: 0; overflow: hidden; font-size: .69rem;
                    font-weight: 750; line-height: 1.2; text-overflow: ellipsis; white-space: nowrap;
                }}
                .fc .event-short {{ min-height: 22px; padding: 3px 5px; }}
                .fc .event-short .fc-event-time {{ display: none; }}
                .fc .event-short .fc-event-title {{ font-size: .67rem; line-height: 1.15; }}
                .fc .event-long .fc-event-time {{ display: block; }}
                .fc .fc-list-event-title, .fc .fc-list-event-time {{ color: {INK}; font-size: .76rem; }}
                .fc .fc-list-event-dot {{ border-color: {YALE}; }}
                @media (max-width: 700px) {{
                    .fc {{ font-size: 11px; }}
                    .fc .fc-toolbar {{ align-items: stretch; flex-direction: column; }}
                    .fc .fc-toolbar-chunk {{ justify-content: center; flex-wrap: wrap; }}
                    .fc .fc-toolbar-title {{ font-size: .92rem; }}
                    .fc .fc-button-primary {{ padding: .34rem .48rem; font-size: .66rem; }}
                    .fc .fc-col-header-cell-cushion {{ font-size: .64rem; }}
                    .fc .fc-event {{ padding: 3px 4px; }}
                    .fc .fc-event-title {{ font-size: .62rem; }}
                }}
            """,
            key="schedule_calendar",
        )

        event_payload = (
            calendar_state.get("eventMouseEnter", {}).get("event")
            or calendar_state.get("eventClick", {}).get("event")
            if calendar_state
            else None
        )
        if event_payload:
            props = event_payload.get("extendedProps", {})
            title = escape(props.get("fullTitle") or event_payload.get("title", "일정"))
            time_label = escape(props.get("timeLabel", ""))
            kind_label = "고정 일정" if props.get("kind") == "fixed" else "AI 배치 일정"
            hover_slot.html(
                f'<div class="hover-detail"><strong>{title}</strong> · {time_label} · {kind_label}</div>'
            )

        st.html(
            f'<div class="legend">'
            f'<span class="legend-item"><span class="legend-dot" style="background:{YALE}"></span>고정 일정</span>'
            f'<span class="legend-item"><span class="legend-dot" style="background:{POWDER}"></span>AI 집중 일정</span>'
            f'<span class="legend-item"><span class="legend-dot" style="background:{SAGE}"></span>30분 이하 습관</span>'
            f'</div>'
        )

    _render_plan_details(result)


render_app()
