"""자연어 해석 → 일정 초안 → 검증 → 한 번의 수정으로 이어지는 핵심 로직입니다.

LLM은 배치를 판단하고, Python은 시간 계산과 명시된 제약을 검사합니다.
이 모듈은 Streamlit 없이도 사용할 수 있습니다.
"""

# %% imports
import json
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

# 배포 환경에 주입된 값은 유지하고, 로컬에서는 현재 폴더의 .env를 읽습니다.
load_dotenv(".env")

Recurrence = Literal["daily", "weekdays", "weekends", "flexible", "once"]
FixedRecurrence = Literal["daily", "weekdays", "weekends", "once"]
Priority = Literal["high", "medium", "low"]
TIME_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
END_TIME_PATTERN = r"^(?:(?:[01]\d|2[0-3]):[0-5]\d|24:00)$"
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
DAY_LABELS = {"daily": "매일", "weekdays": "평일", "weekends": "주말"}
MODEL_DEFAULTS = {
    "google_genai": ("gemini-3.6-flash", "GOOGLE_API_KEY"),
    "openai": ("gpt-4.1-mini", "OPENAI_API_KEY"),
}

# %% input_models
class CalendarEvent(BaseModel):
    """화면과 내보내기에 사용하는 일정입니다. fixed는 원래 일정이고 generated는 AI 배치입니다."""

    title: str
    start: datetime
    end: datetime
    kind: Literal["fixed", "generated"]
    goal: str | None = None
    reason: str | None = None

class ParsedGoal(BaseModel):
    """목표의 시간·마감 조건입니다. 명시되지 않은 값은 None으로 남깁니다."""

    title: str
    deadline: date | None = None
    priority: Priority
    priority_reason: str
    recurrence: Recurrence = "flexible"
    session_minutes: int | None = Field(default=None, ge=1)
    target_sessions_per_week: int | None = Field(default=None, ge=1, le=14)
    required_minutes: int | None = Field(
        default=None, ge=1,
        description="사용자가 전체 계획 기간의 총 필요 시간을 직접 명시한 경우에만 저장합니다. 회당 시간이나 주당 횟수로 계산한 값은 넣지 않습니다.",
    )
    constraints: list[str] = Field(default_factory=list)

class FixedSchedule(BaseModel):
    """반복 규칙을 가진 고정 일정입니다. once일 때 event_date를 사용합니다."""

    title: str
    recurrence: FixedRecurrence
    start_time: str = Field(pattern=TIME_PATTERN)
    end_time: str = Field(pattern=END_TIME_PATTERN)
    event_date: date | None = None

class LifestylePreferences(BaseModel):
    """사용자가 직접 말한 활동·식사·취침 시간만 저장합니다."""

    active_start: str | None = Field(default=None, pattern=TIME_PATTERN)
    active_end: str | None = Field(default=None, pattern=TIME_PATTERN)
    lunch_start: str | None = Field(default=None, pattern=TIME_PATTERN)
    lunch_end: str | None = Field(default=None, pattern=TIME_PATTERN)
    dinner_start: str | None = Field(default=None, pattern=TIME_PATTERN)
    dinner_end: str | None = Field(default=None, pattern=TIME_PATTERN)
    wind_down_start: str | None = Field(default=None, pattern=TIME_PATTERN)

class AvailabilityRule(BaseModel):
    """가능 시간과 하루 한도는 검증하고, preference는 LLM 판단에 맡깁니다."""

    recurrence: Literal["daily", "weekdays", "weekends"]
    start_time: str | None = Field(default=None, pattern=TIME_PATTERN)
    end_time: str | None = Field(default=None, pattern=END_TIME_PATTERN)
    max_minutes_per_day: int | None = Field(default=None, ge=1)
    preference: Literal["preferred", "normal", "avoid"] = "normal"

class PlanSpec(BaseModel):
    """Interpretation 체인이 반환하는 목표와 제약의 묶음입니다."""

    plan_start: date
    plan_end: date
    goals: list[ParsedGoal]
    fixed_schedules: list[FixedSchedule] = Field(default_factory=list)
    lifestyle: LifestylePreferences = Field(default_factory=LifestylePreferences)
    availability_rules: list[AvailabilityRule] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

# %% output_models
class PlannedEvent(BaseModel):
    """Planner가 제안한 학습·작업 한 건입니다. 아직 검증되지 않은 초안입니다."""

    title: str
    date: date
    start_time: str = Field(pattern=TIME_PATTERN)
    end_time: str = Field(pattern=END_TIME_PATTERN)
    goal: str
    reason: str | None = None

class DraftSchedule(BaseModel):
    """Planner와 Repair가 공유하는 구조화된 출력 형식입니다."""

    events: list[PlannedEvent]
    strategy_summary: str = Field(description="2~4개의 짧은 핵심 문장으로 작성한 계획 전략")
    warnings: list[str] = Field(default_factory=list)

class ValidationIssue(BaseModel):
    """event_index로 문제 일정을 가리킵니다. 누락처럼 일정이 없으면 None입니다."""

    code: str
    message: str
    event_index: int | None = None

class ValidationResult(BaseModel):
    """검증 결과를 Repair의 입력으로 전달합니다."""

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

class GoalSummary(BaseModel):
    """최종 남은 일정의 시간을 합산한 목표별 요약입니다."""

    title: str
    priority: Priority
    priority_reason: str
    deadline: date | None = None
    required_hours: float | None = Field(default=None, ge=0)
    allocated_hours: float = Field(default=0, ge=0)
    shortage_hours: float = Field(default=0, ge=0)

class ScheduleResult(BaseModel):
    """검증 후 남은 일정과 가정·주의사항을 화면에 전달합니다."""

    plan_start: date
    plan_end: date
    events: list[CalendarEvent]
    goals: list[GoalSummary] = Field(default_factory=list)
    plan_spec: PlanSpec
    strategy_summary: str | None = None
    warning: str | None = None

# %% interpret_prompt
# 역할과 제약은 system에, 실행마다 바뀌는 사용자 원문은 human에 둡니다.
INTERPRET_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 일정 입력 해석기다. 아직 날짜별 일정을 배치하지 말고 사용자의 의미만 PlanSpec으로 구조화한다.
현재 날짜는 {current_date}다.

계획 기간:
- 시험일·마감일 등 관련 날짜가 있으면 가장 늦은 관련 날짜까지 계획한다.
- 날짜가 전혀 없으면 {default_plan_end}까지 14일로 정한다.
- 관련 날짜가 {max_plan_end}보다 멀면 plan_end는 {max_plan_end}로 제한하고 warnings에 적는다.
- '10월 초', '이달 말' 같은 모호한 날짜는 합리적으로 해석하고 warnings에 적는다.

추출 규칙:
- 목표마다 deadline, priority, priority_reason, recurrence, session_minutes, target_sessions_per_week, required_minutes, constraints를 추출한다.
- recurrence는 daily, weekdays, weekends, flexible, once 중 하나다.
- required_minutes에는 사용자가 '총 5시간 필요'처럼 전체 계획 기간의 총 필요 시간을 직접 명시한 경우에만 값을 넣는다.
- '매일 20분', '주 3회'처럼 반복 기준만 있는 경우 required_minutes는 null로 두고 session_minutes와 target_sessions_per_week에만 반영한다.
- session_minutes × target_sessions_per_week를 required_minutes로 계산하지 않는다.
- 총 시간이 없는 목표는 회당 학습 시간이나 주당 횟수에 대한 reasonable hint를 제공할 수 있지만 근거 없이 정밀한 전체 총시간을 지어내지 않는다.
- '매일 30분'은 recurrence='daily', session_minutes=30이다.
- 고정 일정은 fixed_schedules로 분리하고 24시간제 start_time/end_time을 정확히 적는다.
- '평일 오전 9시부터 오후 6시 부트캠프'는 weekdays 09:00~18:00이다.
- 일회성 고정 일정은 recurrence='once'와 event_date를 사용한다.
- 사용자가 명시한 식사·취침·활동 가능 시간만 lifestyle에 적고, 말하지 않은 항목은 null로 둔다.
- 요일별 가능 시간, 하루 최대 사용 시간, 선호도는 availability_rules로 구조화한다.
- '평일 저녁에는 하루 2시간 정도만 가능'은 recurrence='weekdays', start_time='18:00', max_minutes_per_day=120으로 해석한다.
- '주말은 비교적 여유로워'는 recurrence='weekends', preference='preferred'로 해석한다.
- start_time, end_time, max_minutes_per_day는 명시적 hard constraint이고 preference는 Planner가 참고하는 선호 정보다.
- '저녁'처럼 모호한 시간 표현을 18:00 등으로 해석했다면 warnings에 가정을 남긴다.
- 기타 사용자 제약은 constraints에 보존한다.
- CalendarEvent나 날짜별 학습 일정은 만들지 않는다.
""",
        ),
        ("human", "{user_input}"),
    ]
)

# %% plan_prompt
# 원문 대신 PlanSpec과 날짜별 고정 일정을 보내 배치에 필요한 맥락을 좁힙니다.
PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사람이 실제로 실천할 수 있는 학습·작업 계획을 만드는 플래너다.
PlanSpec과 날짜별로 펼쳐진 fixed_events를 보고 generated 일정만 DraftSchedule에 반환한다. fixed 일정은 events에 다시 넣지 않는다.

계획 원칙:
- 사용자의 고정 일정과 절대 겹치지 않는다.
- 장시간 고정 일정이 있는 평일에는 저녁을 지나치게 빡빡하게 채우지 않는다.
- 긴 고정 일정 직후에는 식사와 휴식을 고려한다.
- 주말처럼 여유가 큰 날은 긴 집중 세션에 적극 활용할 수 있지만 모든 빈 시간을 억지로 채우지 않는다.
- 마감이 가까운 목표에 더 많은 비중을 두되, 중요하지만 마감이 덜 급한 목표도 완전히 무시하지 않는다.
- 매일 해야 하는 짧은 습관은 주요 집중 공부를 방해하지 않는 시간에 넣는다.
- 연속해서 긴 집중 작업을 과도하게 배치하지 않고, 일반적으로 너무 늦은 밤까지 학습을 미루지 않는다.
- 평일과 주말의 현실적인 피로도 차이를 고려한다.
- 같은 목표를 기계적으로 매일 같은 시각에 반복할 필요는 없다.
- 전체 기간을 보고 사람이 실제로 실행할 법한 계획을 만든다.
- daily, weekdays, weekends, once 반복 조건과 명시된 session_minutes를 지킨다.
- required_minutes가 명시된 목표는 전체 계획 기간의 실제 배정 시간이 required_minutes를 크게 초과하거나 부족하지 않게 한다.
- required_minutes는 최소량이 아니라 전체 계획 기간의 목표량이다.
- target_sessions_per_week가 있으면 계획 기간에 맞는 적정 세션 수를 판단하는 기준으로 사용한다.
- 사용자가 명시한 총 시간이나 반복 횟수를 가장 우선한다.
- 현실적으로 목표량을 전부 배치할 수 없다면 억지로 채우지 말고 부족한 내용을 warnings에 남긴다.
- availability_rules의 start_time, end_time, max_minutes_per_day는 반드시 지키는 hard constraint다.
- availability_rules의 preferred/avoid는 배치 판단에 반영하되, 선호만으로 일정량을 기계적으로 채우거나 비우지 않는다.
- event.goal은 PlanSpec의 goal title과 정확히 일치시킨다.
- event.reason에는 필요한 경우 왜 그 시간에 배치했는지 짧게 적는다.
- strategy_summary는 실행 전략만 2~4개의 짧은 핵심 문장으로 작성한다. 긴 한 문단이나 불필요한 세부 설명은 피한다.
""",
        ),
        (
            "human",
            """다음 해석 결과와 고정 일정을 바탕으로 전체 DraftSchedule을 만들어라.

[PlanSpec]
{plan_spec_json}

[fixed_events]
{fixed_events_json}
""",
        ),
    ]
)

# %% repair_prompt
# 기존 초안과 검증 오류를 함께 보내 수정 범위를 제한합니다.
REPAIR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 일정 검증 결과를 반영하는 수정 플래너다.
원래 계획의 의도와 좋은 배치는 최대한 유지하고, validation issues를 해결하는 데 필요한 일정만 수정해 전체 DraftSchedule을 다시 반환한다.
fixed 일정은 events에 넣지 말고, generated 일정만 반환한다.
strategy_summary는 수정된 계획을 반영하되 2~4개의 짧은 핵심 문장으로 유지한다.
""",
        ),
        (
            "human",
            """[PlanSpec]
{plan_spec_json}

[fixed_events]
{fixed_events_json}

[기존 DraftSchedule]
{draft_json}

[validation issues]
{issues_json}
""",
        ),
    ]
)

# %% model_config
def model_settings() -> tuple[str, str, str]:
    """provider, 모델명, 키의 환경변수 이름을 반환합니다. 키 값은 노출하지 않습니다."""
    provider = os.getenv("MODEL_PROVIDER", "google_genai").strip()
    if provider not in MODEL_DEFAULTS:
        raise ValueError(f"지원하지 않는 MODEL_PROVIDER입니다: {provider}")
    default_model, key_name = MODEL_DEFAULTS[provider]
    model_name = os.getenv("MODEL_NAME", "").strip() or default_model
    return provider, model_name, key_name


def is_mock_mode() -> bool:
    """선택한 provider의 키가 없을 때만 개발용 예시를 사용합니다."""
    _, _, key_name = model_settings()
    return os.getenv(key_name, "").strip() in {"", "fake-key"}

# %% time_helpers
def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _matches(day: date, recurrence: Recurrence, event_date: date | None = None) -> bool:
    """고정 일정·반복 목표·가용 조건에 같은 요일 판정을 사용합니다."""
    if recurrence == "once":
        return day == event_date
    return {
        "daily": True,
        "weekdays": day.weekday() < 5,
        "weekends": day.weekday() >= 5,
    }.get(recurrence, False)


def _clock_minutes(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _duration(event: CalendarEvent) -> int:
    return int((event.end - event.start).total_seconds() // 60)


def _event_bounds(
    day: date, start_value: str, end_value: str, *, overnight: bool = False
) -> tuple[datetime, datetime]:
    """24:00은 다음 날 00:00으로 변환합니다. 야간 고정 일정만 자정 넘김을 허용합니다."""
    midnight = datetime.combine(day, time())
    start = midnight + timedelta(minutes=_clock_minutes(start_value))
    end = midnight + timedelta(minutes=_clock_minutes(end_value))
    if overnight and end <= start:
        end += timedelta(days=1)
    return start, end


def _convert_planned_event(event: PlannedEvent) -> CalendarEvent:
    # 초안의 잘못된 시간을 다음 날로 보정하지 않고 Validator에 그대로 전달합니다.
    start, end = _event_bounds(event.date, event.start_time, event.end_time)
    return CalendarEvent(
        title=event.title, start=start, end=end, kind="generated",
        goal=event.goal, reason=event.reason,
    )

# %% fixed_and_windows
def _normalize_plan_period(plan: PlanSpec) -> PlanSpec:
    warnings = list(plan.warnings)
    plan_end = plan.plan_end
    maximum_end = plan.plan_start + timedelta(days=27)
    if plan_end < plan.plan_start:
        plan_end = plan.plan_start + timedelta(days=13)
        warnings.append("계획 종료일이 시작일보다 빨라 기본 14일로 조정했습니다.")
    if plan_end > maximum_end:
        plan_end = maximum_end
        warnings.append(f"상세 일정은 {maximum_end.isoformat()}까지 최대 4주만 배치합니다.")
    return plan.model_copy(update={
        "plan_end": plan_end, "warnings": list(dict.fromkeys(warnings)),
    })


def _expand_fixed_schedules(plan: PlanSpec) -> list[CalendarEvent]:
    """LLM에게 반복 이벤트를 나열시키지 않고 Python으로 날짜별 일정을 펼칩니다."""
    events = []
    for schedule in plan.fixed_schedules:
        for day in _days(plan.plan_start, plan.plan_end):
            if not _matches(day, schedule.recurrence, schedule.event_date):
                continue
            start, end = _event_bounds(
                day, schedule.start_time, schedule.end_time, overnight=True,
            )
            events.append(CalendarEvent(
                title=schedule.title, start=start, end=end, kind="fixed",
            ))
    return sorted(events, key=lambda event: event.start)


def _expected_dates(goal: ParsedGoal, plan: PlanSpec) -> list[date]:
    last_day = min(goal.deadline or plan.plan_end, plan.plan_end)
    if goal.recurrence == "once":
        return [min(max(last_day, plan.plan_start), plan.plan_end)]
    return [day for day in _days(plan.plan_start, last_day) if _matches(day, goal.recurrence)]


def _overlaps(event: CalendarEvent, other: CalendarEvent) -> bool:
    # 끝 시각과 다음 시작 시각이 같은 인접 일정은 충돌이 아닙니다.
    return event.start < other.end and event.end > other.start


def _within_time_window(
    event: CalendarEvent, start_time: str | None, end_time: str | None,
) -> bool:
    """활동 시간과 요일별 가용 시간에 같은 범위 검사를 사용합니다."""
    start_limit = _clock_minutes(start_time or "00:00")
    end_limit = _clock_minutes(end_time or "24:00")
    event_start = event.start.hour * 60 + event.start.minute
    event_end = event_start + _duration(event)
    # 20:00~01:00처럼 자정을 넘는 가능 시간도 하나의 구간으로 비교합니다.
    if end_limit <= start_limit:
        end_limit += 24 * 60
        if event_start < start_limit:
            event_start += 24 * 60
            event_end += 24 * 60
    return event_start >= start_limit and event_end <= end_limit


def _explicit_unavailable_blocks(
    day: date, lifestyle: LifestylePreferences,
) -> list[tuple[datetime, datetime]]:
    pairs = [
        (lifestyle.lunch_start, lifestyle.lunch_end),
        (lifestyle.dinner_start, lifestyle.dinner_end),
        (lifestyle.wind_down_start, lifestyle.active_end),
    ]
    return [_event_bounds(day, start, end, overnight=True)
            for start, end in pairs if start and end]

# %% event_validation
def _event_issues(
    index: int, event: CalendarEvent, plan: PlanSpec,
    goal: ParsedGoal | None, fixed_events: list[CalendarEvent],
) -> list[ValidationIssue]:
    """한 건만 보고 판단할 수 있는 시간·목표·사용자 조건을 검사합니다."""
    issues = []

    def add(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, message=message, event_index=index))

    if event.end <= event.start:
        add("INVALID_TIME", f"'{event.title}'의 종료 시각이 시작 시각보다 늦지 않습니다.")
        return issues  # 음수 길이를 이후 일일 합계에 더하지 않습니다.
    if not plan.plan_start <= event.start.date() <= plan.plan_end:
        add("OUTSIDE_PLAN", f"'{event.title}'이 계획 기간 밖에 있습니다.")
    if goal is None:
        add("UNKNOWN_GOAL", f"'{event.goal}'은 PlanSpec에 없는 목표입니다.")
        return issues
    if goal.deadline and event.start.date() > goal.deadline:
        add("AFTER_DEADLINE", f"'{event.goal}' 일정이 마감일 이후에 있습니다.")
    if goal.recurrence in {"weekdays", "weekends"} and not _matches(event.start.date(), goal.recurrence):
        add("RECURRENCE_DAY", f"'{event.goal}'이 {DAY_LABELS[goal.recurrence]} 반복 조건을 벗어났습니다.")
    if goal.recurrence != "flexible" and goal.session_minutes:
        # 반복 습관의 길이에만 허용 오차를 적용합니다. flexible 세션은 강제하지 않습니다.
        tolerance = max(10, round(goal.session_minutes * 0.2))
        if abs(_duration(event) - goal.session_minutes) > tolerance:
            add("SESSION_DURATION", f"'{event.goal}'의 회당 시간이 {goal.session_minutes}분과 크게 다릅니다.")
    if not _within_time_window(event, plan.lifestyle.active_start, plan.lifestyle.active_end):
        add("USER_TIME_CONSTRAINT", f"'{event.title}'이 사용자의 활동 가능 시간을 벗어났습니다.")
    # 자정을 넘는 생활 시간은 전날 시작한 구간과도 비교합니다.
    blocks = [block for day in (event.start.date() - timedelta(days=1), event.start.date())
              for block in _explicit_unavailable_blocks(day, plan.lifestyle)]
    if any(event.start < end and event.end > start for start, end in blocks):
        add("USER_TIME_CONSTRAINT", f"'{event.title}'이 사용자가 명시한 생활 시간과 겹칩니다.")
    if any(_overlaps(event, fixed) for fixed in fixed_events):
        add("FIXED_OVERLAP", f"'{event.title}'이 고정 일정과 겹칩니다.")

    for rule in plan.availability_rules:
        if _matches(event.start.date(), rule.recurrence) and not _within_time_window(
            event, rule.start_time, rule.end_time,
        ):
            label = DAY_LABELS[rule.recurrence]
            add("AVAILABILITY_TIME", f"'{event.title}'이 {label} 가능 시간 {rule.start_time or '00:00'}~{rule.end_time or '24:00'}을 벗어났습니다.")
    return issues

# %% draft_validation
def validate_draft(
    plan: PlanSpec, draft: DraftSchedule,
    fixed_events: list[CalendarEvent] | None = None,
) -> ValidationResult:
    """개별 일정 → 일정 간 충돌·하루 한도 → 반복 누락 순서로 검증합니다."""
    fixed_events = fixed_events if fixed_events is not None else _expand_fixed_schedules(plan)
    goals = {goal.title: goal for goal in plan.goals}
    events = [_convert_planned_event(event) for event in draft.events]
    issues = []

    # 1. 먼저 개별 일정의 명시적인 오류를 수집합니다.
    for index, event in enumerate(events):
        issues.extend(_event_issues(index, event, plan, goals.get(event.goal), fixed_events))
    invalid = {issue.event_index for issue in issues}

    # 2. 제외된 일정은 다시 비교하거나 누적하지 않습니다. 정상 일정이 연쇄적으로 제외되는 것을 막기 위함입니다.
    accepted: list[CalendarEvent] = []
    daily_total: dict[date, int] = defaultdict(int)
    for index in sorted(range(len(events)), key=lambda i: events[i].start):
        if index in invalid:
            continue
        event = events[index]
        previous = next((other for other in accepted if _overlaps(event, other)), None)
        if previous is not None:
            issues.append(ValidationIssue(
                code="GENERATED_OVERLAP", event_index=index,
                message=f"'{previous.title}'과 '{event.title}'이 겹칩니다.",
            ))
            continue
        day = event.start.date()
        total = daily_total[day] + _duration(event)
        limits = [rule.max_minutes_per_day for rule in plan.availability_rules
                  if _matches(day, rule.recurrence) and rule.max_minutes_per_day is not None]
        if limits and total > min(limits):
            issues.append(ValidationIssue(
                code="DAILY_LIMIT_EXCEEDED", event_index=index,
                message=f"{day.isoformat()} 생성 일정이 하루 최대 {min(limits)}분을 초과했습니다. (합산 {total}분)",
            ))
            continue
        accepted.append(event)
        daily_total[day] = total

    # 3. 실제로 남은 일정의 날짜를 기준으로 daily/평일/주말 누락을 찾습니다.
    for goal in plan.goals:
        if goal.recurrence not in {"daily", "weekdays", "weekends"}:
            continue
        actual_dates = {event.start.date() for event in accepted if event.goal == goal.title}
        missing = sorted(set(_expected_dates(goal, plan)) - actual_dates)
        if missing:
            dates = ", ".join(day.strftime("%m-%d") for day in missing)
            issues.append(ValidationIssue(
                code="RECURRENCE_MISSING",
                message=f"'{goal.title}' 반복 일정이 누락된 날짜: {dates}",
            ))
    return ValidationResult(valid=not issues, issues=issues)

# %% result_building
def _required_minutes(goal: ParsedGoal, plan: PlanSpec) -> int | None:
    if goal.required_minutes is not None:
        return goal.required_minutes
    if goal.recurrence != "flexible" and goal.session_minutes:
        return len(_expected_dates(goal, plan)) * goal.session_minutes
    return None


def _build_schedule_result(
    plan: PlanSpec, draft: DraftSchedule, validation: ValidationResult,
    fixed_events: list[CalendarEvent],
) -> ScheduleResult:
    """위반 이벤트만 제외하고, 실제 남은 이벤트로 시간과 부족량을 집계합니다."""
    invalid = {issue.event_index for issue in validation.issues if issue.event_index is not None}
    generated = [_convert_planned_event(event) for i, event in enumerate(draft.events) if i not in invalid]
    allocated: dict[str, int] = defaultdict(int)
    for event in generated:
        allocated[event.goal] += _duration(event)

    # 해석 가정은 plan_spec.warnings에 유지하고 최종 주의사항과 섞지 않습니다.
    warnings = list(draft.warnings) + [f"미해결: {issue.message}" for issue in validation.issues]
    summaries = []
    for goal in plan.goals:
        required = _required_minutes(goal, plan)
        shortage = max(required - allocated[goal.title], 0) if required is not None else 0
        if shortage:
            warnings.append(f"'{goal.title}'은 {shortage / 60:g}시간이 부족합니다.")
        summaries.append(GoalSummary(
            title=goal.title, priority=goal.priority,
            priority_reason=goal.priority_reason, deadline=goal.deadline,
            required_hours=round(required / 60, 2) if required is not None else None,
            allocated_hours=round(allocated[goal.title] / 60, 2),
            shortage_hours=round(shortage / 60, 2),
        ))
    return ScheduleResult(
        plan_start=plan.plan_start, plan_end=plan.plan_end,
        events=sorted(fixed_events + generated, key=lambda event: event.start),
        goals=sorted(summaries, key=lambda goal: PRIORITY_ORDER[goal.priority]),
        plan_spec=plan, strategy_summary=draft.strategy_summary,
        warning=" ".join(dict.fromkeys(warnings)) or None,
    )

# %% process_draft
def process_draft(
    plan: PlanSpec, draft: DraftSchedule,
    repair: Callable[[dict[str, str]], DraftSchedule] | None = None,
    *, fixed_events: list[CalendarEvent] | None = None,
) -> ScheduleResult:
    """항상 검증하고, 필요한 경우에만 Repair를 한 번 실행합니다."""
    plan = _normalize_plan_period(plan)
    fixed = fixed_events if fixed_events is not None else _expand_fixed_schedules(plan)
    validation = validate_draft(plan, draft, fixed)
    if not validation.valid and repair is not None:
        # 검증 오류와 기존 초안을 수정 체인의 입력으로 전달합니다.
        draft = repair({
            "plan_spec_json": _json(plan), "fixed_events_json": _json(fixed),
            "draft_json": _json(draft), "issues_json": _json(validation.issues),
        })
        validation = validate_draft(plan, draft, fixed)
    return _build_schedule_result(plan, draft, validation, fixed)


def _json(value: BaseModel | list[BaseModel]) -> str:
    data = [item.model_dump(mode="json") for item in value] if isinstance(value, list) else value.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False)

# %% build_chain
def build_chain():
    """형식이 다른 세 LLM 체인을 만들고, Python 분기 처리를 LCEL에 연결합니다."""
    provider, model_name, _ = model_settings()
    model_options = {"temperature": 0} if provider == "openai" else {}
    llm = init_chat_model(model_name, model_provider=provider, **model_options)

    # 실습과 같은 Prompt | Model 구조이며, 각 단계의 출력 스키마만 다릅니다.
    interpret_chain = INTERPRET_PROMPT | llm.with_structured_output(PlanSpec)
    planner_chain = PLAN_PROMPT | llm.with_structured_output(DraftSchedule)
    repair_chain = REPAIR_PROMPT | llm.with_structured_output(DraftSchedule)

    def plan_validate_repair(plan: PlanSpec) -> ScheduleResult:
        plan = _normalize_plan_period(plan)
        fixed = _expand_fixed_schedules(plan)
        context = {"plan_spec_json": _json(plan), "fixed_events_json": _json(fixed)}
        draft = planner_chain.invoke(context)

        return process_draft(plan, draft, repair_chain.invoke, fixed_events=fixed)

    return interpret_chain | RunnableLambda(plan_validate_repair)

# %% mock_plan
def _mock_interpretation(user_input: str, today: date) -> PlanSpec:
    """두 시연 사례의 고정 데이터입니다. 임의 자연어를 해석하는 모델은 아닙니다."""
    if "발표" in user_input:
        deadline = today + timedelta(days=7 - today.weekday() + 4)
        return PlanSpec(
            plan_start=today, plan_end=deadline,
            goals=[
                ParsedGoal(title="발표 자료 제작", deadline=deadline, priority="high",
                           priority_reason="마감 전에 제작 완료 필요", session_minutes=120, required_minutes=360),
                ParsedGoal(title="발표 연습", deadline=deadline, priority="medium",
                           priority_reason="자료 완성 후 반복 연습 필요", session_minutes=60, required_minutes=180),
            ],
            fixed_schedules=[FixedSchedule(
                title="약속", recurrence="once", start_time="19:00", end_time="21:00",
                event_date=today + timedelta(days=(1 - today.weekday()) % 7),
            )],
            availability_rules=[AvailabilityRule(recurrence="weekends", preference="preferred")],
            constraints=["한 번에 너무 오래 하지 않기"],
        )

    exam = date(today.year, 10, 15)
    has_exam = ("10월 중순" in user_input or "10월 초" in user_input) and today <= exam
    end = min(exam if has_exam else today + timedelta(days=13), today + timedelta(days=27))
    plan = PlanSpec(plan_start=today, plan_end=end, goals=[])
    if has_exam:
        plan.warnings.append(f"'10월 초'를 {exam.isoformat()}로 해석했습니다.")
    if "부트캠프" in user_input:
        plan.fixed_schedules.append(FixedSchedule(
            title="부트캠프", recurrence="weekdays", start_time="09:00", end_time="18:00",
        ))
    if "평일 저녁" in user_input:
        plan.availability_rules.append(AvailabilityRule(
            recurrence="weekdays", start_time="18:00",
            max_minutes_per_day=120 if "2시간" in user_input else None,
        ))
        plan.warnings.append("'평일 저녁'의 시작 시각을 18:00으로 해석했습니다.")
    if "주말" in user_input and "여유" in user_input:
        plan.availability_rules.append(AvailabilityRule(recurrence="weekends", preference="preferred"))

    samples = [
        ("한국사", "한국사 시험", 720, 4, "시험일이 가까워 집중 학습이 필요함"),
        ("SKCT", "SKCT 준비", 540, 3, "채용 전형 준비가 필요함"),
    ]
    for keyword, title, minutes, sessions, reason in samples:
        if keyword in user_input:
            plan.goals.append(ParsedGoal(
                title=title, deadline=exam if keyword == "한국사" and has_exam else end,
                priority="high", priority_reason=reason, session_minutes=90,
                target_sessions_per_week=sessions, required_minutes=minutes,
            ))
    if "영단어" in user_input or "영어 단어" in user_input:
        plan.goals.append(ParsedGoal(
            title="영단어", priority="medium", priority_reason="매일 반복해야 누적 효과가 있는 학습",
            recurrence="daily", session_minutes=20 if "20분" in user_input else 30,
        ))
    return plan

# %% mock_draft
def _mock_draft(plan: PlanSpec) -> DraftSchedule:
    """예시 시간표를 날짜별 초안으로 변환합니다. 실제 모델은 호출하지 않습니다."""
    goals = {goal.title: goal for goal in plan.goals}
    if "발표 자료 제작" in goals:
        sessions = [
            (2, "19:00", "21:00", "발표 자료 제작"),
            (5, "10:00", "12:00", "발표 자료 제작"),
            (6, "14:00", "16:00", "발표 자료 제작"),
            (3, "19:00", "20:00", "발표 연습"),
            (6, "10:00", "11:00", "발표 연습"),
            (9, "19:00", "20:00", "발표 연습"),
        ]
        events = [PlannedEvent(
            title=title, goal=title, date=plan.plan_start + timedelta(days=offset),
            start_time=start, end_time=end, reason="평일은 짧게, 주말은 제작과 연습을 나누어 배치",
        ) for offset, start, end, title in sessions
            if plan.plan_start + timedelta(days=offset) <= plan.plan_end]
        return DraftSchedule(events=events, strategy_summary=
            "약속이 있는 화요일은 비우고, 주말에 발표 자료 제작을 길게 배치했습니다. 발표 연습은 제작 세션과 분리했습니다.")

    remaining = {goal.title: goal.required_minutes for goal in plan.goals if goal.required_minutes is not None}
    events = []
    for day in _days(plan.plan_start, plan.plan_end):
        if day.weekday() < 5:
            main = "한국사 시험" if day.weekday() in {0, 2, 4} else "SKCT 준비"
            slots = [(main, "19:00", 90), ("영단어", "21:00", None)]
        else:
            slots = [("한국사 시험", "10:00", 90), ("SKCT 준비", "14:00", 90), ("영단어", "20:00", None)]
        for title, start_value, minutes in slots:
            if title not in goals:
                continue
            minutes = minutes or goals[title].session_minutes or 30
            if remaining.get(title, minutes) < minutes:
                continue
            start = datetime.combine(day, time.fromisoformat(start_value))
            events.append(PlannedEvent(
                title=title, goal=title, date=day, start_time=start_value,
                end_time=(start + timedelta(minutes=minutes)).strftime("%H:%M"),
                reason="API 없는 시연을 위한 예시 배치",
            ))
            if title in remaining:
                remaining[title] -= minutes
    return DraftSchedule(events=events, strategy_summary=
        "평일에는 부트캠프 이후 주요 학습을 하나만 배치했습니다. 주말은 한국사와 SKCT 집중 세션에 활용했습니다. 영단어는 매일 짧게 유지했습니다.")

# %% entry_point
def generate_schedule(user_input: str, current_date: date | None = None) -> ScheduleResult:
    """앱과 노트북이 공유하는 진입점입니다. 실제 키가 있으면 LLM 체인을 실행합니다."""
    if not user_input.strip():
        raise ValueError("목표와 일정을 입력해 주세요.")
    today = current_date or date.today()
    if is_mock_mode():
        plan = _normalize_plan_period(_mock_interpretation(user_input, today))
        return process_draft(plan, _mock_draft(plan))
    return build_chain().invoke({
        "current_date": f"{today.isoformat()} ({today.strftime('%A')})",
        "default_plan_end": (today + timedelta(days=13)).isoformat(),
        "max_plan_end": (today + timedelta(days=27)).isoformat(),
        "user_input": user_input,
    })
