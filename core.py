import json
import os
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


load_dotenv()

Recurrence = Literal["daily", "weekdays", "weekends", "flexible", "once"]
FixedRecurrence = Literal["daily", "weekdays", "weekends", "once"]
Priority = Literal["high", "medium", "low"]
TIME_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
END_TIME_PATTERN = r"^(?:(?:[01]\d|2[0-3]):[0-5]\d|24:00)$"


class CalendarEvent(BaseModel):
    title: str
    start: datetime
    end: datetime
    kind: Literal["fixed", "generated"]
    goal: str | None = None
    reason: str | None = None


class ParsedGoal(BaseModel):
    title: str
    deadline: date | None = None
    priority: Priority
    priority_reason: str
    recurrence: Recurrence = "flexible"
    session_minutes: int | None = Field(default=None, ge=1)
    target_sessions_per_week: int | None = Field(default=None, ge=1, le=14)
    required_minutes: int | None = Field(default=None, ge=1)
    constraints: list[str] = Field(default_factory=list)


class FixedSchedule(BaseModel):
    title: str
    recurrence: FixedRecurrence
    start_time: str = Field(pattern=TIME_PATTERN)
    end_time: str = Field(pattern=END_TIME_PATTERN)
    event_date: date | None = None


class LifestylePreferences(BaseModel):
    active_start: str | None = Field(default=None, pattern=TIME_PATTERN)
    active_end: str | None = Field(default=None, pattern=TIME_PATTERN)
    lunch_start: str | None = Field(default=None, pattern=TIME_PATTERN)
    lunch_end: str | None = Field(default=None, pattern=TIME_PATTERN)
    dinner_start: str | None = Field(default=None, pattern=TIME_PATTERN)
    dinner_end: str | None = Field(default=None, pattern=TIME_PATTERN)
    wind_down_start: str | None = Field(default=None, pattern=TIME_PATTERN)


class AvailabilityRule(BaseModel):
    recurrence: Literal["daily", "weekdays", "weekends"]
    start_time: str | None = Field(default=None, pattern=TIME_PATTERN)
    end_time: str | None = Field(default=None, pattern=END_TIME_PATTERN)
    max_minutes_per_day: int | None = Field(default=None, ge=1)
    preference: Literal["preferred", "normal", "avoid"] = "normal"


class PlanSpec(BaseModel):
    plan_start: date
    plan_end: date
    goals: list[ParsedGoal]
    fixed_schedules: list[FixedSchedule] = Field(default_factory=list)
    lifestyle: LifestylePreferences = Field(default_factory=LifestylePreferences)
    availability_rules: list[AvailabilityRule] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PlannedEvent(BaseModel):
    title: str
    date: date
    start_time: str = Field(pattern=TIME_PATTERN)
    end_time: str = Field(pattern=END_TIME_PATTERN)
    goal: str
    reason: str | None = None


class DraftSchedule(BaseModel):
    events: list[PlannedEvent]
    strategy_summary: str = Field(description="2~4개의 짧은 핵심 문장으로 작성한 계획 전략")
    warnings: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    code: str
    message: str
    event_index: int | None = None


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class GoalSummary(BaseModel):
    title: str
    priority: Priority
    priority_reason: str
    deadline: date | None = None
    required_hours: float | None = Field(default=None, ge=0)
    allocated_hours: float = Field(default=0, ge=0)
    shortage_hours: float = Field(default=0, ge=0)


class ScheduleResult(BaseModel):
    plan_start: date
    plan_end: date
    events: list[CalendarEvent]
    goals: list[GoalSummary] = Field(default_factory=list)
    plan_spec: PlanSpec
    strategy_summary: str | None = None
    warning: str | None = None


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
- 사용자가 총 필요 시간을 명시했다면 required_minutes에 반영한다.
- 총 시간이 없는 목표는 회당 학습 시간이나 주당 횟수에 대한 reasonable hint를 제공할 수 있다. 근거 없이 정밀한 총시간을 지어내지 않는다.
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


def _model():
    provider = os.getenv("MODEL_PROVIDER", "google_genai")
    if provider == "openai":
        default_model = "gpt-4.1-mini"
    elif provider == "google_genai":
        default_model = "gemini-3.6-flash"
    else:
        raise ValueError(f"지원하지 않는 MODEL_PROVIDER입니다: {provider}")
    return init_chat_model(
        os.getenv("MODEL_NAME", default_model),
        model_provider=provider,
        temperature=0,
    )


def _matches(day: date, recurrence: Recurrence, event_date: date | None) -> bool:
    if recurrence == "daily":
        return True
    if recurrence == "weekdays":
        return day.weekday() < 5
    if recurrence == "weekends":
        return day.weekday() >= 5
    if recurrence == "once":
        return day == event_date
    return False


def _event_bounds(day: date, start_value: str, end_value: str) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.fromisoformat(start_value))
    if end_value == "24:00":
        end = datetime.combine(day + timedelta(days=1), time())
    else:
        end = datetime.combine(day, time.fromisoformat(end_value))
        if end <= start:
            end += timedelta(days=1)
    return start, end


def _planned_bounds(event: PlannedEvent) -> tuple[datetime, datetime]:
    start = datetime.combine(event.date, time.fromisoformat(event.start_time))
    end = (
        datetime.combine(event.date + timedelta(days=1), time())
        if event.end_time == "24:00"
        else datetime.combine(event.date, time.fromisoformat(event.end_time))
    )
    return start, end


def _normalize_plan_period(plan: PlanSpec) -> PlanSpec:
    warnings = list(plan.warnings)
    maximum_end = plan.plan_start + timedelta(days=27)
    plan_end = plan.plan_end
    if plan_end < plan.plan_start:
        plan_end = plan.plan_start + timedelta(days=13)
        warnings.append("계획 종료일이 시작일보다 빨라 기본 14일로 조정했습니다.")
    if plan_end > maximum_end:
        plan_end = maximum_end
        warnings.append(f"상세 일정은 {maximum_end.isoformat()}까지 최대 4주만 배치합니다.")
    return plan.model_copy(update={"plan_end": plan_end, "warnings": list(dict.fromkeys(warnings))})


def _expand_fixed_schedules(plan: PlanSpec) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    total_days = (plan.plan_end - plan.plan_start).days + 1
    for schedule in plan.fixed_schedules:
        for offset in range(total_days):
            day = plan.plan_start + timedelta(days=offset)
            if not _matches(day, schedule.recurrence, schedule.event_date):
                continue
            start, end = _event_bounds(day, schedule.start_time, schedule.end_time)
            events.append(CalendarEvent(title=schedule.title, start=start, end=end, kind="fixed"))
    return sorted(events, key=lambda event: event.start)


def _expected_dates(goal: ParsedGoal, plan: PlanSpec) -> list[date]:
    days = [plan.plan_start + timedelta(days=offset) for offset in range((plan.plan_end - plan.plan_start).days + 1)]
    if goal.deadline:
        days = [day for day in days if day <= goal.deadline]
    if goal.recurrence == "daily":
        return days
    if goal.recurrence == "weekdays":
        return [day for day in days if day.weekday() < 5]
    if goal.recurrence == "weekends":
        return [day for day in days if day.weekday() >= 5]
    if goal.recurrence == "once":
        return [min(max(goal.deadline or plan.plan_end, plan.plan_start), plan.plan_end)]
    return []


def _overlaps(event: CalendarEvent, other: CalendarEvent) -> bool:
    return event.start < other.end and event.end > other.start


def _clock_minutes(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _within_active_hours(event: CalendarEvent, lifestyle: LifestylePreferences) -> bool:
    if not lifestyle.active_start and not lifestyle.active_end:
        return True
    start_limit = _clock_minutes(lifestyle.active_start or "00:00")
    end_limit = _clock_minutes(lifestyle.active_end or "23:59")
    event_start = event.start.hour * 60 + event.start.minute
    event_end = event.end.hour * 60 + event.end.minute
    if event.end.date() > event.start.date():
        event_end += 24 * 60
    if end_limit <= start_limit:
        end_limit += 24 * 60
        if event_start < start_limit:
            event_start += 24 * 60
            event_end += 24 * 60
    return event_start >= start_limit and event_end <= end_limit


def _explicit_unavailable_blocks(day: date, lifestyle: LifestylePreferences) -> list[tuple[datetime, datetime]]:
    blocks: list[tuple[datetime, datetime]] = []
    for start_value, end_value in (
        (lifestyle.lunch_start, lifestyle.lunch_end),
        (lifestyle.dinner_start, lifestyle.dinner_end),
        (lifestyle.wind_down_start, lifestyle.active_end),
    ):
        if start_value and end_value:
            blocks.append(_event_bounds(day, start_value, end_value))
    return blocks


def _convert_planned_event(event: PlannedEvent) -> CalendarEvent:
    start, end = _planned_bounds(event)
    return CalendarEvent(title=event.title, start=start, end=end, kind="generated", goal=event.goal, reason=event.reason)


def _availability_applies(day: date, recurrence: Literal["daily", "weekdays", "weekends"]) -> bool:
    if recurrence == "daily":
        return True
    if recurrence == "weekdays":
        return day.weekday() < 5
    return day.weekday() >= 5


def validate_draft(plan: PlanSpec, draft: DraftSchedule, fixed_events: list[CalendarEvent] | None = None) -> ValidationResult:
    fixed_events = fixed_events if fixed_events is not None else _expand_fixed_schedules(plan)
    goal_map = {goal.title: goal for goal in plan.goals}
    issues: list[ValidationIssue] = []
    converted: list[tuple[int, CalendarEvent]] = []
    invalid_indices: set[int] = set()

    def add(code: str, message: str, index: int | None = None) -> None:
        issues.append(ValidationIssue(code=code, message=message, event_index=index))
        if index is not None:
            invalid_indices.add(index)

    for index, planned in enumerate(draft.events):
        event = _convert_planned_event(planned)
        converted.append((index, event))
        if event.end <= event.start:
            add("INVALID_TIME", f"'{planned.title}'의 종료 시각이 시작 시각보다 늦지 않습니다.", index)
        if not plan.plan_start <= planned.date <= plan.plan_end or event.end.date() > plan.plan_end + timedelta(days=1):
            add("OUTSIDE_PLAN", f"'{planned.title}'이 계획 기간 밖에 있습니다.", index)

        goal = goal_map.get(planned.goal)
        if goal is None:
            add("UNKNOWN_GOAL", f"'{planned.goal}'은 PlanSpec에 없는 목표입니다.", index)
            continue
        if goal.deadline and planned.date > goal.deadline:
            add("AFTER_DEADLINE", f"'{planned.goal}' 일정이 마감일 이후에 있습니다.", index)
        if goal.recurrence == "weekdays" and planned.date.weekday() >= 5:
            add("RECURRENCE_DAY", f"'{planned.goal}'은 평일 목표이지만 주말에 배치됐습니다.", index)
        if goal.recurrence == "weekends" and planned.date.weekday() < 5:
            add("RECURRENCE_DAY", f"'{planned.goal}'은 주말 목표이지만 평일에 배치됐습니다.", index)

        duration = int((event.end - event.start).total_seconds() // 60)
        if goal.recurrence != "flexible" and goal.session_minutes:
            tolerance = max(10, round(goal.session_minutes * 0.2))
            if abs(duration - goal.session_minutes) > tolerance:
                add("SESSION_DURATION", f"'{planned.goal}'의 회당 시간이 {goal.session_minutes}분과 크게 다릅니다.", index)
        if not _within_active_hours(event, plan.lifestyle):
            add("USER_TIME_CONSTRAINT", f"'{planned.title}'이 사용자의 활동 가능 시간을 벗어났습니다.", index)
        if any(event.start < block_end and event.end > block_start for block_start, block_end in _explicit_unavailable_blocks(planned.date, plan.lifestyle)):
            add("USER_TIME_CONSTRAINT", f"'{planned.title}'이 사용자가 명시한 생활 시간과 겹칩니다.", index)
        if any(_overlaps(event, fixed) for fixed in fixed_events):
            add("FIXED_OVERLAP", f"'{planned.title}'이 고정 일정과 겹칩니다.", index)

        event_start = event.start.hour * 60 + event.start.minute
        event_end = event.end.hour * 60 + event.end.minute
        if event.end.date() > event.start.date():
            event_end += 24 * 60
        for rule in plan.availability_rules:
            if not _availability_applies(planned.date, rule.recurrence):
                continue
            label = {"daily": "매일", "weekdays": "평일", "weekends": "주말"}[rule.recurrence]
            if rule.start_time and event_start < _clock_minutes(rule.start_time):
                add("AVAILABILITY_TIME", f"'{planned.title}'이 {label} 가능 시작 시각 {rule.start_time} 이전에 배치됐습니다.", index)
            if rule.end_time and event_end > _clock_minutes(rule.end_time):
                add("AVAILABILITY_TIME", f"'{planned.title}'이 {label} 가능 종료 시각 {rule.end_time} 이후에 배치됐습니다.", index)

    for rule in plan.availability_rules:
        if rule.max_minutes_per_day is None:
            continue
        daily_total: dict[date, int] = {}
        for index, event in sorted(converted, key=lambda item: item[1].start):
            day = event.start.date()
            if not _availability_applies(day, rule.recurrence):
                continue
            duration = int((event.end - event.start).total_seconds() // 60)
            daily_total[day] = daily_total.get(day, 0) + duration
            if daily_total[day] > rule.max_minutes_per_day:
                label = {"daily": "매일", "weekdays": "평일", "weekends": "주말"}[rule.recurrence]
                add(
                    "DAILY_LIMIT_EXCEEDED",
                    f"{day.isoformat()} {label} generated 일정이 하루 최대 {rule.max_minutes_per_day}분을 초과했습니다. (총 {daily_total[day]}분)",
                    index,
                )

    valid_for_overlap = [(index, event) for index, event in converted if index not in invalid_indices]
    valid_for_overlap.sort(key=lambda item: item[1].start)
    for position, (index, event) in enumerate(valid_for_overlap):
        for _, previous in valid_for_overlap[:position]:
            if _overlaps(event, previous):
                add("GENERATED_OVERLAP", f"'{previous.title}'과 '{event.title}'이 겹칩니다.", index)
                break

    valid_dates_by_goal: dict[str, set[date]] = {}
    for index, event in converted:
        if index not in invalid_indices and event.goal:
            valid_dates_by_goal.setdefault(event.goal, set()).add(event.start.date())

    for goal in plan.goals:
        if goal.recurrence not in {"daily", "weekdays", "weekends"}:
            continue
        missing = sorted(set(_expected_dates(goal, plan)) - valid_dates_by_goal.get(goal.title, set()))
        if missing:
            dates = ", ".join(day.strftime("%m-%d") for day in missing)
            add("RECURRENCE_MISSING", f"'{goal.title}' 반복 일정이 누락된 날짜: {dates}")

    return ValidationResult(valid=not issues, issues=issues)


def _valid_generated_events(draft: DraftSchedule, validation: ValidationResult) -> list[CalendarEvent]:
    invalid_indices = {issue.event_index for issue in validation.issues if issue.event_index is not None}
    events = [_convert_planned_event(event) for index, event in enumerate(draft.events) if index not in invalid_indices]
    return sorted(events, key=lambda event: event.start)


def _required_minutes(goal: ParsedGoal, plan: PlanSpec) -> int | None:
    if goal.required_minutes is not None:
        return goal.required_minutes
    if goal.recurrence != "flexible" and goal.session_minutes:
        return len(_expected_dates(goal, plan)) * goal.session_minutes
    return None


def _build_schedule_result(plan: PlanSpec, draft: DraftSchedule, validation: ValidationResult, fixed_events: list[CalendarEvent]) -> ScheduleResult:
    generated_events = _valid_generated_events(draft, validation)
    events = sorted(fixed_events + generated_events, key=lambda event: event.start)
    warnings = list(draft.warnings)
    if not validation.valid:
        warnings.extend(f"자동 수정 후 미해결: {issue.message}" for issue in validation.issues)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    summaries: list[GoalSummary] = []
    for goal in plan.goals:
        allocated_minutes = sum(int((event.end - event.start).total_seconds() // 60) for event in generated_events if event.goal == goal.title)
        required_minutes = _required_minutes(goal, plan)
        shortage_minutes = max(required_minutes - allocated_minutes, 0) if required_minutes is not None else 0
        if shortage_minutes:
            warnings.append(f"'{goal.title}'은 {shortage_minutes / 60:g}시간이 부족합니다.")
        summaries.append(GoalSummary(
            title=goal.title,
            priority=goal.priority,
            priority_reason=goal.priority_reason,
            deadline=goal.deadline,
            required_hours=round(required_minutes / 60, 2) if required_minutes is not None else None,
            allocated_hours=round(allocated_minutes / 60, 2),
            shortage_hours=round(shortage_minutes / 60, 2),
        ))

    return ScheduleResult(
        plan_start=plan.plan_start,
        plan_end=plan.plan_end,
        events=events,
        goals=sorted(summaries, key=lambda goal: priority_order[goal.priority]),
        plan_spec=plan,
        strategy_summary=draft.strategy_summary,
        warning=" ".join(dict.fromkeys(warnings)) or None,
    )


RepairFunction = Callable[[PlanSpec, list[CalendarEvent], DraftSchedule, ValidationResult], DraftSchedule]


def process_draft(plan: PlanSpec, draft: DraftSchedule, repair: RepairFunction | None = None) -> ScheduleResult:
    plan = _normalize_plan_period(plan)
    fixed_events = _expand_fixed_schedules(plan)
    validation = validate_draft(plan, draft, fixed_events)
    final_draft = draft
    if not validation.valid and repair is not None:
        final_draft = repair(plan, fixed_events, draft, validation)
        validation = validate_draft(plan, final_draft, fixed_events)
    return _build_schedule_result(plan, final_draft, validation, fixed_events)


def _json(value: BaseModel | list[BaseModel]) -> str:
    data = [item.model_dump(mode="json") for item in value] if isinstance(value, list) else value.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False)


def build_chain():
    llm = _model()
    interpret_chain = INTERPRET_PROMPT | llm.with_structured_output(PlanSpec)
    planner_chain = PLAN_PROMPT | llm.with_structured_output(DraftSchedule)
    repair_chain = REPAIR_PROMPT | llm.with_structured_output(DraftSchedule)

    def plan_validate_repair(plan: PlanSpec) -> ScheduleResult:
        plan = _normalize_plan_period(plan)
        fixed_events = _expand_fixed_schedules(plan)
        draft = planner_chain.invoke({"plan_spec_json": _json(plan), "fixed_events_json": _json(fixed_events)})

        def repair(current_plan: PlanSpec, current_fixed: list[CalendarEvent], current_draft: DraftSchedule, validation: ValidationResult) -> DraftSchedule:
            return repair_chain.invoke({
                "plan_spec_json": _json(current_plan),
                "fixed_events_json": _json(current_fixed),
                "draft_json": _json(current_draft),
                "issues_json": _json(validation.issues),
            })

        return process_draft(plan, draft, repair)

    return interpret_chain | RunnableLambda(plan_validate_repair)


def is_mock_mode() -> bool:
    provider = os.getenv("MODEL_PROVIDER", "google_genai")
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
    elif provider == "google_genai":
        api_key = os.getenv("GOOGLE_API_KEY", "")
    else:
        return True
    return not api_key or api_key == "fake-key"


def _mock_interpretation(user_input: str, today: date) -> PlanSpec:
    october_fifth = date(today.year, 10, 5)
    has_exam_date = "10월 초" in user_input and today <= october_fifth
    plan_end = october_fifth if has_exam_date else today + timedelta(days=13)
    warnings = [f"'10월 초'를 {october_fifth.isoformat()}로 해석했습니다."] if has_exam_date else []
    availability_rules: list[AvailabilityRule] = []
    if "평일 저녁" in user_input:
        max_minutes = 120 if "2시간" in user_input else None
        availability_rules.append(AvailabilityRule(recurrence="weekdays", start_time="18:00", max_minutes_per_day=max_minutes))
        warnings.append("'평일 저녁'의 시작 시각을 18:00으로 해석했습니다.")
    if "주말" in user_input and ("여유" in user_input or "여유로워" in user_input):
        availability_rules.append(AvailabilityRule(recurrence="weekends", preference="preferred"))
    fixed_schedules = [FixedSchedule(title="부트캠프", recurrence="weekdays", start_time="09:00", end_time="18:00")] if "부트캠프" in user_input else []
    goals: list[ParsedGoal] = []
    if "한국사" in user_input:
        goals.append(ParsedGoal(title="한국사 시험", deadline=october_fifth if has_exam_date else plan_end, priority="high", priority_reason="시험일이 가까워 집중 학습이 필요함", recurrence="flexible", session_minutes=90, target_sessions_per_week=4, required_minutes=720))
    if "SKCT" in user_input:
        goals.append(ParsedGoal(title="SKCT 준비", deadline=plan_end, priority="high", priority_reason="채용 전형 준비가 필요함", recurrence="flexible", session_minutes=90, target_sessions_per_week=3, required_minutes=540))
    if "영단어" in user_input or "영어 단어" in user_input:
        goals.append(ParsedGoal(title="영단어", priority="medium", priority_reason="매일 반복해야 누적 효과가 있는 학습", recurrence="daily", session_minutes=30 if "30분" in user_input else 20))
    return PlanSpec(plan_start=today, plan_end=min(plan_end, today + timedelta(days=27)), goals=goals, fixed_schedules=fixed_schedules, availability_rules=availability_rules, warnings=warnings)


def _mock_draft(plan: PlanSpec) -> DraftSchedule:
    goal_titles = {goal.title for goal in plan.goals}
    remaining_minutes = {
        goal.title: goal.required_minutes
        for goal in plan.goals
        if goal.required_minutes is not None
    }
    events: list[PlannedEvent] = []
    for offset in range((plan.plan_end - plan.plan_start).days + 1):
        day = plan.plan_start + timedelta(days=offset)
        if day.weekday() < 5:
            main_goal = "한국사 시험" if day.weekday() in {0, 2, 4} else "SKCT 준비"
            if main_goal in goal_titles and remaining_minutes.get(main_goal, 90) >= 90:
                events.append(PlannedEvent(title=main_goal, date=day, start_time="19:00", end_time="20:30", goal=main_goal, reason="부트캠프 이후 식사와 휴식 시간을 둔 저녁 집중 세션"))
                if main_goal in remaining_minutes:
                    remaining_minutes[main_goal] -= 90
            if "영단어" in goal_titles:
                events.append(PlannedEvent(title="영단어", date=day, start_time="21:00", end_time="21:30", goal="영단어", reason="주요 공부 후 유지하는 짧은 습관"))
        else:
            if "한국사 시험" in goal_titles and remaining_minutes.get("한국사 시험", 90) >= 90:
                events.append(PlannedEvent(title="한국사 시험", date=day, start_time="10:00", end_time="11:30", goal="한국사 시험", reason="주말 오전 집중 시간 활용"))
                if "한국사 시험" in remaining_minutes:
                    remaining_minutes["한국사 시험"] -= 90
            if "SKCT 준비" in goal_titles and remaining_minutes.get("SKCT 준비", 90) >= 90:
                events.append(PlannedEvent(title="SKCT 준비", date=day, start_time="14:00", end_time="15:30", goal="SKCT 준비", reason="주말 오후 집중 세션 활용"))
                if "SKCT 준비" in remaining_minutes:
                    remaining_minutes["SKCT 준비"] -= 90
            if "영단어" in goal_titles:
                events.append(PlannedEvent(title="영단어", date=day, start_time="20:00", end_time="20:30", goal="영단어", reason="주말에도 짧은 반복 습관 유지"))
    return DraftSchedule(events=events, strategy_summary="평일에는 부트캠프 이후 주요 학습을 하나만 배치했습니다. 주말은 한국사와 SKCT 집중 세션에 활용했습니다. 영단어는 매일 짧게 유지했습니다.")


def generate_schedule(user_input: str, current_date: date | None = None) -> ScheduleResult:
    today = current_date or date.today()
    inputs = {
        "current_date": f"{today.isoformat()} ({today.strftime('%A')})",
        "default_plan_end": (today + timedelta(days=13)).isoformat(),
        "max_plan_end": (today + timedelta(days=27)).isoformat(),
        "user_input": user_input,
    }
    if is_mock_mode():
        plan = _normalize_plan_period(_mock_interpretation(user_input, today))
        return process_draft(plan, _mock_draft(plan))
    return build_chain().invoke(inputs)


def to_calendar_events(result: ScheduleResult | None) -> list[dict]:
    if result is None:
        return []
    colors = {"fixed": "#64748b", "generated": "#6d5dfc"}
    return [{
        "title": event.title,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "backgroundColor": colors[event.kind],
        "borderColor": colors[event.kind],
    } for event in result.events]
