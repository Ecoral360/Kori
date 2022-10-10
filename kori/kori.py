from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias, Callable, Any, Iterator


# ############################# Kori ############################# #

@dataclass
class Kori:
    test_suite: KoriTestSuite

    def run_kori(self, folder_path: str, *, file_prefix: str) -> list[KoriTestSuiteResult]:
        files = os.listdir(folder_path)
        return [self.test_suite.run_suite(f"{folder_path}/{file}") for file in files if file.startswith(file_prefix)]


@dataclass
class KoriTestSuite:
    tests: list[KoriTest]
    before_each: list[KoriTestAction] = field(default_factory=list, kw_only=True)
    after_each: list[KoriTestAction] = field(default_factory=list, kw_only=True)

    def __post_init__(self):
        for test in self.tests:
            if test.before_each:
                test.actions = [*self.before_each, *test.actions]
            if test.after_each:
                test.actions = [*test.actions, *self.after_each]

    @staticmethod
    def _extract_team(file_path: str) -> list[str]:
        def add_space(match: re.Match): return f"{match.group()} "

        file_without_ext = file_path[:-3]
        names = file_without_ext.split("_")[1:]
        return [re.sub(r"[A-Z][a-z\-]+", add_space, name).strip() for name in names]

    def run_suite(self, file_path: str) -> KoriTestSuiteResult:
        with open(file_path, "r", encoding="utf8") as f:
            code = "\n".join(f.readlines())

        test_results = [test.run_test(code) for test in self.tests]

        return KoriTestSuiteResult(self, code, self._extract_team(file_path), test_results)


@dataclass
class KoriTestSuiteResult:
    parent: KoriTestSuite
    code: str
    team: list[str]
    test_results: list[KoriTestResult]

    def generate_result_file(self, dest_folder: str, *, file_prefix: str):
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        file_name = f"{file_prefix}_{'_'.join(member.replace(' ', '') for member in self.team)}.py"
        results = KoriResultFormatter(self).format_test_results().replace("\t", " " * 4)
        lines = [f'"""\n{results}\n"""\n', self.code.replace("\n\n", "\n")]
        with open(f"{dest_folder}/{file_name}", "w+", encoding="utf8") as f:
            f.writelines(lines)


@dataclass
class KoriTest:
    name: str
    actions: list[KoriTestAction]
    before_each: bool = field(default=True, repr=False, kw_only=True)
    after_each: bool = field(default=True, repr=False, kw_only=True)
    ignored_actions: list[KoriTestAction] = field(default_factory=list, kw_only=True)

    def _should_ignore(self, ctx: KoriTestCtx, fn: Callable, *fn_args, **fn_kwargs) -> Optional[KoriTestActionResult]:
        result = None
        for ignored_action in self.ignored_actions:
            try:
                result = ignored_action.call(ctx, fn.__name__, *fn_args, **fn_kwargs)
                if result.result_state.is_success():
                    break
                result = None
            except Exception as e:
                print(e)

        return result

    def _action_called(self, fn: Callable, ctx: KoriTestCtx):
        def action_wrapper(*fn_args, **fn_kwargs):
            if result := self._should_ignore(ctx, fn, *fn_args, **fn_kwargs):
                return result.function_result
            action: KoriTestAction = ctx.next_action()
            action_result = action.call(ctx, fn.__name__, *fn_args, **fn_kwargs)
            ctx.test_report.append(action_result)
            if isinstance(action_result.function_result, KoriEarlyExit):
                raise _KoriExitTest()
            return action_result.function_result

        return action_wrapper

    def _kori_test_context(self, iter_action, test_report: list[KoriTestActionResult]) -> KoriTestCtx:
        ctx = KoriTestCtx({}, {}, iter_action, test_report)
        for action in self.actions:
            if action.mocked_fn is not None:
                ctx.globals_[action.mocked_fn.__name__] = self._action_called(action.mocked_fn, ctx)
        return ctx

    def run_test(self, code: str) -> KoriTestResult:
        iter_action = iter(self.actions)
        test_report: list[KoriTestActionResult] = []

        ctx = self._kori_test_context(iter_action, test_report)
        try:
            exec(code, ctx.globals_, ctx.locals_)
        except _KoriExitTest as e:
            print(e)

        while (action := ctx.next_action()) is not None:
            print(f"Not called: {action.name}")

        return KoriTestResult(self, KoriTestState.combine(*[report.result_state for report in test_report]),
                              test_report)


@dataclass
class KoriTestResult:
    parent: KoriTest
    final_state: KoriTestState
    test_reports: list[KoriTestActionResult]

    @property
    def name(self):
        return self.parent.name


@dataclass
class KoriTestState:
    outcome: KoriTestOutcome = field(default="Success")
    errors: list[KoriTestError] = field(default_factory=list, kw_only=True)
    warnings: list[KoriTestWarning] = field(default_factory=list, kw_only=True)

    @classmethod
    def fail(cls, *errors: KoriTestError):
        return cls("Failure", errors=[*errors])

    @classmethod
    def warn(cls, *warnings: KoriTestWarning, outcome: KoriTestOutcome = "Success"):
        return cls(outcome, warnings=[*warnings])

    @classmethod
    def success(cls):
        return cls()

    def err_into_warn(self):
        return KoriTestState(outcome="Success", warnings=self.warnings + [
            KoriTestWarning(error.name, error.expected, error.actual) for error in self.errors
        ])

    def warn_into_err(self):
        return KoriTestState(outcome="Success", errors=self.errors + [
            KoriTestError(warning.name, warning.expected, warning.actual) for warning in self.warnings
        ])

    def is_failure(self):
        return self.outcome == "Failure"

    def is_success(self):
        return self.outcome == "Success"

    def is_complete_success(self):
        return self.is_success() and not self.has_warning()

    def has_errors(self):
        return len(self.errors) != 0

    def has_warning(self):
        return len(self.warnings) != 0

    @staticmethod
    def combine(*states: KoriTestState) -> KoriTestState:
        final = KoriTestState.success()
        for state in states:
            if final.is_failure() or state.is_failure():
                final = KoriTestState("Failure", errors=final.errors + state.errors,
                                      warnings=final.warnings + state.warnings)
            else:
                final = KoriTestState("Success", warnings=final.warnings + state.warnings)
        return final


@dataclass
class KoriTestAction:
    name: str
    action: Callable[[KoriTestCtx, ...], tuple[KoriTestState, Any]]
    mocked_fn: Optional[Callable] = None
    sub_actions: list[KoriTestSubAction] = field(default_factory=list, kw_only=True)
    action_args: list[Any] = field(default_factory=list, kw_only=True)
    on_fail: Optional[KoriTestSubAction] = None
    on_success: Optional[KoriTestSubAction] = None
    only_warns: bool = False

    def _call_sub_actions(self, ctx: KoriTestCtx, fn_name: str) -> list[KoriTestSubActionReport]:
        sub_action_results = []
        for sub_action in self.sub_actions:
            if "fn_name" in sub_action.action.__code__.co_varnames:  # type: ignore
                result_state = sub_action.action(ctx, fn_name=fn_name)  # type: ignore
            else:
                result_state = sub_action.action(ctx)
            sub_action_results.append(KoriTestSubActionReport(
                action_name=sub_action.action_name,
                result_state=result_state,
                action_args=sub_action.action_args)
            )

        return sub_action_results

    def call(self, ctx: KoriTestCtx, fn_name: str, *fn_args, **fn_kwargs) -> KoriTestActionResult:
        if self.mocked_fn is not None and self.mocked_fn.__name__ != fn_name:
            action_state, function_result = KoriTestState.fail(
                KoriTestError("<invalid call>", self.mocked_fn.__name__, fn_name)
            ), KoriEarlyExit()
        else:
            if "fn_name" in self.action.__code__.co_varnames:  # type: ignore
                action_state, function_result = self.action(ctx, *fn_args, **fn_kwargs, fn_name=fn_name)  # type: ignore
            else:
                action_state, function_result = self.action(ctx, *fn_args, **fn_kwargs)
            if self.only_warns and action_state.is_failure():
                action_state = action_state.err_into_warn()
        sub_action_results = self._call_sub_actions(ctx, fn_name)
        result_state = KoriTestState.combine(action_state,
                                             *[sub_action_result.result_state for sub_action_result
                                               in sub_action_results])

        result = KoriTestActionResult(
            function_result=function_result,
            result_state=result_state,
            action_report=KoriTestActionReport(
                action_name=self.name,
                result_state=action_state,
                action_args=self.action_args,
                fn_name=fn_name,
                fn_args=[*fn_args],
                fn_kwargs=fn_kwargs
            ),
            sub_actions_reports=sub_action_results
        )
        return result

    def warn_only(self):
        self.only_warns = True
        return self

    def also(self, *sub_actions: KoriTestSubAction):
        self.sub_actions += sub_actions
        return self


@dataclass
class KoriTestCtx:
    globals_: dict[str, Any]
    locals_: dict[str, Any]
    iter_actions: Iterator[KoriTestAction]
    test_report: list[KoriTestActionResult]
    current_action: Optional[KoriTestAction] = field(init=False, default=None)

    def next_action(self):
        try:
            next_action = next(self.iter_actions)
        except StopIteration:
            return None
        else:
            self.current_action = next_action
            return next_action


# ############################# Types ############################# #

KoriTestOutcome: TypeAlias = Literal["Success", "Failure"]


@dataclass(frozen=True, kw_only=True)
class KoriTestSubAction:
    action_name: str
    action: Callable[[KoriTestCtx], KoriTestState]
    action_args: list[Any] = field(default_factory=list, kw_only=True)


@dataclass(frozen=True, kw_only=True)
class KoriTestActionResult:
    function_result: Any
    result_state: KoriTestState
    action_report: KoriTestActionReport
    sub_actions_reports: list[KoriTestSubActionReport]


@dataclass(frozen=True, kw_only=True)
class KoriTestActionReport:
    action_name: str
    result_state: KoriTestState
    action_args: list[Any]
    fn_name: str
    fn_args: list[Any]
    fn_kwargs: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class KoriTestSubActionReport:
    action_name: str
    result_state: KoriTestState
    action_args: list[Any]


# ############################# Errors & Warnings ############################# #

class KoriEarlyExit:
    pass


class _KoriExitTest(BaseException):
    def __init__(self):
        super().__init__()


class _KoriNoMoreTests(BaseException):
    def __init__(self):
        super().__init__()


class KoriTestError(Exception):
    def __init__(self, name: str, expected: Any, actual: Any):
        super().__init__(f"Expected:{expected!r}, found:{actual!r}")
        self.expected = expected
        self.actual = actual
        self.name = name


class KoriTestWarning(Warning):
    def __init__(self, name: str, expected: Any, actual: Any):
        super().__init__(f"Expected:{expected!r}, found:{actual!r}")
        self.expected = expected
        self.actual = actual
        self.name = name


# ############################# Kori Result Formatter ############################# #

class KoriResultFormatter:
    def __init__(self, test_suite_result: KoriTestSuiteResult):
        self.test_suite_result = test_suite_result

    @staticmethod
    def _get_state_icon(state: KoriTestState):
        if state.is_failure():
            return "❌"
        elif state.has_warning():
            return "⚠"
        else:
            return "✔"

    @staticmethod
    def _format_error_or_warning(err_or_warn: KoriTestError | KoriTestWarning, prefix: str):
        icon = "❌" if isinstance(err_or_warn, KoriTestError) else "⚠"
        return f"""{prefix}-{icon}- {err_or_warn.name}
{prefix}| Expected: {err_or_warn.expected!r}
{prefix}| Actual: {err_or_warn.actual!r}"""

    @classmethod
    def _format_test_report(cls, test_report: KoriTestActionReport | KoriTestSubActionReport):
        is_sub_action = isinstance(test_report, KoriTestSubActionReport)
        state = test_report.result_state
        icon = cls._get_state_icon(state)
        indent = "\t" * (is_sub_action + 1)
        action_args = f": {', '.join(map(repr, test_report.action_args))}" if test_report.action_args else ""
        result = f"{indent}[{icon}] {test_report.action_name}{action_args}"

        if state.is_complete_success():
            return result + "\n"

        err_indent = "\t" * (is_sub_action + 2)  # 2 tabs away for actions and 3 tabs away for sub_actions
        if state.has_warning():
            result += "\n" + "\n".join(cls._format_error_or_warning(warning, err_indent) for warning in state.warnings)
        if state.has_errors():
            result += "\n" + "\n".join(cls._format_error_or_warning(err, err_indent) for err in state.errors)
        return result + f"\n{err_indent}--\n"

    def _format_test_reports(self, test_result: KoriTestResult):
        name = test_result.name
        test_reports = test_result.test_reports
        final_result = KoriTestState.combine(*[report.result_state for report in test_reports])
        formatted_result = f"{name}: {final_result.outcome}\n"
        for report in test_reports:
            action_formatted = self._format_test_report(report.action_report)
            sub_action_formatted = "".join(map(self._format_test_report, report.sub_actions_reports))
            formatted_result += action_formatted + sub_action_formatted
        return formatted_result

    def format_test_results(self):
        return "\n".join(map(self._format_test_reports, self.test_suite_result.test_results))
