from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias, Callable, Any, Iterator
import importlib


# ############################# Kori ############################# #

@dataclass
class Kori:
    test_suite: KoriTestSuite

    def run_test_kori(self, folder_path: str, *, file_prefix: str) -> list[KoriTestSuiteResult]:
        files = os.listdir(folder_path)
        return [self.test_suite.run_suite(f"{folder_path}/{file}") for file in files if file.startswith(file_prefix)]


@dataclass(kw_only=True)
class KoriTestConfig:
    before: list[KoriTestAction] = field(default_factory=list)
    after: list[KoriTestAction] = field(default_factory=list)
    capture_stdout: bool = field(default=False)
    ignored_actions: list[KoriTestAction] = field(default_factory=list)
    default_state_on_fail: KoriStateOnFail = field(default="Warning")
    actions_state_on_fail: dict[str, KoriStateOnFail] = field(default_factory=dict)


@dataclass
class KoriTestSuite:
    config: KoriTestConfig = KoriTestConfig()
    tests: list[KoriTest | KoriTestGroup] = field(default_factory=list)

    def __post_init__(self):
        for test in [t for t in self.tests if t.config is None]:
            test.config = self.config

    @staticmethod
    def _extract_team(file_path: str) -> list[str]:
        def add_space(match: re.Match): return f"{match.group()} "

        file_without_ext = file_path[:-3]
        names = file_without_ext.split("_")[1:]
        return [re.sub(r"[A-Z][a-z\-]+", add_space, name).strip() for name in names]

    def run_suite(self, file_path: str) -> KoriTestSuiteResult:
        with open(file_path, "r", encoding="utf8") as f:
            code = "\n".join(f.readlines())

        test_results = [test.run_test(code, file_path) for test in self.tests]

        return KoriTestSuiteResult(self, code, self._extract_team(file_path), test_results)


@dataclass
class KoriTestSuiteResult:
    parent: KoriTestSuite
    code: str
    team: list[str]
    test_results: list[KoriTestResult | KoriTestGroupResult]

    def generate_result_file(self, dest_folder: str, *, file_prefix: str):
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        file_name = f"{file_prefix}_{'_'.join(member.replace(' ', '') for member in self.team)}.py"
        results = KoriResultFormatter(self).format_test_results().replace("\t", " " * 4)
        lines = [f'""""""\nr"""\n{results}\n"""\n', self.code.replace("\n\n", "\n")]
        with open(f"{dest_folder}/{file_name}", "w+", encoding="utf8") as f:
            f.writelines(lines)


@dataclass
class KoriTestGroup:
    name: str
    tests: list[KoriTest]
    _config: KoriTestConfig = field(default=None, kw_only=True)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: KoriTestConfig):
        self._config = config
        for test in [t for t in self.tests if t.config is None]:
            test.config = self.config

    def run_test(self, code: str, file_path: str) -> KoriTestGroupResult:
        test_results = [test.run_test(code, file_path) for test in self.tests]
        final_state = KoriTestState.combine(*[test_result.final_state for test_result in test_results])
        return KoriTestGroupResult(self, final_state, test_results)


@dataclass
class KoriTestGroupResult:
    parent: KoriTestGroup
    final_state: KoriTestState
    test_results: list[KoriTestResult]

    @property
    def name(self):
        return self.parent.name


@dataclass
class KoriTest:
    name: str
    actions: list[KoriTestAction | list]
    _config: KoriTestConfig = field(default=None, kw_only=True)
    mocked_modules: list[str] = field(default_factory=list, kw_only=True)

    def __post_init__(self):
        self.actions = flatten(self.actions)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: KoriTestConfig):
        self._config = config
        default = self.config.default_state_on_fail
        actions_states = self.config.actions_state_on_fail
        for action in [a for a in self.actions if a.state_on_fail is None]:
            action.state_on_fail = actions_states.get(action.action_name, default)

            for sub_action in [sub_a for sub_a in action.sub_actions if sub_a.state_on_fail is None]:
                sub_action.state_on_fail = actions_states.get(sub_action.action_name, default)

    def _should_ignore(self, ctx: KoriTestCtx, fn: Callable, *fn_args, **fn_kwargs) -> Optional[KoriTestActionReport]:
        result = None
        for ignored_action in self.config.ignored_actions:
            try:
                result = ignored_action.call(ctx, fn.__name__, *fn_args, **fn_kwargs)
                if result.result_state.is_complete_success():
                    break
                result = None
            except Exception as e:
                print(e)

        return result

    def _action_called(self, fn: Callable, ctx: KoriTestCtx):
        def action_wrapper(*fn_args, **fn_kwargs):
            ctx.current_fn = fn

            if result := self._should_ignore(ctx, fn, *fn_args, **fn_kwargs):
                return result.action_result.function_result

            action = ctx.next_action()

            action_report = action.call(ctx, fn.__name__, *fn_args, **fn_kwargs)
            ctx.test_report.append(action_report)

            if action_report.action_result.redo_with_next_action:
                action = ctx.next_action(force_next=True)
                action_report = action.call(ctx, fn.__name__, *fn_args, **fn_kwargs)
                ctx.test_report.append(action_report)

            if action_report.action_result.end_test:
                raise _KoriExitTest()
            return action_report.action_result.function_result

        return action_wrapper

    def _kori_test_context(self, iter_action, test_report: list[KoriTestActionReport]) -> KoriTestCtx:
        ctx = KoriTestCtx({}, {}, self, iter_action, test_report)
        actions_with_mock = (action for action in [act for act in self.actions if act.mocked_fn is not None])
        mocked_functions = {mocked_fn for action in actions_with_mock for mocked_fn in action.mocked_fn}
        for fn in mocked_functions:
            ctx.globals_[fn.__name__] = self._action_called(fn, ctx)
        ctx.locals_ = ctx.globals_
        return ctx

    def _mock_modules(self, code: str, ctx: KoriTestCtx):
        action_called = self._action_called

        for mocked_module in self.mocked_modules:
            real_module = importlib.import_module(mocked_module)

            # import {module_name} [as alias]
            module_name = mocked_module
            alias = re.search(fr"import {mocked_module} *as *(\w+) *(#.*?)?\n", code)
            if alias is not None:
                module_name = alias.group(1).strip()

            code = re.sub(f"import {mocked_module}.*?\n", "", code)

            class module:
                def __getattr__(self, item):
                    return action_called(getattr(real_module, item), ctx)

            module.__name__ = module_name

            ctx.globals_[module_name] = module()

            # from {module_name} import {* | functions}
            functions = re.search(f"from {mocked_module} import (.*?)(#.*?)?\n", code)
            code = re.sub(f"from {mocked_module} import (.*?)(#.*?)?\n", "", code)
            if functions is None:
                continue
            fn_names = functions.group(1).split(",")
            for fn_name in fn_names:
                ctx.globals_[fn_name] = action_called(getattr(real_module, fn_name), ctx)

        return code

    def run_test(self, code: str, file_path: str) -> KoriTestResult:
        iter_action = iter(self.actions)
        test_report: list[KoriTestActionReport] = []

        ctx = self._kori_test_context(iter_action, test_report)

        code = self._mock_modules(code, ctx)
        err = None
        not_called = []
        try:
            exec(compile(code, file_path, mode="exec"), ctx.globals_, ctx.locals_)
        except _KoriExitTest:
            # print(e)
            pass
        except BaseException as e:
            print(f"Execption in {file_path}: \n{e}", file=sys.stderr)
            err = _KoriPythonError(f"{e.__class__.__name__}: \n{e}")
        finally:
            while (action := ctx.next_action(force_next=True)) is not None:
                not_called.append(action)

        state = KoriTestState.combine(*[report.result_state for report in test_report])
        if err is not None:
            state.outcome = "Failure"
            state.errors.append(KoriTestError("Python error", "-", str(err)))
        return KoriTestResult(self, state,
                              test_report, not_called=not_called, error=err)


@dataclass
class KoriTestResult:
    parent: KoriTest
    final_state: KoriTestState
    test_reports: list[KoriTestActionReport]
    not_called: list[KoriTestAction]
    error: _KoriPythonError | None = field(default=None, kw_only=True)

    @property
    def name(self):
        return self.parent.name


@dataclass
class KoriTestState:
    outcome: KoriTestOutcome = field(default="Success")
    errors: list[KoriTestError] = field(default_factory=list, kw_only=True)
    warnings: list[KoriTestWarning] = field(default_factory=list, kw_only=True)

    @classmethod
    def fail(cls, *errors: KoriTestFail):
        return cls("Failure", errors=[*errors])

    @classmethod
    def err(cls, *errors: KoriTestError):
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
        if self.has_warning():
            return KoriTestState(outcome="Failure", errors=self.errors + [
                KoriTestError(warning.name, warning.expected, warning.actual) for warning in self.warnings
            ])
        return self

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
    action_name: str
    action: Callable[[KoriTestCtx, ...], KoriTestActionResult]
    mocked_fn: Optional[list[Callable]] = None
    sub_actions: list[KoriTestSubAction] = field(default_factory=list, kw_only=True)
    action_args: list[Any] = field(default_factory=list, kw_only=True)
    on_fail: Optional[KoriTestSubAction] = None
    on_success: Optional[KoriTestSubAction] = None
    state_on_fail: KoriStateOnFail = field(default=None, init=False)

    def _call_sub_actions(self, ctx: KoriTestCtx) -> list[KoriTestSubActionReport]:
        sub_action_results = []
        for sub_action in self.sub_actions:
            result_state = sub_action.action(ctx)
            if sub_action.state_on_fail == "Warning":
                result_state = result_state.err_into_warn()
            else:
                result_state = result_state.warn_into_err()
            sub_action_results.append(KoriTestSubActionReport(
                action_name=sub_action.action_name,
                result_state=result_state,
                action_args=sub_action.action_args)
            )

        return sub_action_results

    def mocked_fn_names(self):
        if self.mocked_fn is None:
            return []
        return [mocked_fn.__name__ for mocked_fn in self.mocked_fn]

    def call(self, ctx: KoriTestCtx, fn_name: str, *fn_args, **fn_kwargs) -> KoriTestActionReport:
        if not self.mocks_fn(fn_name):
            action_result = KoriTestActionResult(
                KoriTestState.err(KoriTestError("<invalid call>", self.mocked_fn_names(),
                                                f"{fn_name}({', '.join(map(repr, fn_args))})")),
                end_test=True
            )
        else:
            # if the function has an arg named "fn_name", we pass the name of the function called as an argument
            action_result = self.action(ctx, *fn_args, **fn_kwargs)

            if self.state_on_fail == "Warning":
                action_result.result_state = action_result.result_state.err_into_warn()
            else:
                action_result.result_state = action_result.result_state.warn_into_err()
        if action_result.is_done:
            sub_action_results = self._call_sub_actions(ctx)
        else:
            sub_action_results = []
        result_state = KoriTestState.combine(action_result.result_state,
                                             *[sub_action_result.result_state for sub_action_result
                                               in sub_action_results])

        result = KoriTestActionReport(
            action_name=self.action_name,
            result_state=result_state,
            action_result=action_result,
            action_args=self.action_args,
            fn_name=fn_name,
            fn_args=[*fn_args],
            fn_kwargs=fn_kwargs,
            sub_actions_reports=sub_action_results
        )
        return result

    def mocks_fn(self, fn_name: str):
        return self.mocked_fn is None or fn_name in self.mocked_fn_names()

    def warn_on_fail(self):
        self.state_on_fail = "Warning"
        return self

    def err_on_fail(self):
        self.state_on_fail = "Error"
        return self

    def also(self, *sub_actions: KoriTestSubAction):
        self.sub_actions += sub_actions
        return self


@dataclass
class KoriTestCtx:
    globals_: dict[str, Any] = field(repr=False)
    locals_: dict[str, Any] = field(repr=False)
    _current_test: KoriTest
    iter_actions: Iterator[KoriTestAction] = field(repr=False)
    test_report: list[KoriTestActionReport]
    current_action: Optional[KoriTestAction] = field(init=False, default=None)
    current_fn: Optional[Callable] = field(init=False, default=None)
    _stdout: str = ""

    @property
    def current_test(self) -> KoriTest:
        return self._current_test

    def next_action(self, *, force_next: bool = False):
        # if the last action was persistent, don't advance the iterator and return it instead
        if not force_next and len(self.test_report) > 0 and not self.test_report[-1].action_result.is_done:
            self.test_report.pop()
            return self.current_action
        try:
            next_action = next(self.iter_actions)
        except StopIteration:
            return None
        else:
            self.current_action = next_action
            return next_action

    def peek_next_action(self):
        """Returns the next action without consuming it"""
        next_action = next(self.iter_actions)
        self.iter_actions = iter([next_action, *self.iter_actions])
        return next_action

    def read_stdout(self):
        """Returns stdout and resets it in the ctx"""
        stdout = self._stdout
        self._stdout = ""
        return stdout

    def peek_stdout(self):
        """Returns stdout"""
        return self._stdout

    def write_stdout(self, s: str):
        self._stdout += s


# ############################# Types ############################# #

KoriTestOutcome: TypeAlias = Literal["Success", "Failure"]
KoriStateOnFail: TypeAlias = Literal["Warning", "Error"]


@dataclass
class KoriTestSubAction:
    action_name: str
    action: Callable[[KoriTestCtx], KoriTestState]
    action_args: list[Any] = field(default_factory=list)
    state_on_fail: KoriStateOnFail = field(default=None, init=False)

    def warn_on_fail(self):
        self.state_on_fail = "Warning"
        return self

    def err_on_fail(self):
        self.state_on_fail = "Error"
        return self


@dataclass
class KoriTestActionResult:
    result_state: KoriTestState
    function_result: Any = field(default=None)
    is_done: bool = field(default=True, kw_only=True)
    end_test: bool = field(default=False, kw_only=True)

    # does the action again, but calls ctx.next_action(force_next=True) before doing so
    redo_with_next_action: bool = field(default=False, kw_only=True)

    @classmethod
    def success(cls, function_result: Any, *, end_test: bool = False,
                is_done: bool = True, redo_with_next_action: bool = False):
        return cls(KoriTestState.success(),
                   function_result=function_result,
                   end_test=end_test,
                   is_done=is_done,
                   redo_with_next_action=redo_with_next_action)

    @classmethod
    def fail(cls, fail: KoriTestFail, function_result: Any, *, end_test: bool = False,
             is_done: bool = True, redo_with_next_action: bool = False):
        return cls(KoriTestState.fail(fail),
                   function_result=function_result,
                   end_test=end_test,
                   is_done=is_done,
                   redo_with_next_action=redo_with_next_action)


@dataclass(frozen=True, kw_only=True)
class KoriTestActionReport:
    action_name: str
    result_state: KoriTestState
    action_result: KoriTestActionResult
    action_args: list[Any]
    fn_name: str
    fn_args: list[Any]
    fn_kwargs: dict[str, Any]
    sub_actions_reports: list[KoriTestSubActionReport]


@dataclass(frozen=True, kw_only=True)
class KoriTestSubActionReport:
    action_name: str
    result_state: KoriTestState
    action_args: list[Any]


# ############################# Errors & Warnings ############################# #


class _KoriExitTest(BaseException):
    def __init__(self):
        super().__init__()


class _KoriPythonError(BaseException):
    def __init__(self, msg: str):
        super().__init__(msg)


class _KoriNoMoreTests(BaseException):
    def __init__(self):
        super().__init__()


class KoriTestFail:
    def __init__(self, name: str, expected: Any, actual: Any):
        self.expected = expected
        self.actual = actual
        self.name = name

    @property
    def message(self):
        return f"Expected:{self.expected!r}, found:{self.actual!r}"


class KoriTestError(KoriTestFail):
    def __init__(self, name: str, expected: Any, actual: Any):
        super().__init__(name, expected, actual)


class KoriTestWarning(KoriTestFail):
    def __init__(self, name: str, expected: Any, actual: Any):
        super().__init__(name, expected, actual)


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
        if is_sub_action:
            state = test_report.result_state
        else:
            state = test_report.action_result.result_state
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

    def _format_test_group_result(self, test_result: KoriTestGroupResult):
        name = test_result.name
        test_results = test_result.test_results
        final_result = test_result.final_state
        formatted_result = f"{name}: {final_result.outcome + ' with warnings' * final_result.has_warning()}\n\t"
        for result in test_results:
            formatted_result += "\n\t\t".join(self._format_test_reports(result).split("\n"))
        return formatted_result

    def _format_test_reports(self, test_result: KoriTestResult | KoriTestGroupResult):
        if isinstance(test_result, KoriTestGroupResult):
            return self._format_test_group_result(test_result)

        name = test_result.name
        test_reports = test_result.test_reports
        final_result = test_result.final_state
        formatted_result = f"{name}: {final_result.outcome + ' with warnings' * final_result.has_warning()}\n"
        for report in test_reports:
            action_formatted = self._format_test_report(report)
            sub_action_formatted = "".join(map(self._format_test_report, report.sub_actions_reports))
            formatted_result += action_formatted + sub_action_formatted

        formatted_result += "\n\tnot called: \n\t" * (len(test_result.not_called) > 0) + "".join(
            f"{action.action_name!r}, " + "\n\t" * (i % 5 == 0)
            for i, action in enumerate(test_result.not_called, start=1)
        ) + "\n"

        if test_result.error is not None:
            err_msg = str(test_result.error).replace("\n", "\n\t")
            formatted_result += f"\n\t❌PYTHON ERROR❌: {err_msg}\n"
        return formatted_result

    def _format_success_rate(self) -> str:
        nb_test = len(self.test_suite_result.test_results)
        success_rate = {
            "Successes": sum(
                1 for test_result in self.test_suite_result.test_results if test_result.final_state.is_success()
            ),
            "Complete Successes": sum(
                1 for test_result in self.test_suite_result.test_results if
                test_result.final_state.is_complete_success()
            ),
            "Failures": sum(
                1 for test_result in self.test_suite_result.test_results if test_result.final_state.is_failure()
            )
        }
        return "\t" + "\n\t".join(f"{name}: {result} / {nb_test}" for name, result in success_rate.items())

    def format_test_results(self) -> str:
        formated_test_reports = "\n".join(map(self._format_test_reports, self.test_suite_result.test_results))

        return f"Équipe: {', '.join(self.test_suite_result.team)}\n\nSummary:\n{self._format_success_rate()}\n" \
               f"\n{formated_test_reports}\n"


def flatten(ls: list):
    result = []
    for l in ls:
        if isinstance(l, list):
            result += flatten(l)
        else:
            result.append(l)
    return result
