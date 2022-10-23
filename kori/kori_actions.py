# ############################# Kori test builtins ############################# #
import re
from typing import Any, Callable, Optional, overload, TypeAlias

from kori.kori import KoriTestAction, KoriTestSubAction, KoriTestState, KoriTestError, KoriTestCtx, \
    KoriTestActionResult, KoriTestFail

StrTest = str | re.Pattern | list[str]


def remove_ignored(s: str | re.Pattern, ignored: str):
    s_is_regex = isinstance(s, re.Pattern)
    if isinstance(ignored, re.Pattern):
        if s_is_regex:
            s = re.compile("".join(ignored.split(s.pattern)), s.flags | re.I)
        else:
            s = "".join(ignored.split(s))
    else:
        if s_is_regex:
            s = re.compile(s.pattern.replace(ignored, ""), s.flags | re.I)
        else:
            s = s.replace(s, "")
    return s


def str_match(expected: StrTest, actual: str, ignore: str | re.Pattern = re.compile("\n| *")) -> \
        tuple[bool, tuple[str, str]]:
    original_expected = expected
    original_actual = actual
    actual = remove_ignored(actual, ignore)

    if isinstance(expected, list):
        expected = [remove_ignored(s, ignore) for s in expected]
        matches = all(
            (s.lower() in actual.lower()) if isinstance(s, str) else s.search(actual) is not None for s in expected)
        return matches, (original_expected, original_actual)

    expected = remove_ignored(expected, ignore)

    expected_is_regex = isinstance(expected, re.Pattern)
    if expected_is_regex:
        return expected.search(actual) is not None, (original_expected.pattern, original_actual)

    # expected_message is not a regex
    return expected == actual, (original_expected, original_actual)


def one_of(*sub_actions: KoriTestSubAction) -> KoriTestSubAction:
    def inner_one_of(ctx: KoriTestCtx):
        for sub_action in sub_actions:
            result = sub_action.action(ctx)
            if result.is_success():
                return result
        return KoriTestState.fail(KoriTestFail("one_of", "One of the action to match", "None have matched"))

    return KoriTestSubAction("one_of", inner_one_of, action_args=[sub_action.action_name for sub_action in sub_actions])


def every(*sub_actions: KoriTestSubAction) -> KoriTestSubAction:
    def inner_one_of(ctx: KoriTestCtx):
        for sub_action in sub_actions:
            result = sub_action.action(ctx)
            if result.is_failure():
                break
        else:
            return KoriTestState.success()
        return KoriTestState.fail(KoriTestFail("every", "All actions must match", "At least one did not match"))

    return KoriTestSubAction("one_of", inner_one_of, action_args=[sub_action.action_name for sub_action in sub_actions])


def when(action: KoriTestAction, sub_action: KoriTestSubAction) -> KoriTestAction:
    return action.also(sub_action)


def prints(expected_message: StrTest, ignore: StrTest = re.compile(r"\n| *")) -> KoriTestAction:
    def inner_prints(_ctx: KoriTestCtx, *args, **kwargs) -> KoriTestActionResult:
        actual_sep = kwargs.get("sep", " ")
        actual_end = kwargs.get("end", "\n")
        actual = actual_sep.join(map(str, args)) + actual_end

        matches, expected_actual = str_match(expected_message, actual, ignore)

        if not matches:
            return KoriTestActionResult.fail(KoriTestFail("printed message did not match", *expected_actual), None)
        return KoriTestActionResult.success(None)

    return KoriTestAction("prints", inner_prints, [print], action_args=[expected_message, ignore])


def asks_for_input(expected_prompt: StrTest, injected_value: str) -> KoriTestAction:
    def inner_asks_for_input(_ctx: KoriTestCtx, prompt: str = "") -> KoriTestActionResult:
        matches, expected_actual = str_match(expected_prompt, prompt)
        if not matches:
            return KoriTestActionResult.fail(KoriTestFail("asks_for_input", *expected_actual), injected_value)

        return KoriTestActionResult.success(injected_value)

    return KoriTestAction("asks_for_input", inner_asks_for_input, [input],
                          action_args=[expected_prompt, injected_value])


def simulate_randint(expected_range: range, result: int) -> KoriTestAction:
    from random import randint

    def inner_simulate_randint(_ctx: KoriTestCtx, *args, **_kwargs) -> KoriTestActionResult:
        start, stop = args
        if expected_range.start != start or expected_range.stop != stop:
            return KoriTestActionResult.fail(KoriTestFail(
                "Invalid range",
                f"start={expected_range.start}, stop={expected_range.stop}",
                f"{start=}, {stop=}"
            ), result)

        return KoriTestActionResult.success(result)

    return KoriTestAction("simulate_randint", inner_simulate_randint, [randint], action_args=[])


@overload
def capture_stdout_until_input(returned: str) -> KoriTestAction:
    ...


@overload
def capture_stdout_until_input(returned: Callable[[KoriTestCtx], str]) -> KoriTestAction:
    ...


@overload
def capture_stdout_until_input(returned: None) -> KoriTestAction:
    ...


def capture_stdout_until_input(returned):
    def inner_capture_stdout_until(ctx: KoriTestCtx, *args, **kwargs) -> KoriTestActionResult:
        fn_name = ctx.current_fn.__name__

        if fn_name == "print":
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            ctx.write_stdout(sep.join(map(str, args)) + end)
            return KoriTestActionResult.success(None, is_done=False)

        elif fn_name == "input":
            if returned is None:
                return KoriTestActionResult.success(None, redo_with_next_action=True)
            result: str = returned if isinstance(returned, str) else returned(ctx)
            ctx.write_stdout(args[0] if len(args) > 0 else "")
            return KoriTestActionResult.success(result)

        else:
            return KoriTestActionResult.fail(KoriTestFail("capture_stdout_until_input", "Unreachable", "Reached"), None)

    return KoriTestAction("capture_stdout_until_input",
                          inner_capture_stdout_until,
                          [print, input],
                          action_args=[returned])


def simulate_input(injected_value: str):
    def inner_simulate_input(_ctx: KoriTestCtx, _prompt: str = "") -> KoriTestActionResult:
        return KoriTestActionResult.success(injected_value)

    return KoriTestAction("simulates_input", inner_simulate_input, [input], action_args=[injected_value])


def ignore_rest():
    def ignore(_ctx: KoriTestCtx, *_args, **_kwargs) -> KoriTestActionResult:
        return KoriTestActionResult.success(None, end_test=True)

    return KoriTestAction("ignore_rest", ignore, None)


def ends():
    def inner_ends(ctx: KoriTestCtx, *_args, **_kwargs) -> KoriTestActionResult:
        fn_name = ctx.current_fn.__name__
        return KoriTestActionResult.fail(
            KoriTestFail("ends", "No more tests", f"Called action related function: {fn_name!r}"),
            None,
            end_test=True
        )

    return KoriTestAction("ends", inner_ends, None)


# ############################# Kori sub actions ############################# #


def assert_var_equals(var_name: str, expected_value: Any):
    def inner(ctx: KoriTestCtx) -> KoriTestState:
        try:
            actual_value = eval(var_name, ctx.globals_, ctx.locals_)
        except NameError as e:
            return KoriTestState.err(
                KoriTestError(f"Var {var_name!r} not defined",
                              expected_value, str(e))
            )
        else:
            if actual_value != expected_value:
                return KoriTestState.err(
                    KoriTestError(f"Invalid value for variable {var_name!r}",
                                  expected_value, actual_value)
                )
            return KoriTestState.success()

    return KoriTestSubAction(action_name="assert_var_equals", action=inner, action_args=[var_name, expected_value])


def assert_stdout_matches(expected_stdout: StrTest, ignore: StrTest = re.compile(r"\n| *")):
    def inner_assert_stdout_matches(ctx: KoriTestCtx) -> KoriTestState:
        stdout = ctx.read_stdout()

        matches, expected_actual = str_match(expected_stdout, stdout, ignore)

        if not matches:
            return KoriTestState.err(KoriTestError("assert_stdout_matches", *expected_actual))

        return KoriTestState.success()

    return KoriTestSubAction("assert_stdout_matches", inner_assert_stdout_matches, [expected_stdout, ignore])


def assert_stdout_contains(*sub_str: StrTest):
    def inner_assert_stdout_matches(ctx: KoriTestCtx) -> KoriTestState:
        stdout = ctx.read_stdout()

        if not all((s in stdout) if isinstance(s, str) else s.match(stdout) is not None for s in sub_str):
            return KoriTestState.err(KoriTestError("assert_stdout_contains", f"{sub_str!r} in printed message", stdout))

        return KoriTestState.success()

    return KoriTestSubAction("assert_stdout_contains", inner_assert_stdout_matches, [sub_str])


def fail_and_stop():
    def inner(_ctx: KoriTestCtx):
        return KoriTestState.err(KoriTestError(
            name="Test should fail",
            expected="Not to not fail",
            actual="Test was failed and stopped"
        ))

    return KoriTestSubAction(action_name="abort_test", action=inner)
